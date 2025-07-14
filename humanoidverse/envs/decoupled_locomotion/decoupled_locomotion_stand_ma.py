import numpy as np

# from isaacgym.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch

from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi

from humanoidverse.envs.env_utils.visualization import Point

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

from humanoidverse.envs.locomotion.locomotion_ma import LeggedRobotLocomotion

from loguru import logger

import trimesh


def quat_rotate(q, v):
    """
    Rotate vector(s) v by quaternion(s) q.
    q: (B, 4), where q = [w, x, y, z]
    v: (B, 3)
    """
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return v + 2 * (q[:, :1] * uv + uuv)


def quat_rotate_inverse(q, v):
    """
    Rotate vector(s) v by the inverse of quaternion(s) q.
    """
    q_conj = q.clone()
    q_conj[:, 1:] *= -1
    return quat_rotate(q_conj, v)


def torch_rand_float(lower, upper, shape, device=None):
    """
    Generate uniform random floats between lower and upper.
    """
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def get_euler_xyz_in_tensor(q):
    """
    Convert quaternion (w, x, y, z) to Euler angles (XYZ order).
    Input: q (..., 4)
    Output: angles (..., 3)
    """
    w, x, y, z = q.unbind(-1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    Y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = torch.atan2(t3, t4)

    return torch.stack([X, Y, Z], dim=-1)


DEBUG = False


class LeggedRobotDecoupledLocomotionStance(LeggedRobotLocomotion):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self._motion_lib = None
        self._init_motion_lib()
        self._init_motion_extend()
        self._init_tracking_config()

        self.init_done = True
        self.debug_viz = True
        self.num_upper_dofs = self.config.robot.upper_body_actions_dim
        self.num_lower_dofs = self.config.robot.lower_body_actions_dim
        self.action_scale_upper_body = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.stand_prob = self.config.stand_prob
        self.tapping_in_place_prob = self.config.tapping_in_place_prob
        self.fix_waist_yaw = self.config.fix_waist_yaw
        self.fix_waist_roll = self.config.fix_waist_roll
        self.fix_waist_pitch = self.config.fix_waist_pitch

        self.min_feet_height = torch.tensor((self.num_envs, 2), dtype=torch.float, device=self.device)

        if not self.config.obs.add_noise:
            self.config.obs.noise_scales = {
                key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
            }

        self.lower_left_dofs_idx_no = self.config.robot.symmetric_dofs_idx.get("lower_left_dofs_idx_no", [])
        self.lower_right_dofs_idx_no = self.config.robot.symmetric_dofs_idx.get("lower_right_dofs_idx_no", [])
        self.lower_left_dofs_idx_op = self.config.robot.symmetric_dofs_idx.get("lower_left_dofs_idx_op", [])
        self.lower_right_dofs_idx_op = self.config.robot.symmetric_dofs_idx.get("lower_right_dofs_idx_op", [])

    def _init_motion_lib(self):
        if self.config.rewards.fix_upper_body:
            return
        self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        if self.is_evaluating:
            self._motion_lib.load_motions(random_sample=False)

        else:
            self._motion_lib.load_motions(random_sample=True)
        # res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=self.env_origins)
        if not self.config.robot.motion.reverse_motion: self._resample_motion_times(torch.arange(self.num_envs))
        self.motion_dt = self._motion_lib._motion_dt
        self.motion_start_idx = 0
        if self._motion_lib.standardize_motion_length and self.config.termination.terminate_when_motion_end:
            self.max_episode_length_s = self._motion_lib.standardize_motion_length_value
            self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in
                                       self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in
                                  self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in
                                  self.config.robot.motion.upper_body_link]
            self.pelvis_id = self.simulator._body_list.index(self.config.robot.motion.pelvis_link)
        if "hips_link" in self.config.robot.motion:
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in
                                self.config.robot.motion.hips_link]  # Yuanhang: -1 for the base link (pelvis)
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)

    def _init_motion_extend(self):
        if "extend_config" in self.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.robot.motion.extend_config:
                extend_parent_ids.append(self.simulator._body_list.index(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)

            self.marker_coords = torch.zeros(self.num_envs,
                                             self.num_bodies + self.num_extend_bodies,
                                             3,
                                             dtype=torch.float,
                                             device=self.device,
                                             requires_grad=False)  # extend
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3,
                                                   dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3,
                                                   dtype=torch.float, device=self.device, requires_grad=False)

    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 5), dtype=torch.float32, device=self.device
        )
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.command_ranges = self.config.locomotion_command_ranges
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                              requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.ref_upper_dof_pos = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, \
                                             dtype=torch.float32, device=self.device, requires_grad=False)
        self.episode_motion_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
        self.tapping_in_place = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device,
                                            requires_grad=False)
        self.fix_waist_yaw_range = self.config.fix_waist_yaw_range
        self.fix_waist_pitch_range = self.config.fix_waist_pitch_range
        self.fix_waist_roll_range = self.config.fix_waist_roll_range
        self.zero_fix_waist_roll = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                               requires_grad=False)
        self.zero_fix_waist_pitch = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                                requires_grad=False)
        self.zero_fix_waist_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                              requires_grad=False)
        self.fixed_waist_yaw_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                               requires_grad=False)
        self.fixed_waist_pitch_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                                 requires_grad=False)
        self.fixed_waist_roll_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device,
                                                requires_grad=False)
        self.apply_waist_yaw_only_when_stance = self.config.apply_waist_yaw_only_when_stance
        self.apply_waist_roll_only_when_stance = self.config.apply_waist_roll_only_when_stance
        self.apply_waist_pitch_only_when_stance = self.config.apply_waist_pitch_only_when_stance
        self.apply_waist_roll_pitch_only_when_stance = 1 if (
                    self.config.apply_waist_roll_only_when_stance and self.config.apply_waist_pitch_only_when_stance) else 0

        if self.config.rewards.get("upper_body_motion_scale_curriculum", False):
            self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                      requires_grad=False) * self.config.rewards.upper_body_motion_initial_scale
        else:
            self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        if self.config.rewards.get("command_height_scale_curriculum", False):
            self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                   requires_grad=False) * self.config.rewards.command_height_scale_initial_scale
        else:
            self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                   requires_grad=False)

    def _init_domain_rand_buffers(self):
        super()._init_domain_rand_buffers()
        self.ref_episodic_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                               requires_grad=False)

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Push the robots randomly
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(
                as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0],
                                                                     self.config.domain_rand.push_interval_s[1],
                                                                     (len(push_robot_env_ids),), device=self.device,
                                                                     requires_grad=False)
            self._push_robots(push_robot_env_ids)
        # Update locomotion commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(
                self.config.locomotion_command_resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(
            0.5 * wrap_to_pi(self.commands[:, 3] - heading),
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1]
        )
        # only apply the velocity command if it is tapping command or not tapping in place
        # print("commands: ", self.commands)
        # print("tapping_in_place: ", self.tapping_in_place)
        # import pdb; pdb.set_trace()
        self.commands[:, 0] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        self.commands[:, 1] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        self.commands[:, 2] *= (self.commands[:, 4] * self.tapping_in_place[:, 0])
        # print("commands: ", self.commands)
        # If fixed, no need to update the upper body motion
        if self.config.rewards.fix_upper_body:
            return
        # Resample/Update upper body motions
        if self.config.resample_motion_when_training:
            if self.common_step_counter % self.resample_time_interval == 0:
                logger.info(f"Resampling motion at step {self.common_step_counter}")
                self.resample_motion()
        self.motion_len = self._motion_lib.get_motion_length(self.motion_ids)
        motion_times = (self.episode_motion_length) * self.dt + self.motion_start_times  # current frame
        if self.config.robot.motion.reverse_motion:
            # Yichao: Here we consider a full video contains forward then its reverse motions, so double the length
            # reverse motions are addressed at get_motion_state in motion_lib
            env_ids = torch.where(motion_times > 2 * self.motion_len)[0]  # check if the motion is finished
        else:
            env_ids = torch.where(motion_times > self.motion_len)[0]  # check if the motion is finished
            self._resample_motion_times(
                env_ids)  # Yuanhang: resample the motion start times only when non-reverse motion
        self.episode_motion_length[env_ids] = 0  # reset the episode motion length

    def _reset_tasks_callback(self, env_ids):
        super()._reset_tasks_callback(env_ids)
        self.left_feet_height[env_ids] *= 0
        self.right_feet_height[env_ids] *= 0

    def next_task(self):
        # This function is only called when evaluating
        if self.config.rewards.fix_upper_body:
            return
        self.motion_start_idx += self.num_envs
        self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        self.reset_all()

    def resample_motion(self):
        self._motion_lib.load_motions(random_sample=True)
        # Yuanhang: do not reset the envs, otherwise the episode lengths conflict with the ppo buffer
        # self.reset_envs_idx(torch.arange(self.num_envs, device=self.device))

    def _resample_motion_times(self, env_ids):
        if len(env_ids) == 0:
            return
        self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
        if self.is_evaluating:
            self.motion_start_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

    def _update_episode_motion_length(self):
        env_ids_stance = torch.where(self.commands[:, 4] == 0)[0]
        self.episode_motion_length[env_ids_stance] += 1

    def _post_physics_step(self):
        super()._post_physics_step()
        self._update_episode_motion_length()

    def _pre_compute_observations_callback(self, debug=DEBUG):
        super()._pre_compute_observations_callback()
        if self.config.rewards.fix_upper_body:
            self.ref_upper_dof_pos *= 0.0
            return
        # Get the reference upper body joint positions
        offset = self.env_origins
        # print("env_ids_stance: ", env_ids_stance)
        self.motion_times = (self.episode_motion_length + 1) * self.dt + self.motion_start_times  # next frames so +1
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=offset)

        # Update the upper body joint positions from motion library
        ref_joint_pos = motion_res["dof_pos"]  # [num_envs, num_dofs]
        if self.fix_waist_yaw and self.waist_yaw_dof_indice:
            ref_joint_pos[:, self.waist_yaw_dof_indice] = self.fixed_waist_yaw_pos
            if self.apply_waist_yaw_only_when_stance:
                ref_joint_pos[:, self.waist_yaw_dof_indice] *= (1 - self.commands[:, 4])
        if self.fix_waist_roll and self.waist_roll_dof_indice:
            ref_joint_pos[:, self.waist_roll_dof_indice] = self.fixed_waist_roll_pos
            if self.apply_waist_roll_only_when_stance:
                ref_joint_pos[:, self.waist_roll_dof_indice] *= (1 - self.commands[:, 4])
        if self.fix_waist_pitch and self.waist_pitch_dof_indice:
            ref_joint_pos[:, self.waist_pitch_dof_indice] = self.fixed_waist_pitch_pos
            if self.apply_waist_pitch_only_when_stance:
                ref_joint_pos[:, self.waist_pitch_dof_indice] *= (1 - self.commands[:, 4])
        self.ref_upper_dof_pos = ref_joint_pos[:, self.upper_dof_indices]  # [num_envs, upper_body_actions_dim]
        # Apply upper body action scale
        self.ref_upper_dof_pos *= self.action_scale_upper_body

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = self.config.robot.control.action_scale * actions
        control_type = self.config.robot.control.control_type
        if control_type == "P":
            torques = self._kp_scale * self.p_gains * (
                        actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains * self.simulator.dof_vel
        elif control_type == "V":
            torques = self._kp_scale * self.p_gains * (
                        actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains * (
                                  self.simulator.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(
                torques) * 2. - 1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits

        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques

    def _resample_commands(self, env_ids):
        if self._motion_lib and not self.config.robot.motion.reverse_motion: self._resample_motion_times(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0],
                                                     self.command_ranges["lin_vel_x"][1], (len(env_ids), 1),
                                                     device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0],
                                                     self.command_ranges["lin_vel_y"][1], (len(env_ids), 1),
                                                     device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                     self.command_ranges["heading"][1], (len(env_ids), 1),
                                                     device=self.device).squeeze(1)

        # Sample the tapping or stand command with a probability
        self.commands[env_ids, 4] = (torch.rand(len(env_ids), device=self.device) > self.stand_prob).float()
        # Sample the tapping in place command with a probability
        self.tapping_in_place[env_ids, 0] = (
                    torch.rand(len(env_ids), device=self.device) > self.tapping_in_place_prob).float()
        self.commands[env_ids, 0] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 1] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 2] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # Sample the waist yaw, pitch and roll fixed dof pos
        self.zero_fix_waist_yaw[env_ids] = (
                    torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_yaw_prob).float()
        self.zero_fix_waist_roll[env_ids] = (
                    torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_roll_prob).float()
        self.zero_fix_waist_pitch[env_ids] = (
                    torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_pitch_prob).float()
        if self.fix_waist_yaw:
            self.fixed_waist_yaw_pos[env_ids] = torch_rand_float(self.fix_waist_yaw_range[0],
                                                                 self.fix_waist_yaw_range[1], (len(env_ids), 1),
                                                                 device=self.device).squeeze(1) * \
                                                self.zero_fix_waist_yaw[env_ids]
        if self.fix_waist_pitch:
            self.fixed_waist_pitch_pos[env_ids] = torch_rand_float(self.fix_waist_pitch_range[0],
                                                                   self.fix_waist_pitch_range[1], (len(env_ids), 1),
                                                                   device=self.device).squeeze(1) * \
                                                  self.zero_fix_waist_pitch[env_ids]
        if self.fix_waist_roll:
            self.fixed_waist_roll_pos[env_ids] = torch_rand_float(self.fix_waist_roll_range[0],
                                                                  self.fix_waist_roll_range[1], (len(env_ids), 1),
                                                                  device=self.device).squeeze(1) * \
                                                 self.zero_fix_waist_roll[env_ids]

    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()

        self.commands = torch.zeros((self.num_envs, 5), dtype=torch.float32, device=self.device)
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)

    def _draw_debug_vis(self, debug=DEBUG):
        return
        if not self.is_evaluating and not debug:
            return
        else:
            self.simulator.gym.clear_lines(self.viewer)
            self._refresh_sim_tensors()

            for env_id in range(self.num_envs):
                for pos_id, pos_joint in enumerate(self.marker_coords[env_id]):  # idx 0 torso (duplicate with 11)
                    # Draw the tracking markers for the whole body
                    if self.config.robot.motion.visualization.customize_color:
                        color_inner = self.config.robot.motion.visualization.marker_joint_colors[
                            pos_id % len(self.config.robot.motion.visualization.marker_joint_colors)]
                    else:
                        color_inner = (0.3, 0.3, 0.3)
                    color_inner = tuple(color_inner)
                    sphere_geom_marker = gymutil.WireframeSphereGeometry(0.04, 20, 20, None, color=color_inner)
                    sphere_pose = gymapi.Transform(gymapi.Vec3(pos_joint[0], pos_joint[1], pos_joint[2]), r=None)
                    gymutil.draw_lines(sphere_geom_marker, self.simulator.gym, self.viewer, self.simulator.envs[env_id],
                                       sphere_pose)
                    # Draw the tracking lines for the extended body only
                    if pos_id in self.motion_tracking_id:
                        color_schems = (0.851, 0.144, 0.07)
                        start_point = self._rigid_body_pos_extend[env_id, pos_id]
                        end_point = pos_joint
                        line_width = 0.03
                        for _ in range(50):
                            gymutil.draw_line(Point(start_point + torch.rand(3, device=self.device) * line_width),
                                              Point(end_point + torch.rand(3, device=self.device) * line_width),
                                              Point(color_schems),
                                              self.simulator.gym, self.viewer, self.simulator.envs[env_id])

    ################ Curriculum #################
    def _update_upper_body_motion_scale_curriculum(self, env_ids):
        """
        Update the upper body motion scale based on the episode length for each environment.
        Returns:
            None
        """
        if self.config.rewards.fix_upper_body:
            return
        env_ids_scale_up_mask = self.episode_length_buf[
                                    env_ids] > self.config.rewards.upper_body_motion_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = self.episode_length_buf[
                                      env_ids] < self.config.rewards.upper_body_motion_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.action_scale_upper_body[env_ids_scale_up] += self.config.rewards.upper_body_motion_scale_up
        self.action_scale_upper_body[env_ids_scale_down] -= self.config.rewards.upper_body_motion_scale_down
        # Clip the scale
        self.action_scale_upper_body[env_ids] = torch.clip(self.action_scale_upper_body[env_ids],
                                                           self.config.rewards.upper_body_motion_scale_min,
                                                           self.config.rewards.upper_body_motion_scale_max)

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Resets the environments with the given ids and optionally to the target states
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.config.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        if self.config.rewards.upper_body_motion_scale_curriculum:
            self._update_upper_body_motion_scale_curriculum(env_ids)
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)  # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()

    def _reset_dofs(self, env_ids, target_state=None):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        ## Lower body dof reset
        if target_state is not None:
            self.simulator.dof_pos[env_ids.unsqueeze(1), self.lower_dof_indices] = target_state[..., 0]
            self.simulator.dof_vel[env_ids.unsqueeze(1), self.lower_dof_indices] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids.unsqueeze(1), self.lower_dof_indices] = \
                self.default_dof_pos[:, self.lower_dof_indices] * \
                torch_rand_float(0.5, 1.5, (len(env_ids), self.config.robot.lower_body_actions_dim),
                                 device=str(self.device))
            self.simulator.dof_vel[env_ids.unsqueeze(1), self.lower_dof_indices] = 0.
        if self.config.rewards.fix_upper_body:
            return
        ## Upper body dof reset
        motion_times = (self.episode_motion_length) * self.dt + self.motion_start_times  # current frame
        offset = self.env_origins
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, motion_times, offset=offset)
        self.simulator.dof_pos[env_ids.unsqueeze(1), self.upper_dof_indices] = \
            motion_res['dof_pos'][env_ids.unsqueeze(1), self.upper_dof_indices]
        self.simulator.dof_vel[env_ids.unsqueeze(1), self.upper_dof_indices] = \
            motion_res['dof_vel'][env_ids.unsqueeze(1), self.upper_dof_indices]

    ########################### GAIT REWARDS ###########################

    ########################### FEET REWARDS ###########################

    ######################## LIMITS REWARDS #########################
    def _reward_limits_dof_pos(self):
        # Penalize dof positions too close to the limit (lower body only)
        out_of_limits = -(self.simulator.dof_pos[:, self.lower_dof_indices] - \
                          self.simulator.dof_pos_limits[self.lower_dof_indices, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos[:, self.lower_dof_indices] - \
                          self.simulator.dof_pos_limits[self.lower_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_dof_vel(self):
        # Penalize dof velocities too close to the limit (lower body only)
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel[:, self.lower_dof_indices]) - \
                          self.simulator.dof_vel_limits[self.lower_dof_indices, 1]).clip(min=0., max=1.), dim=1)

    def _reward_limits_torque(self):
        # penalize torques too close to the limit (lower body only)
        return torch.sum((torch.abs(self.torques[:, self.lower_dof_indices]) - \
                          self.torque_limits[self.lower_dof_indices, 1]).clip(min=0.), dim=1)

    ######################### PENALTY REWARDS #########################
    def _reward_penalty_negative_knee_joint(self):
        # Penalize negative knee joint angles (lower body only)
        return torch.sum((self.simulator.dof_pos[:, self.knee_dof_indices] < self.knee_joint_min_threshold).float(),
                         dim=1)

    def _reward_penalty_torques(self):
        # Penalize torques (lower body only)
        return torch.sum(torch.square(self.torques[:, self.lower_dof_indices]), dim=1)

    def _reward_penalty_dof_vel(self):
        # Penalize dof velocities (lower body only)
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_dof_indices]), dim=1)

    def _reward_penalty_dof_acc(self):
        # Penalize dof accelerations (lower body only)
        return torch.sum(torch.square((self.last_dof_vel[:, self.lower_dof_indices] - \
                                       self.simulator.dof_vel[:, self.lower_dof_indices]) / self.dt), dim=1)

    def _reward_penalty_action_rate(self):
        # Penalize changes in actions (lower body only)
        return torch.sum(torch.square(self.last_actions[:, self.lower_dof_indices] - \
                                      self.actions[:, self.lower_dof_indices]), dim=1)

    def _reward_penalty_feet_swing_height(self):
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        # self.min_feet_height = torch.min(feet_height, self.min_feet_height)
        # print("min_feet_height: ", self.min_feet_height)
        # set to zero if not tappinging (standing)
        target_height = self.config.rewards.feet_height_target * self.commands[:, 4:5] + \
                        self.config.rewards.feet_height_stand * (1.0 - self.commands[:, 4:5])
        height_error = torch.square(feet_height - target_height) * ~contact
        return torch.sum(height_error, dim=(1))

    def _reward_penalty_torso_orientation(self):
        # Penalize non flat torso orientation
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        projected_gravity_torso = quat_rotate_inverse(torso_quat, self.gravity_vec)
        return torch.abs(projected_gravity_torso[:, 1]) * (1.0 - self.commands[:, 4]) * (
                    1.0 - self.zero_fix_waist_roll) + \
            torch.square(projected_gravity_torso[:, 0]) * (1.0 - self.commands[:, 4]) * (
                        1.0 - self.zero_fix_waist_pitch) + \
            torch.sum(torch.square(projected_gravity_torso[:, :2]), dim=1) * self.commands[:,
                                                                             4] * self.apply_waist_roll_pitch_only_when_stance

    def _reward_penalty_feet_height(self):
        # Penalize base height away from target
        feet_height = self.simulator._rigid_body_pos[:, self.feet_indices, 2]
        dif = torch.abs(feet_height - self.config.rewards.feet_height_target)
        dif = torch.min(dif, dim=1).values  # [num_env], # select the foot closer to target
        return torch.clip(dif - 0.02, min=0.) * self.commands[:,
                                                4]  # target - 0.02 ~ target + 0.02 is acceptable, apply only when tapping

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(2):  # left and right feet
            is_stance = (self.leg_phase[:, i] < 0.55) | (self.commands[:, 4] == 0)
            contact = self.simulator.contact_forces[:, self.feet_indices[i], 2] > 1
            contact_reward = ~(contact ^ is_stance)
            contact_penalty = contact ^ is_stance
            # res += contact_reward
            res += contact_reward.int() - contact_penalty.int()
        return res

    def _reward_penalty_contact(self):
        # Initialize the penalty reward tensor
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Check if the agent is in stance mode (commands[:, 4] == 0)
        is_stance = (self.commands[:, 4] == 0)
        # Determine foot contact (contact force in Z-axis > 1 is considered ground contact)
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1
        # Count the number of feet in contact with the ground
        num_feet_on_ground = contact.sum(dim=1)
        # Penalize if any foot is off the ground when commands[:, 4] == 0 (stance mode)
        res[is_stance & (num_feet_on_ground < 2)] = 1.0
        # Penalize if both feet are on the ground when commands[:, 4] == 1 (walking mode)
        res[~is_stance & ((num_feet_on_ground == 2) | (num_feet_on_ground == 0))] = 1.0
        return res

    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        return torch.sum(torch.square(hip_pos), dim=1) * self.commands[:, 4]  # only apply when walking

    # def _reward_tapping_in_place(self):
    #     static_tapping_command = (torch.norm(self.commands[:, :2], dim=1) < 0.1) & (self.commands[:, 4] == 1)

    def _reward_alive(self):
        # Reward for staying alive
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    ######################### NOT USED REWARDS #########################
    def _reward_waist_joint_angle_freeze(self):
        # returns keep the upper body joint angles close to the default (NOT used)
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(
            self.simulator.dof_pos[:, self.waist_dof_indices] - self.default_dof_pos[:, self.waist_dof_indices])
        return torch.sum(deviation, dim=1)

    def _reward_penalty_stance_dof(self):
        # Penalize the lower body dof velocity
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_dof_indices]), dim=1) * (
                    1.0 - self.commands[:, 4])
        # Penalize the pelvis velocity
        return torch.sum(torch.abs(self.simulator._rigid_body_vel[:, self.pelvis_id]), dim=1) * (
                    1.0 - self.commands[:, 4])
        # Penalize dof pos change of the lower body
        return torch.sum(torch.square(self.last_dof_pos[:, self.lower_dof_indices] - \
                                      self.simulator.dof_pos[:, self.lower_dof_indices]), dim=1) * (
                    1.0 - self.commands[:, 4])

    def _reward_penalty_stance_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(
            self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:,
                                                                          self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_id]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - self.commands[:, 4])

    def _reward_penalty_stance_tap_feet(self):
        # Penalize the feet distance on the x axis of base frame
        feet_diff = torch.abs(
            self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] - self.simulator._rigid_body_pos[:,
                                                                          self.feet_indices[1], :3])
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_id]
        projected_feet_diff = quat_rotate_inverse(pelvis_quat, feet_diff)
        stance_tap = self.commands[:, 4] * (torch.abs(self.commands[:, 0]) > 0.0)
        return torch.abs(projected_feet_diff[:, 0]) * (1.0 - stance_tap)

    def _reward_penalty_stance_root(self):
        # Penalize the root position
        feet_mid_pos = (self.simulator._rigid_body_pos[:, self.feet_indices[0], :3] + self.simulator._rigid_body_pos[:,
                                                                                      self.feet_indices[1], :3]) / 2
        root_pos = self.simulator._rigid_body_pos[:, self.pelvis_id, :3]
        root_feet_diff = root_pos - feet_mid_pos
        pelvis_quat = self.simulator._rigid_body_rot[:, self.pelvis_id]
        projected_root_feet_diff = quat_rotate_inverse(pelvis_quat, root_feet_diff)
        # return torch.norm(projected_root_feet_diff[:, :2], dim=1) * (1.0 - self.commands[:, 4])
        return torch.abs(projected_root_feet_diff[:, 1]) * (1.0 - self.commands[:, 4])

    def _reward_penalty_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_vel = self.simulator._rigid_body_vel[:, self.feet_indices]
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=(1, 2))
        stance_envs_id = torch.where(self.commands[:, 4] == 0)[0]
        penalize[stance_envs_id] *= 10.0
        # print(f"base_height: {self.simulator.robot_root_states[:, 2]}")
        return penalize

    def _reward_penalty_stance_contact_no_ang_vel(self):
        # Penalize contact with no angular velocity
        contact = torch.norm(self.simulator.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        feet_ang_vel = self.simulator._rigid_body_ang_vel[:, self.feet_indices]
        contact_feet_ang_vel = feet_ang_vel * contact.unsqueeze(-1)
        penalize = torch.sum(torch.square(contact_feet_ang_vel[:, :, :3]), dim=(1, 2))
        penalize *= (1.0 - self.commands[:, 4])
        # print(f"base_height: {self.simulator.robot_root_states[:, 2]}")
        return penalize

    def _reward_penalty_stand_still(self):
        # Penalize standing still
        no_contacts = torch.sum(self.simulator.contact_forces[:, self.feet_indices, 2] < 0.1, dim=1) > 0
        return no_contacts.float() * (1.0 - self.commands[:, 4])

    def _reward_penalty_stance_symmetry(self):
        # TODO: Hardcoded
        diff_lower_body_dof_po_no = self.simulator.dof_pos[:, self.lower_left_dofs_idx_no] - self.simulator.dof_pos[:,
                                                                                             self.lower_right_dofs_idx_no]
        diff_lower_body_dof_pos_op = self.simulator.dof_pos[:, self.lower_left_dofs_idx_op] + self.simulator.dof_pos[:,
                                                                                              self.lower_right_dofs_idx_op]
        return torch.sum(torch.abs(diff_lower_body_dof_po_no) +
                         torch.abs(diff_lower_body_dof_pos_op), dim=1) * (1.0 - self.commands[:, 4])

    ######################### Phase Time #########################

    def _calc_phase_time(self):
        # Calculate the phase time
        episode_length_np = self.episode_length_buf.cpu().numpy()
        phase_time = (episode_length_np * self.dt + self.phi_offset) % self.T / self.T
        phase_time *= self.commands[:, 4].cpu().numpy()  # only apply when locomotion
        return phase_time

    ######################### Observations #########################

    def _get_obs_ref_upper_dof_pos(self):
        return self.ref_upper_dof_pos

    def _get_obs_actions(self, ):
        return self.actions[:, self.lower_dof_indices]

    def _get_obs_command_stand(self):
        return self.commands[:, 4:5]

    def _get_obs_base_orientation(self):
        return self.base_quat[:, 0:4]