import numpy as np
# from isaacgym.torch_utils import *

import torch
from isaac_utils.rotations import get_euler_xyz_in_tensor

from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_ma import LeggedRobotDecoupledLocomotionStance

from loguru import logger

def torch_rand_float(lower, upper, shape, device=None):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

def quat_rotate_inverse(q, v):
    # Rotate v by inverse of quaternion q
    q_conj = q.clone()
    q_conj[:, 1:] *= -1
    return quat_apply(q_conj, v)

def quat_apply(q, v):
    # Applies quaternion rotation: q * [0,v] * q_conj
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return v + 2 * (q[:, :1] * uv + uuv)

DEBUG = False
class LeggedRobotDecoupledLocomotionStanceHeightWBC(LeggedRobotDecoupledLocomotionStance):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.command_height_scale = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

    def _init_tracking_config(self):
        if "motion_tracking_link" in self.config.robot.motion:
            self.motion_tracking_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.motion_tracking_link]
        if "lower_body_link" in self.config.robot.motion:
            self.lower_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.lower_body_link]
        if "upper_body_link" in self.config.robot.motion:
            self.upper_body_id = [self.simulator._body_list.index(link) for link in self.config.robot.motion.upper_body_link]
            self.pelvis_id = self.simulator._body_list.index(self.config.robot.motion.pelvis_link)
        if "hips_link" in self.config.robot.motion:
            self.hips_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.hips_link] # Yuanhang: -1 for the base link (pelvis)
        if "waist_link" in self.config.robot.motion:
            self.waist_dof_id = [self.simulator._body_list.index(link) - 1 for link in self.config.robot.motion.waist_link]
        if self.config.resample_motion_when_training:
            self.resample_time_interval = np.ceil(self.config.resample_time_interval_s / self.dt)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.commands = torch.zeros(
            (self.num_envs, 9), dtype=torch.float32, device=self.device
        )
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.command_ranges = self.config.locomotion_command_ranges
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.motion_len = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.ref_upper_dof_pos = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, \
                                               dtype=torch.float32, device=self.device, requires_grad=False)
        self.episode_motion_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.tapping_in_place = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fix_waist_yaw_range = self.config.fix_waist_yaw_range
        self.fix_waist_pitch_range = self.config.fix_waist_pitch_range
        self.fix_waist_roll_range = self.config.fix_waist_roll_range
        self.zero_fix_waist_roll = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.zero_fix_waist_pitch = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.zero_fix_waist_yaw = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_yaw_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_pitch_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.fixed_waist_roll_pos = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device, requires_grad=False)
        self.apply_waist_yaw_only_when_stance = self.config.apply_waist_yaw_only_when_stance
        self.apply_waist_roll_only_when_stance = self.config.apply_waist_roll_only_when_stance
        self.apply_waist_pitch_only_when_stance = self.config.apply_waist_pitch_only_when_stance
        self.apply_waist_roll_pitch_only_when_stance = 1 if (self.config.apply_waist_roll_only_when_stance and self.config.apply_waist_pitch_only_when_stance) else 0

        # Upper body dof pos tracking termination buf
        self.far_upper_dof_pos_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # Residual upper body action
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)
        self.ref_pd_torques = torch.zeros(self.num_envs, self.config.robot.actions_dim, device=self.device, dtype=torch.float32)
        
        # Fix upper body prob
        self.fix_upper_body_prob = self.config.get("fix_upper_body_prob", 0.0)
        self.fix_upper_body = (torch.rand(self.num_envs, device=self.device) < self.fix_upper_body_prob).float()
        self.fix_upper_body_motion_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = actor_state["actions"]

        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.config.robot.control.action_scale
        ref_actions_scaled = actions_scaled.clone()
        if self.residual_upper_body_action:
            actions_scaled[:, self.upper_dof_indices] += (self.ref_upper_dof_pos - self.default_dof_pos[:, self.upper_dof_indices])
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            ref_actions_scaled[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            self.ref_pd_torques = self._kp_scale * self.p_gains*(ref_actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques
    
    def _pre_compute_observations_callback(self, debug=DEBUG):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        # print("self.base_lin_vel", self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        # print("self.base_ang_vel", self.base_ang_vel)
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.config.rewards.fix_upper_body:
            self.ref_upper_dof_pos *= 0.0
            return
        # Get the reference upper body joint positions
        offset = self.env_origins
        # print("env_ids_stance: ", env_ids_stance)
        # print("episode_length_buf: ", self.episode_length_buf)
        # self.motion_times = (self.episode_length_buf + 1) * self.dt + self.motion_start_times # next frames so +1
        self.motion_times = ((self.episode_length_buf + 1) * self.dt + self.motion_start_times) * (1 - self.fix_upper_body) + \
                            self.fix_upper_body_motion_times * self.fix_upper_body
        motion_res = self._motion_lib.get_motion_state(self.motion_ids, self.motion_times, offset=offset)
        
        # Update the upper body joint positions from motion library
        ref_joint_pos = motion_res["dof_pos"] # [num_envs, num_dofs]
        self.ref_body_pos_extend[:, :motion_res["rg_pos_t"].shape[1], :] = motion_res["rg_pos_t"] # [num_envs, 3]

        self.ref_upper_dof_pos = ref_joint_pos[:, self.upper_dof_indices] # [num_envs, upper_body_actions_dim]
        # Yuanhang: only test for evaluation
        # if self.is_evaluating:
        #     self.ref_upper_dof_pos[:, :] *= 0.0
        # print("waist yaw: ", self.ref_upper_dof_pos[:, 0])
        # print("waist roll: ", self.ref_upper_dof_pos[:, 1])
        # print("waist pitch: ", self.ref_upper_dof_pos[:, 2])
        # Apply upper body action scale
        self.ref_upper_dof_pos *= self.action_scale_upper_body


    def _resample_commands(self, env_ids):
        if self._motion_lib and not self.config.robot.motion.reverse_motion: self._resample_motion_times(env_ids)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=str(self.device)).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Sample the tapping or stand command with a probability
        self.commands[env_ids, 4] = (torch.rand(len(env_ids), device=self.device) > self.stand_prob).float()
        # Sample the tapping in place command with a probability
        self.tapping_in_place[env_ids, 0] = (torch.rand(len(env_ids), device=self.device) > self.tapping_in_place_prob).float()
        self.commands[env_ids, 0] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 1] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        self.commands[env_ids, 2] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        # Sample the waist yaw, pitch and roll fixed dof pos
        self.zero_fix_waist_yaw[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_yaw_prob).float()
        self.zero_fix_waist_roll[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_roll_prob).float()
        self.zero_fix_waist_pitch[env_ids] = (torch.rand(len(env_ids), device=self.device) > self.config.zero_fix_waist_pitch_prob).float()
        if self.apply_waist_yaw_only_when_stance:
            self.zero_fix_waist_yaw[env_ids] *= (1 - self.commands[env_ids, 4]) # only apply when stance
        if self.apply_waist_roll_only_when_stance:
            self.zero_fix_waist_roll[env_ids] *= (1 - self.commands[env_ids, 4])
        if self.apply_waist_pitch_only_when_stance:
            self.zero_fix_waist_pitch[env_ids] *= (1 - self.commands[env_ids, 4])
        self.commands[env_ids, 5] = torch_rand_float(self.fix_waist_yaw_range[0], self.fix_waist_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_yaw[env_ids]
        self.commands[env_ids, 6] = torch_rand_float(self.fix_waist_roll_range[0], self.fix_waist_roll_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_roll[env_ids]
        self.commands[env_ids, 7] = torch_rand_float(self.fix_waist_pitch_range[0], self.fix_waist_pitch_range[1], (len(env_ids), 1), device=self.device).squeeze(1) * self.zero_fix_waist_pitch[env_ids]
        # Sample the desired base height
        self.commands[env_ids, 8] = self.config.rewards.desired_base_height
        self.commands[env_ids, 8] += torch_rand_float(self.command_ranges["base_height"][0], 
                                                      self.command_ranges["base_height"][1], 
                                                      (len(env_ids), 1), device=self.device).squeeze(1) * \
                                                      (self.command_height_scale[env_ids]).squeeze(1) * \
                                                      (1.0 - self.commands[env_ids, 4]) # only apply the base height if standing
        # Sample the fixed upper body dof pos
        self.fix_upper_body[env_ids] = (torch.rand(len(env_ids), device=self.device) < self.fix_upper_body_prob).float()
        self.fix_upper_body_motion_times[env_ids] = torch.rand(len(env_ids), device=self.device) * self.max_episode_length_s
        
    def set_is_evaluating(self, command=None):
        super().set_is_evaluating()
        
        self.commands = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.commands[:, 8] = self.config.rewards.desired_base_height
        # Apply full upper body action scale
        self.action_scale_upper_body = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_height_scale = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # TODO: haotian: adding command configuration
        if command is not None:
            self.commands[:, :3] = torch.tensor(command).to(self.device)  # only set the first 3 commands

        # self.config.obs.noise_scales = {
        #     key: value * 0.0 for key, value in self.config.obs.noise_scales.items()
        # }
        # print("noise scales: ", self.config.obs.noise_scales)

    def _compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                        for key in self.reward_group_names}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            try:
                assert rew.shape[0] == self.num_envs
            except:
                import ipdb; ipdb.set_trace()
            # penalty curriculum
            if name in self.config.rewards.reward_penalty_reward_names:
                if self.config.rewards.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale
            self.rew_buf[self.reward_groups[name]] += rew
            self.episode_sums[name] += rew
        if self.config.rewards.only_positive_rewards:
            self.rew_buf = {key: torch.clamp(value, min=0.) for key, value in self.rew_buf.items()}
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            # Yuanhang [TODO]: Hardcode for the termination reward
            # self.rew_buf["upper_body"][self.far_upper_dof_pos_buf] += rew[self.far_upper_dof_pos_buf]
            # self.rew_buf["lower_body"][~self.far_upper_dof_pos_buf] += rew[~self.far_upper_dof_pos_buf]
            self.rew_buf[self.reward_groups["termination"]] += rew
            self.episode_sums["termination"] += rew

        if self.use_reward_penalty_curriculum:
            self.log_dict["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
        self.log_dict["average_episode_length"] = self.average_episode_length
    
    ################ Curriculum #################
    
    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.config.rewards.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > self.config.rewards.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.config.rewards.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.config.rewards.reward_min_penalty_scale, self.config.rewards.reward_max_penalty_scale)

    def _update_tasks_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # Push the robots randomly
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
        # Update locomotion commands
        if not self.is_evaluating:
            env_ids = (self.episode_length_buf % int(self.config.locomotion_command_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
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
        # motion_times = (self.episode_motion_length) * self.dt + self.motion_start_times # current frame
        # if self.config.robot.motion.reverse_motion:
        #     # Yichao: Here we consider a full video contains forward then its reverse motions, so double the length
        #     # reverse motions are addressed at get_motion_state in motion_lib
        #     env_ids = torch.where(motion_times > 2 * self.motion_len)[0] # check if the motion is finished
        # else: 
        #     env_ids = torch.where(motion_times > self.motion_len)[0] # check if the motion is finished
        #     self._resample_motion_times(env_ids) # Yuanhang: resample the motion start times only when non-reverse motion
        # self.episode_motion_length[env_ids] = 0 # reset the episode motion length

    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0

        self._update_reset_buf()
        self._update_timeout_buf()
        self._update_far_upper_dof_pos_buf()
        # print("reset_buf: ", self.reset_buf)
        # print("time_out_buf: ", self.time_out_buf)
        # print("far_upper_dof_pos_buf: ", self.far_upper_dof_pos_buf)

    def _update_timeout_buf(self):
        super()._update_timeout_buf()
        if self.config.termination.terminate_when_motion_end:
            current_time = (self.episode_length_buf) * self.dt + self.motion_start_times
            self.time_out_buf |= current_time > self.motion_len
        # print("time_out_buf: ", self.time_out_buf)
        self.reset_buf |= self.time_out_buf
    
    def _update_far_upper_dof_pos_buf(self):
        if self.config.termination.terminate_when_low_upper_dof_tracking:
            # Yuanhang: upper body dof position tracking error
            # upper_body_dof_pos_tracking_reward = self._reward_tracking_upper_body_dofs()
            # self.far_upper_dof_pos_buf[:] = upper_body_dof_pos_tracking_reward < self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold
            # ExBody
            dof_dev = torch.exp(-0.5 * torch.norm((self.simulator.dof_pos[:, self.upper_dof_indices] - self.ref_upper_dof_pos), dim=1))
            self.far_upper_dof_pos_buf[:] = dof_dev < self.config.termination_scales.terminate_when_low_upper_dof_tracking_threshold
            # print("upper_body_dof_pos_tracking_error: ", upper_body_dof_pos_tracking_error)
            self.reset_buf |= self.far_upper_dof_pos_buf
        else:
            self.far_upper_dof_pos_buf[:] = False

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
        else:
            self.action_scale_upper_body[env_ids] = 1.0
        if self.config.rewards.get("command_height_scale_curriculum", False):
            self._update_command_height_curriculum(env_ids)
        else:
            self.command_height_scale[env_ids] = 1.0
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)        # if target_states is not None, reset to target states
        self._reset_robot_states_callback(env_ids, target_states)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        # self._refresh_sim_tensors()

    ########################### CURRICULUM ###########################
    def _update_command_height_curriculum(self, env_ids):
        """
        Update the command height scale based on the episode length for each environment.
        Returns:
            None
        """
        env_ids_scale_up_mask = self.episode_length_buf[env_ids] > self.config.rewards.command_height_scale_up_threshold
        env_ids_scale_up = env_ids[torch.where(env_ids_scale_up_mask)[0]]
        env_ids_scale_down_mask = self.episode_length_buf[env_ids] < self.config.rewards.command_height_scale_down_threshold
        env_ids_scale_down = env_ids[torch.where(env_ids_scale_down_mask)[0]]
        self.command_height_scale[env_ids_scale_up] += self.config.rewards.command_height_scale_up
        self.command_height_scale[env_ids_scale_down] -= self.config.rewards.command_height_scale_down
        # self.command_height_scale[env_ids_scale_up] *= (1 + self.config.rewards.command_height_scale_degree)
        # self.command_height_scale[env_ids_scale_down] *= (1 - self.config.rewards.command_height_scale_degree)
        # Clip the scale
        self.command_height_scale[env_ids] = torch.clip(self.command_height_scale[env_ids], 
                                                        self.config.rewards.command_height_scale_min, 
                                                        self.config.rewards.command_height_scale_max)
        
    ########################### FEET REWARDS ###########################
    def _reward_penalty_hip_pos(self):
        # Penalize the hip joints (only roll and yaw)
        hips_roll_yaw_indices = self.hips_dof_id[1:3] + self.hips_dof_id[4:6]
        hip_pos = self.simulator.dof_pos[:, hips_roll_yaw_indices]
        penalty_hip_pos = torch.sum(torch.square(hip_pos), dim=1)
        return penalty_hip_pos * (self.commands[:, 4] + (1 - self.commands[:, 4]) * self.commands[:, 8])

    def _reward_penalty_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) * self.commands[:, 4] # only apply the base linear z-axis velocity penalty if locomoting

    def _reward_penalty_torso_orientation(self):
        # Penalize non flat torso orientation
        torso_quat = self.simulator._rigid_body_rot[:, self.torso_index]
        projected_gravity_torso = quat_rotate_inverse(torso_quat, self.gravity_vec)
        return torch.sum(torch.abs(projected_gravity_torso[:, :2]), dim=1)
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.simulator.robot_root_states[:, 2]
        # return torch.square(base_height - self.config.rewards.desired_base_height)*self.commands[:, 4] # only apply the base height penalty if locomoting
        penalty_base_height = torch.square(base_height - self.commands[:, 8]) # only apply the base height penalty if standing
        stance_env_idx = torch.where(self.commands[:, 4] < 1)[0]
        penalty_base_height[stance_env_idx] *= self.stance_base_height_penalty_scale # double the penalty if standing
        return penalty_base_height
    
    def _reward_tracking_base_height(self):
        # Tracking of base height commands (z axe)
        base_height_error = torch.abs(self.commands[:, 8] - self.simulator.robot_root_states[:, 2])
        return torch.exp(-base_height_error/self.config.rewards.reward_tracking_sigma.base_height)
    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity x commands
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_x_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity y commands
        lin_vel_y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.config.rewards.reward_tracking_sigma.lin_vel)
    
    ######################## LIMITS REWARDS #########################
    def _reward_limits_lower_body_dof_pos(self):
        # Penalize dof positions too close to the limit (lower body only)
        out_of_limits = -(self.simulator.dof_pos[:, self.lower_dof_indices] - \
                          self.simulator.dof_pos_limits[self.lower_dof_indices, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos[:, self.lower_dof_indices] - \
                          self.simulator.dof_pos_limits[self.lower_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_limits_upper_body_dof_pos(self):
        # Penalize dof positions too close to the limit (upper body only)
        out_of_limits = -(self.simulator.dof_pos[:, self.upper_dof_indices] - \
                          self.simulator.dof_pos_limits[self.upper_dof_indices, 0]).clip(max=0.)
        out_of_limits += (self.simulator.dof_pos[:, self.upper_dof_indices] - \
                          self.simulator.dof_pos_limits[self.upper_dof_indices, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_limits_lower_body_dof_vel(self):
        # Penalize dof velocities too close to the limit (lower body only)
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel[:, self.lower_dof_indices]) - \
                          self.simulator.dof_vel_limits[self.lower_dof_indices]).clip(min=0., max=1.), dim=1)

    def _reward_limits_upper_body_dof_vel(self):
        # Penalize dof velocities too close to the limit (upper body only)
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.simulator.dof_vel[:, self.upper_dof_indices]) - \
                          self.simulator.dof_vel_limits[self.upper_dof_indices]).clip(min=0., max=1.), dim=1)

    def _reward_limits_lower_body_torque(self):
        # penalize torques too close to the limit (lower body only)
        return torch.sum((torch.abs(self.torques[:, self.lower_dof_indices]) - \
                          self.torque_limits[self.lower_dof_indices]).clip(min=0.), dim=1)
    
    def _reward_limits_upper_body_torque(self):
        # penalize torques too close to the limit (upper body only)
        # TODO: Hardcode the 0.9
        return torch.sum((torch.abs(self.torques[:, self.upper_dof_indices]) - \
                          self.torque_limits[self.upper_dof_indices]*0.9).clip(min=0.), dim=1)
    
    ######################### PENALTY REWARDS #########################
    
    def _reward_penalty_lower_body_torques(self):
        # Penalize torques (lower body only)
        return torch.sum(torch.square(self.torques[:, self.lower_dof_indices]), dim=1)
    
    def _reward_penalty_upper_body_torques(self):
        # Penalize torques (upper body only)
        return torch.sum(torch.square(self.torques[:, self.upper_dof_indices]), dim=1)
    
    def _reward_penalty_lower_body_dof_vel(self):
        # Penalize dof velocities (lower body only)
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.lower_dof_indices]), dim=1)
    
    def _reward_penalty_upper_body_dof_vel(self):
        # Penalize dof velocities (upper body only)
        return torch.sum(torch.square(self.simulator.dof_vel[:, self.upper_dof_indices]), dim=1)
    
    def _reward_penalty_lower_body_dof_acc(self):
        # Penalize dof accelerations (lower body only)
        return torch.sum(torch.square((self.last_dof_vel[:, self.lower_dof_indices] - \
                                       self.simulator.dof_vel[:, self.lower_dof_indices]) / self.dt), dim=1)
    
    def _reward_penalty_upper_body_dof_acc(self):
        # Penalize dof accelerations (upper body only)
        return torch.sum(torch.square((self.last_dof_vel[:, self.upper_dof_indices] - \
                                       self.simulator.dof_vel[:, self.upper_dof_indices]) / self.dt), dim=1)
    
    def _reward_penalty_lower_body_action_rate(self):
        # Penalize changes in actions (lower body only)
        return torch.sum(torch.square(self.last_actions[:, self.lower_dof_indices] - \
                                      self.actions[:, self.lower_dof_indices]), dim=1)
    
    def _reward_penalty_upper_body_action_rate(self):
        # Penalize changes in actions (upper body only)
        return torch.sum(torch.square(self.last_actions[:, self.upper_dof_indices] - \
                                      self.actions[:, self.upper_dof_indices]), dim=1)
    
    ######################### TRACKING REWARDS #########################
    
    def _reward_tracking_upper_body_dofs(self):
        # Reward the difference between the waist dof pos and the reference
        upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
        upper_body_dofs_error =  torch.sum(torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1)
        return torch.exp(-upper_body_dofs_error/self.config.rewards.reward_tracking_sigma.upper_body_dofs)

    def _reward_penalty_upper_body_dofs_freeze(self):
        # returns keep the upper body joint angles close to the default
        assert self.config.robot.has_upper_body_dof
        deviation = torch.abs(self.simulator.dof_pos[:, self.upper_dof_indices] - self.default_dof_pos[:, self.upper_dof_indices])
        # print(torch.sum(deviation, dim=1))
        return torch.sum(deviation, dim=1)
    
    def _reward_tracking_waist_dofs(self):
        # Penalize the difference between the waist dof pos and the reference
        waist_dofs_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.waist_yaw_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_yaw_dof_indice] - self.commands[:, 5])
        if self.waist_roll_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_roll_dof_indice] - self.commands[:, 6])
        if self.waist_pitch_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_pitch_dof_indice] - self.commands[:, 7])
        return torch.exp(-waist_dofs_error/self.config.rewards.reward_tracking_sigma.waist_dofs)
    
    def _reward_tracking_waist_dofs_stance(self):
        # Penalize the difference between the waist dof pos and the reference
        waist_dofs_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.waist_yaw_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_yaw_dof_indice] - self.commands[:, 5])
        if self.waist_roll_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_roll_dof_indice] - self.commands[:, 6])
        if self.waist_pitch_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_pitch_dof_indice] - self.commands[:, 7])
        return torch.exp(-waist_dofs_error/self.config.rewards.reward_tracking_sigma.waist_dofs) * (1 - self.commands[:, 4])
    
    def _reward_tracking_waist_dofs_tapping(self):
        # Penalize the difference between the waist dof pos and the reference
        waist_dofs_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.waist_yaw_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_yaw_dof_indice] - self.commands[:, 5])
        if self.waist_roll_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_roll_dof_indice] - self.commands[:, 6])
        if self.waist_pitch_dof_indice:
            waist_dofs_error += torch.square(self.simulator.dof_pos[:, self.waist_pitch_dof_indice] - self.commands[:, 7])
        return torch.exp(-waist_dofs_error/self.config.rewards.reward_tracking_sigma.waist_dofs) * self.commands[:, 4]

    ######################### Observations #########################
    def _get_obs_command_waist_dofs(self):
        return self.commands[:, 5:8]
    
    def _get_obs_command_base_height(self):
        return self.commands[:, 8:9]

    def _get_obs_base_orientation(self):
        return self.base_quat[:, 0:4]
    
    def _get_obs_actions(self):
        return self.actions