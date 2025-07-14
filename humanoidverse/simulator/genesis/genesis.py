import sys
import os
from loguru import logger
# from isaacgym import gymtorch, gymapi, gymutil
import torch
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from humanoidverse.simulator.genesis.tmp_gs_utils import *
from humanoidverse.simulator.genesis.genesis_viewer import Viewer
from humanoidverse.utils.torch_utils import to_torch, torch_rand_float
import numpy as np
from termcolor import colored
from rich.progress import Progress
from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
import copy
import trimesh
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import time
import functools


# debugging purposes
def time_method(func):
    """Decorator to time any method"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result

    return wrapper


def time_all_methods(cls):
    """Class decorator to time all methods"""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('__'):
            setattr(cls, attr_name, time_method(attr))
    return cls


def quat_rotate(q, v):
    # q: (B, L, 4), v: (B, L, 3)
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


class Genesis(BaseSimulator):
    """
    Base class for robotic simulation environments, providing a framework for simulation setup,
    environment creation, and control over robotic assets and simulation properties.
    """

    def __init__(self, config, device):
        """
        Initializes the base simulator with configuration settings and simulation device.

        Args:
            config (dict): Configuration dictionary for the simulation.
            device (str): Device type for simulation ('cpu' or 'cuda').
        """
        self.cfg = config
        self.sim_cfg = config.simulator.config
        self.robot_cfg = config.robot
        self.device = device
        self.sim_device = device
        self.headless = False
        gs.init(backend=gs.gpu if 'cuda' in self.device else gs.cpu)
        self.debug_lines = []

    def _cache_full_jacobian(self):
        """
        Build `self.jacobian` with the same shape and semantics that
        Isaac Gym code expects:  (num_envs , num_bodies , 6 , num_dof)
        """
        jac_list = []
        for name in self.body_names:  # same ordering as cfg
            link = self.robot.get_link(name)
            # (num_envs , 6 , num_dof)
            jac = self.robot.get_jacobian(link)
            jac_list.append(jac)

        # stack along body axis  ➜  (num_envs , num_bodies , 6 , num_dof)
        self.jacobian = torch.stack(jac_list, dim=1).to(self.device)
        # print(f"[Jacobian Cached] Shape: {self.jacobian.shape}  | Expected: (num_envs={self.jacobian.shape[0]}, num_bodies={self.jacobian.shape[1]}, 6, num_dof={self.jacobian.shape[-1]})")

    # @time_method
    # def apply_rigid_body_force_at_pos_tensor(
    #         self,
    #         force_tensor: torch.Tensor,  # (B, L, 3)
    #         pos_tensor: torch.Tensor,  # (B, L, 3)
    # ):
    #     if force_tensor.numel() == 0:
    #         return
    #
    #     # Get rigid solver
    #     if not hasattr(self, "_rigid_solver"):
    #         for _solver in self.scene.sim.solvers:
    #             if isinstance(_solver, RigidSolver):
    #                 self._rigid_solver = _solver
    #                 break
    #
    #     rigid_solver = self._rigid_solver
    #
    #     forces_np = force_tensor.detach().cpu().to(torch.float32).numpy()  # (B,L,3)
    #     active = np.linalg.norm(forces_np, axis=2) > 1e-8  # (B,L)
    #
    #     # links_idx: only the two hand links
    #     links_idx = np.array([23, 31], dtype=int)
    #
    #     # envs_idx: every env that has a non-zero force on *either* hand link
    #     envs_idx = np.where(np.any(active[:, links_idx], axis=1))[0]  # (E,)
    #
    #     if envs_idx.size == 0:
    #         return  # nothing to apply
    #
    #     # Slice out the dense (E,2,3) block; zero rows stay zero automatically
    #     forces_dense = forces_np[np.ix_(envs_idx, links_idx)]  # (E, 2, 3)
    #
    #     # for torque
    #     link_com_pos = self.robot.get_links_pos()[:, links_idx]
    #     pos_tensor = pos_tensor[:, links_idx]  # match active links
    #     pos_tensor_world = pos_tensor.detach().cpu().float()
    #     lever_arm = pos_tensor_world - link_com_pos[envs_idx].cpu()  # (E, 2, 3)
    #     torque_tensor = torch.cross(lever_arm, torch.tensor(forces_dense), dim=2)  # (E, 2, 3)
    #     torque_tensor_np = torque_tensor.numpy()
    #
    #     print(self.robot.get_links_pos().shape)
    #
    #     rigid_solver.apply_links_external_force(
    #         force=forces_dense,  # shape (E, 2, 3)
    #         links_idx=links_idx.tolist(),  # [23, 31]
    #         envs_idx=envs_idx.tolist(),  # active envs
    #     )
    #     rigid_solver.apply_links_external_torque(
    #         torque=torque_tensor_np,
    #         links_idx=links_idx.tolist(),
    #         envs_idx=envs_idx.tolist()
    #     )
    #     print('test')

    def apply_rigid_body_force_at_pos_tensor(
            self,
            force_tensor: torch.Tensor,  # (B, L, 3)
            pos_tensor: torch.Tensor  # (B, L, 3)
    ):
        if force_tensor.numel() == 0:
            return

        if not hasattr(self, "_rigid_solver"):
            for _solver in self.scene.sim.solvers:
                if isinstance(_solver, RigidSolver):
                    self._rigid_solver = _solver
                    break

        rigid_solver = self._rigid_solver

        links_idx = np.array([23, 31], dtype=int)

        # Select hand-related forces/positions
        force_tensor_sub = force_tensor[:, links_idx]  # (B, 2, 3)
        pos_tensor_sub = pos_tensor[:, links_idx]  # (B, 2, 3)
        link_com_pos = self.robot.get_links_pos()[:, links_idx]  # (B, 2, 3)

        # Compute torque = r × F
        lever_arm = (pos_tensor_sub - link_com_pos).detach().cpu().float()  # (B, 2, 3)
        forces_np = force_tensor_sub.detach().cpu().float().numpy()
        torque_tensor = torch.cross(lever_arm, torch.tensor(forces_np), dim=2)  # (B, 2, 3)

        # Zero out all-zero environments (optional for perf)
        active = torch.norm(force_tensor_sub, dim=2) > 1e-8  # (B, 2)
        inactive_mask = ~(active.any(dim=1))  # (B,)
        forces_np[inactive_mask.cpu().numpy()] = 0.0

        # Send to solver
        rigid_solver.apply_links_external_force(
            force=forces_np,
            links_idx=links_idx.tolist(),
            envs_idx=None  # Apply to all envs
        )
        rigid_solver.apply_links_external_torque(
            torque=torque_tensor.numpy(),
            links_idx=links_idx.tolist(),
            envs_idx=None
        )

    # def apply_rigid_body_force_at_pos_tensor(
    #     self,
    #     force_tensor: torch.Tensor,   # (B, L, 3)  world-frame forces
    #     pos_tensor:  torch.Tensor,    # (B, L, 3)  kept for API parity, unused
    # ):
    #     """
    #     Apply world-frame forces stored in `force_tensor` to the corresponding
    #     links for each environment.  Forces are injected at each link’s origin
    #     (ref='link_origin') in world coordinates, matching Genesis semantics.
    #
    #     Parameters
    #     force_tensor : torch.Tensor  [num_envs, num_links, 3]
    #     pos_tensor   : torch.Tensor  [num_envs, num_links, 3]  (ignored)
    #     """
    #
    #     if force_tensor.numel() == 0:
    #         return  # nothing to do
    #
    #     if not hasattr(self, "_rigid_solver"):
    #         for _solver in self.scene.sim.solvers:          # type: ignore[attr-defined]
    #             if isinstance(_solver, RigidSolver):
    #                 self._rigid_solver = _solver
    #                 break
    #         else:
    #             raise RuntimeError("No RigidSolver found in scene.sim.solvers")
    #
    #     rigid_solver: RigidSolver = self._rigid_solver      # alias for brevity
    #
    #     forces_np = force_tensor.detach().cpu().to(torch.float32).numpy()  # (B, L, 3)
    #     B, L, _ = forces_np.shape
    #
    #     for env_id in range(B):
    #         f_env = forces_np[env_id]                             # (L, 3)
    #         active = np.any(np.abs(f_env) > 1e-8, axis=1)         # mask of non-zero rows
    #         if not np.any(active):
    #             continue
    #
    #         rigid_solver.apply_links_external_force(
    #             force=f_env[active][None, :, :],
    #             links_idx=np.where(active)[0].tolist(),           # global link ids
    #             envs_idx=[env_id],                                # which environment
    #         )

    # ----- Configuration Setup Methods -----
    def set_headless(self, headless):
        """
        Sets the headless mode for the simulator.

        Args:
            headless (bool): If True, runs the simulation without graphical display.
        """
        self.headless = headless

    def setup(self):
        """
        Initializes the simulator parameters and environment. This method should be implemented
        by subclasses to set specific simulator configurations.
        """

        self.sim_dt = 1 / self.sim_cfg.sim.fps
        self.sim_substeps = self.sim_cfg.sim.substeps
        # print('gh1: headless', self.headless)
        vis_options = None if self.headless else gs.options.VisOptions(n_rendered_envs=1)
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.sim_dt * self.sim_cfg.sim.control_decimation),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
            ),
            # vis_options=gs.options.VisOptions(
            #     n_rendered_envs=1,
            # ),
            show_viewer=not self.headless,
            show_FPS=False,
        )

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

    # ----- Terrain Setup Methods -----

    def setup_terrain(self, mesh_type):
        """
        Configures the terrain based on specified mesh type.

        Args:
            mesh_type (str): Type of terrain mesh ('plane', 'heightfield', 'trimesh').
        """
        if mesh_type == 'plane':
            # this is somehow deprecated
            # self.scene.add_entity(
            #     gs.morphs.URDF(file='urdf/plane/plane.urdf', scale=20.0, fixed=True),
            # )
            plane = self.scene.add_entity(gs.morphs.Plane())
        elif mesh_type == 'trimesh':
            raise NotImplementedError(f"Mesh type {mesh_type} hasn't been implemented in genesis subclass.")

    # ----- Robot Asset Setup Methods -----

    def load_assets(self):
        """
        Loads the robot assets into the simulation environment.
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names
        Args:
            robot_config (dict): HumanoidVerse Configuration for the robot asset.
        """
        init_quat_xyzw = self.robot_cfg.init_state.rot
        init_quat_wxyz = init_quat_xyzw[-1:] + init_quat_xyzw[:3]
        self.base_init_pos = torch.tensor(
            self.robot_cfg.init_state.pos, device=self.device
        )
        # self.base_init_pos[2] += 1.5
        self.base_init_quat = torch.tensor(
            init_quat_wxyz, device=self.device
        )

        asset_root = self.robot_cfg.asset.asset_root
        asset_file = self.robot_cfg.asset.urdf_file
        asset_path = os.path.join(asset_root, asset_file)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=asset_path,
                merge_fixed_links=True,
                links_to_keep=self.robot_cfg.body_names,
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=False,
        )

        dof_names_list = copy.deepcopy(self.robot_cfg.dof_names)

        self.genesis_link_names = [link.name for link in self.robot.links]
        self.humanoidverse_link_names = self.robot_cfg.body_names
        self.link_mapping_genesis_to_humanoidverse_idx = [self.genesis_link_names.index(name) for name in
                                                          self.humanoidverse_link_names]

        # names to indices
        self.dof_ids = [
            self.robot.get_joint(name).dof_idx_local
            for name in dof_names_list
        ]

        self.body_names = self.robot_cfg.body_names
        self.num_bodies = len(self.body_names)  # = len(self.rigid_solver.links) - 1
        self.dof_names = dof_names_list
        self.num_dof = len(dof_names_list)  # = len(self.rigid_solver.joints) - 2

    # ----- Environment Creation Methods -----

    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        for link in self.robot.links:
            flag = False
            if body_name in link.name:
                return link.idx - self.robot.link_start

    def create_envs(self, num_envs, env_origins, base_init_state):
        """
        Creates and initializes environments with specified configurations.

        Args:
            num_envs (int): Number of environments to create.
            env_origins (list): List of origin positions for each environment.
            base_init_state (array): Initial state of the base.
            env_config (dict): Configuration for each environment.
        """
        # build
        self.num_envs = num_envs
        self._body_list = [link.name for link in self.robot.links]
        # build a body list that are compatible with the robot_config
        # self._body_list = []
        # for body_name in self.robot_cfg.body_names:
        #     self._body_list.append(self._genesis_body_list[body_name])
        self.scene.build(n_envs=num_envs)
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        return None, None

    # ----- Property Retrieval Methods -----

    def get_dof_limits_properties(self):
        """
        Retrieves the DOF (degrees of freedom) limits and properties.

        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device,
                                               requires_grad=False)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.sim_device,
                                          requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_cfg.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_cfg.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_cfg.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_cfg.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.reward_limit.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        Prepares the simulation environment and refreshes any relevant tensors.
        """
        self.scene.step()
        self._cache_full_jacobian()

        self.base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()
        self.base_quat = base_quat[..., [1, 2, 3, 0, ]]

        # inv_base_quat = gs_inv_quat(base_quat)
        # self.base_lin_vel = gs_transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # self.base_ang_vel = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.base_lin_vel = self.robot.get_vel()
        self.base_ang_vel = self.robot.get_ang()

        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        self.robot_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )

        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)

        self.contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )

    def refresh_sim_tensors(self):
        """
        Refreshes the state tensors in the simulation to ensure they are up-to-date.
        """
        self._cache_full_jacobian()
        self.base_pos = self.robot.get_pos()
        base_quat = self.robot.get_quat()
        self.base_quat = base_quat[..., [1, 2, 3, 0, ]]

        # inv_base_quat = gs_inv_quat(base_quat)
        # self.base_lin_vel = gs_transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # self.base_ang_vel = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.base_lin_vel = self.robot.get_vel()
        self.base_ang_vel = self.robot.get_ang()

        self.all_root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        self.robot_root_states = self.all_root_states

        self.dof_pos = self.robot.get_dofs_position(self.dof_ids)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_ids)

        self.contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        self._rigid_body_pos = self.robot.get_links_pos()[:, self.link_mapping_genesis_to_humanoidverse_idx]
        self._rigid_body_rot = self.robot.get_links_quat()[:,
                               self.link_mapping_genesis_to_humanoidverse_idx]  # (num_envs, 4) isaacsim uses wxyz, we keep xyzw for consistency
        self._rigid_body_rot = self._rigid_body_rot[..., [1, 2, 3, 0]]
        self._rigid_body_vel = self.robot.get_links_vel()[:, self.link_mapping_genesis_to_humanoidverse_idx]
        self._rigid_body_ang_vel = self.robot.get_links_ang()[:, self.link_mapping_genesis_to_humanoidverse_idx]

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        Applies the specified torques to the robot's degrees of freedom (DOF).

        Args:
            torques (tensor): Tensor containing torques to apply.
        """
        self.robot.control_dofs_force(torques, self.dof_ids)

    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """
        Sets the root state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            root_states (tensor): New root states to apply.
        """
        root_states = torch.cat(
            [
                self.base_pos,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
            ], dim=-1
        )
        root_states = self.robot_root_states[set_env_ids]

        base_pos = root_states[..., :3]
        base_quat = root_states[..., [6, 3, 4, 5]]
        base_lin_vel = root_states[..., 7:10]
        base_ang_vel = root_states[..., 10:13]

        # reset root states - position
        self.robot.set_pos(
            base_pos, zero_velocity=False, envs_idx=set_env_ids
        )
        self.robot.set_quat(
            base_quat, zero_velocity=False, envs_idx=set_env_ids
        )
        self.robot.set_dofs_velocity(
            base_lin_vel, dofs_idx_local=[0, 1, 2], envs_idx=set_env_ids
        )
        self.robot.set_dofs_velocity(
            base_ang_vel, dofs_idx_local=[3, 4, 5], envs_idx=set_env_ids
        )

    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """
        Sets the DOF state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            dof_states (tensor): New DOF states to apply.
        """
        dof_pos = dof_states.view(self.num_envs, -1, 2)[set_env_ids, :, 0]
        dof_vel = dof_states.view(self.num_envs, -1, 2)[set_env_ids, :, 1]

        # reset dofs
        self.robot.set_dofs_position(
            position=dof_pos,
            dofs_idx_local=self.dof_ids,
            envs_idx=set_env_ids,
        )
        self.robot.set_dofs_velocity(
            velocity=dof_vel,
            dofs_idx_local=self.dof_ids,
            envs_idx=set_env_ids,
        )

    def simulate_at_each_physics_step(self):
        """
        Advances the simulation by a single physics step.
        """
        self.scene.step()
        # self.refresh_sim_tensors()

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        self.viewer = Viewer()

    # def render(self, sync_frame_time=True):
    #     """
    #     Renders the simulation frame-by-frame, syncing frame time if required.
    #
    #     Args:
    #         sync_frame_time (bool): Whether to synchronize the frame time.
    #     """
    #     return

    def render(self, sync_frame_time=True):
        """
        Advances the simulation by one frame and handles viewer input.
        """
        self.scene.step()

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def add_visualize_entities(self, num_visualize_markers):
        # self.scene.add_entity(gs.morphs.Sphere())
        self.visualize_entities = []
        for i in range(num_visualize_markers):
            self.visualize_entities.append(
                self.scene.add_entity(gs.morphs.Sphere(radius=0.04, visualization=True, collision=False)))

    # debug visualization
    # def clear_lines(self):
    # self.scene.clear_debug_objects()
    # pass

    # def clear_lines(self):
    #     for line in getattr(self, "debug_lines", []):
    #         self.scene.remove_entity(line)
    #     self.debug_lines = []

    def clear_lines(self):
        """Clear all debug lines from the scene"""
        self.scene.clear_debug_objects()
        self.debug_lines = []

    def draw_sphere(self, pos, radius, color, env_id, pos_id=0):
        # self.scene.draw_debug_sphere(pos, radius, color)
        self.visualize_entities[pos_id].set_pos(pos.reshape(1, 3))

    # def draw_line(self, start_point, end_point, color, env_id):
    #     pass

    def draw_line(self, start_point, end_point, color=(1.0, 0.0, 0.0, 1.0), env_id=0):
        """
        Draws a line using Genesis's built-in debug line function.
        Handles CUDA tensors by converting to CPU first.
        """
        # Convert CUDA tensors to CPU floats
        start = [
            float(start_point.x.detach().cpu().item()),
            float(start_point.y.detach().cpu().item()),
            float(start_point.z.detach().cpu().item()),
        ]
        end = [
            float(end_point.x.detach().cpu().item()),
            float(end_point.y.detach().cpu().item()),
            float(end_point.z.detach().cpu().item()),
        ]
        color = [color.x, color.y, color.z, 1.0]

        # Compute distance
        dist = ((start_point.x - end_point.x) ** 2 +
                (start_point.y - end_point.y) ** 2 +
                (start_point.z - end_point.z) ** 2).sqrt().item()

        # # If length is less than 0.5, extend to length 0.5
        # if dist < 0.2:
        #     # Calculate direction vector
        #     direction_x = end_point.x - start_point.x
        #     direction_y = end_point.y - start_point.y
        #     direction_z = end_point.z - start_point.z
        #
        #     # Normalize and scale to 0.5
        #     if dist > 0:  # Avoid division by zero
        #         scale = 0.2 / dist
        #         new_end_x = start_point.x + direction_x * scale
        #         new_end_y = start_point.y + direction_y * scale
        #         new_end_z = start_point.z + direction_z * scale
        #
        #         end = [
        #             float(new_end_x.detach().cpu().item()),
        #             float(new_end_y.detach().cpu().item()),
        #             float(new_end_z.detach().cpu().item()),
        #         ]
        #         dist = 0.5

        # Set radius proportional to line length (or cap it)
        line_radius = 0.002

        # Draw line
        debug_line = self.scene.draw_debug_line(
            start=start,
            end=end,
            radius=line_radius,
            color=color
        )

        # Store reference for cleanup
        if not hasattr(self, 'debug_lines'):
            self.debug_lines = []
        self.debug_lines.append(debug_line)

        return debug_line
