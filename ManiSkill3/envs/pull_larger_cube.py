from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import random

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PullLargerCube-v1", max_episode_steps=50)
@register_env("PullLargerCubeSwapped-v1", max_episode_steps=50, larger_cube_color="blue")
class PullLargerCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pull a larger cube out of two cubes (red & blue) over the green line.
    - if the blue_is_larger flag is set to True, the blue cube will be larger than the red cube 

    **Randomizations:**
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other
    - if the blue_is_larger flag is set to True, colors are randomized.

    **Success Conditions:**
    - the cube's x position is over goal_site's x position.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    goal_site_x = -0.25
    goal_error_threshold = 0.04
    larger_cube_half_size = 0.025
    smaller_cube_half_size = 0.015

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, larger_cube_color="red", **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.larger_cube_color = larger_cube_color
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
    # @property
    # def _default_sensor_configs(self):
    #     pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
    #     return [CameraConfig("base_camera", pose, 250, 250, 1, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        if self.larger_cube_color == "red":
            self.colors = [(1, 0, 0, 1), (0, 0, 1, 1)]
        elif self.larger_cube_color == "blue":
            self.colors = [(0, 0, 1, 1), (1, 0, 0, 1)]
        else: # random
            self.colors = [(1, 0, 0, 1), (0, 0, 1, 1)]
            random.shuffle(self.colors)
        larger_cube_color, smaller_cube_color = self.colors
            
        self.larger_cube = actors.build_cube(
            self.scene,
            half_size=self.larger_cube_half_size,
            color=larger_cube_color,
            name="larger_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.smaller_cube = actors.build_cube(
            self.scene,
            half_size=self.smaller_cube_half_size,
            color=smaller_cube_color,
            name="smaller_cube",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        self.goal_site = actors.build_box(
            self.scene,
            half_sizes=[0.003, 1, 0.003],
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        # self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            # xy = torch.rand((b, 2)) * 0.2 - 0.1
            xy = torch.rand((b, 2))
            xy[:, 0] = xy[:, 0] * 0.2 - 0.1  # Scale x to [-0.1, 0.1]
            xy[:, 1] = xy[:, 1] * (-0.1)     # Scale y to [-0.1, 0.0]
            
            region = [[0.0, -0.15], [0.015, 0.15]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            # radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            radius = 0.05
            larger_cube_xy = xy + sampler.sample(radius, 100)
            smaller_cube_xy = xy + sampler.sample(radius, 100, verbose=False)
            q = [1, 0, 0, 0]

            # make sure cubes do not 

            xyz[:, :2] = larger_cube_xy
            self.larger_cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=q))

            xyz[:, :2] = smaller_cube_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.smaller_cube.set_pose(Pose.create_from_pq(p=xyz, q=q))
            
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz = torch.tensor([self.goal_site_x, 0, 0.003])

            # # slight randomize randomize the goal x
            # target_region_xyz[0] += (torch.rand(1) * 0.03 - 0.015)

            self.goal_site.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=q,
                )
            )

    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position) and
        # the cube is still on the surface
        is_obj_pulled = (
            torch.abs(self.larger_cube.pose.p[..., 0] - self.goal_site.pose.p[..., 0])
            < self.goal_error_threshold
        ) & (self.larger_cube.pose.p[..., 2] < self.larger_cube_half_size + 5e-3)

        return {
            "success": is_obj_pulled,
        }

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        larger_cube_color, smaller_cube_color = self.colors
        b = self.num_envs  # Number of environments

        # Expand size and color to match the environment dimension
        larger_cube_size = torch.full((b, 1), self.larger_cube_half_size, device=self.device)
        smaller_cube_size = torch.full((b, 1), self.smaller_cube_half_size, device=self.device)
        larger_cube_color = torch.tensor(larger_cube_color, device=self.device).unsqueeze(0).repeat(b, 1)
        smaller_cube_color = torch.tensor(smaller_cube_color, device=self.device).unsqueeze(0).repeat(b, 1)

        # Combine cube data into tensors
        larger_cube_data = torch.cat(
            [self.larger_cube.pose.raw_pose, larger_cube_size, larger_cube_color], dim=1
        )
        smaller_cube_data = torch.cat(
            [self.smaller_cube.pose.raw_pose, smaller_cube_size, smaller_cube_color], dim=1
        )

        # Combine cube data into a single tensor and shuffle their order
        cubes = torch.cat([larger_cube_data.unsqueeze(1), smaller_cube_data.unsqueeze(1)], dim=1)  # Shape: (b, 2, data_dim)
        shuffle_mask = torch.rand(b, device=self.device) < 0.5  # Generate a boolean mask directly with 50% probability
        cubes[shuffle_mask] = cubes[shuffle_mask][:, [1, 0]]  # Swap larger and smaller cube data for selected environments

        # Reshape cubes to (b, 2 * data_dim) where data_dim is the number of features per cube
        cubes = cubes.view(b, -1)

        # Construct observation
        obs = dict(
            # is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            cubes=cubes, 
        )

        return obs
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_pull_pose = Pose.create_from_pq(
            p=self.larger_cube.pose.p
            + torch.tensor([self.larger_cube_half_size + 2 * 0.005, 0, 0], device=self.device)
        )
        tcp_to_pull_pose = tcp_pull_pose.p - self.agent.tcp.pose.p
        tcp_to_pull_pose_dist = torch.linalg.norm(tcp_to_pull_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_pull_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_pull_pose_dist < 0.01
        obj_to_goal_x_dist = torch.abs(self.larger_cube.pose.p[..., 0] - self.goal_site.pose.p[..., 0]) 
        place_reward = 1 - torch.tanh(5 * obj_to_goal_x_dist)
        reward += place_reward * reached


        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
