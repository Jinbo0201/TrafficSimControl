from __future__ import annotations

import gymnasium as gym
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.merge_env import MergeEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle


class MergeAutoEnv(AbstractEnv):
    """A merge environment where no explicit controlled agent is required.

    All vehicles follow automatic behaviors.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "simulation_frequency": 15,
                "policy_frequency": 1,
                "duration": 30,
                "screen_width": 800,  # 增加宽度
                "screen_height": 400,  # 增加高度
                "scaling": 8.0,  # 增加缩放使车辆更大更清楚
                "centering_position": [0.5, 0.6],
            }
        )
        return config

    def _reward(self, action=None) -> float:
        return 0.0

    def _rewards(self, action=None) -> dict[str, float]:
        return {}

    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.road.vehicles)

    def _is_truncated(self) -> bool:
        return self.time >= self.config.get("duration", 30)

    def reset(self, *, seed=None, options=None):
        # 不调用 AbstractEnv.reset 避免在 observer_vehicle 未设置时触发错误
        gym.Env.reset(self, seed=seed)
        if options and "config" in options:
            self.configure(options["config"])

        self.update_metadata()
        self.define_spaces()
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()

        if self.road and self.road.vehicles:
            self.observation_type.observer_vehicle = self.road.vehicles[0]

        obs = self.observation_type.observe()
        info = self._info(obs, action=None)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()
        ends = [150, 80, 80, 150]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]

        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        self.road.vehicles = []
        other_vehicles_type = IDMVehicle

        # highway traffic - 增加数量
        highway_positions = [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5), (120.0, 28.5), (50.0, 32.0), (30.0, 30.5)]
        for position, speed in highway_positions:
            lane = self.road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            speed += self.np_random.uniform(-1.0, 1.0)
            self.road.vehicles.append(other_vehicles_type(self.road, position, speed=speed))

        # merging vehicles - 增加数量
        merging_positions = [(110.0, 20.0), (130.0, 18.0)]
        for pos, spd in merging_positions:
            merging_v = other_vehicles_type(
                self.road,
                self.road.network.get_lane(("j", "k", 0)).position(pos, 0.0),
                speed=spd,
            )
            merging_v.target_speed = 30.0
            self.road.vehicles.append(merging_v)

        # 设定观察者车辆，避免观察逻辑报错
        if self.road.vehicles:
            self.observation_type.observer_vehicle = self.road.vehicles[0]

    def _simulate(self, action=None) -> None:
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            if frame < frames - 1:
                self._automatic_rendering()

    def _info(self, obs, action=None):
        return {
            "vehicles": len(self.road.vehicles),
            "crashed": any(v.crashed for v in self.road.vehicles),
        }

    def step(self, action=None):
        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)
        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info
