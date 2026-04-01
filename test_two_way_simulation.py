import time
import numpy as np
import matplotlib.pyplot as plt
import random  # 修改: 导入 random 模块用于生成随机目标速度

from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.behavior import IDMVehicle


def test_new():
    """直接构造 two-way 路网＋IDM 车辆，逐步推理仿真并可视化。"""

    # 1) 建路网（两车道双向整段）
    net = RoadNetwork()
    length = 1800
    width = StraightLane.DEFAULT_WIDTH

    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, 0], [length, 0], line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            speed_limit=40.0  # 修改: 显式设置车道限速，确保高于车辆目标速度，防止底层限制
        ),
    )
    net.add_lane(
        "a",
        "b",
        StraightLane(
            [0, width],
            [length, width],
            line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            speed_limit=40.0  # 修改: 显式设置车道限速
        ),
    )
    net.add_lane(
        "b",
        "a",
        StraightLane(
            [length, 0], [0, 0], line_types=(LineType.NONE, LineType.NONE),
            speed_limit=40.0  # 修改: 显式设置车道限速
        ),
    )
    net.add_lane(
        "b",
        "a",
        StraightLane(
            [length, width],
            [0, width],
            line_types=(LineType.NONE, LineType.NONE),
            speed_limit=40.0  # 修改: 显式设置车道限速
        ),
    )

    road = Road(network=net, np_random=np.random.RandomState(0), record_history=False)

    # 2) 加车（IDM）到某个车道
    def add_car(lane_index, s, speed, target_speed=None):
        lane = road.network.get_lane(lane_index)
        pos = lane.position(s, 0)
        v = IDMVehicle(
            road,
            pos,
            heading=lane.heading_at(s),
            speed=speed,
            # 若未指定 target_speed，则默认等于当前速度；建议显式设定以控制期望行为
            target_speed=target_speed if target_speed is not None else speed,
            target_lane_index=lane_index,
        )
        road.vehicles.append(v)
        return v

    # 主车：初始位置 0，速度 30，目标速度设为 30 (保持匀速)
    add_car(("a", "b", 0), 0.0, 30.0, 30.0)
    # add_car(("a", "b", 1), 70.0, 20.0)
    # add_car(("a", "b", 1), 110.0, 18.0)
    # add_car(("a", "b", 0), 170.0, 15.0)
    # add_car(("a", "b", 0), 250.0, 22.0)

    # 3) 可视化准备
    sim_fps = 15
    dt = 1.0 / sim_fps
    max_steps = 500

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.ion()

    for step in range(max_steps):
        road.act()
        road.step(dt)

        if step % 50 == 25:
            # 修改：动态加车时，target_speed 在 20-40 区间内随机生成
            random_target_speed = random.uniform(20, 40)
            add_car(("a", "b", 0), 0.0, 10.0, target_speed=random_target_speed)


        xs = [v.position[0] for v in road.vehicles]
        ys = [v.position[1] for v in road.vehicles]
        colors = ["r" if v.crashed else "b" for v in road.vehicles]

        ax.clear()
        ax.set_title(f"Step {step}    t={step*dt:.2f}s")
        ax.set_xlim(-10, length + 10)
        ax.set_ylim(-5, width + 5)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # 路面与车道线
        ax.hlines([0, width], -10, length + 10, colors="k", linewidth=2)
        ax.hlines([0.0 + 0.5 * width, width + 0.5 * width], -10, length + 10, colors="gray", linestyles="--", linewidth=0.5)

        # 画车辆
        for i, (x, y, c, v) in enumerate(zip(xs, ys, colors, road.vehicles)):
            ax.scatter(x, y, color=c, s=80)
            ax.text(x + 2, y + 0.2, f"v={v.speed:.1f}", fontsize=8)

        plt.pause(0.001)

        if any(v.crashed for v in road.vehicles):
            print(f"碰撞发生 at step {step}")
            break

    plt.ioff()
    plt.show()

    print("模拟结束")


if __name__ == "__main__":
    test_new()