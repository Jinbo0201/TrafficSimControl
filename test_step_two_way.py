import numpy as np
import time

from highway_env.envs.two_way_env import TwoWayEnv

def test_step():
    """
    测试 TwoWayEnv 的仿真推进：生成路网和车辆，然后一步步推进仿真，并显示动画。
    不使用 Gym 框架，直接操作 Road 和 Vehicle。
    """
    # 创建环境实例，启用渲染
    env = TwoWayEnv(config={"show_trajectories": True}, render_mode="human")

    # 重置环境，生成路网和车辆
    env._reset()

    print("路网和车辆已生成。")
    print(f"车辆数量: {len(env.road.vehicles)}")
    print(f"路网车道: {list(env.road.network.graph.keys())}")

    # 仿真参数
    dt = 1 / 15  # 时间步长，15 Hz，与默认 simulation_frequency 匹配
    max_steps = 100  # 增加步数以观察动画

    # 一步步推进仿真
    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")

        # 车辆决定动作（act）
        env.road.act()

        # 更新车辆状态（step）
        env.road.step(dt)

        # 更新仿真时间
        env.time += dt

        # 渲染动画
        env.render()

        # 打印关键信息（可选，减少输出以专注动画）
        if step % 10 == 0:  # 每10步打印一次
            print(f"仿真时间: {env.time:.2f} s")
            for i, vehicle in enumerate(env.road.vehicles):
                print(f"车辆 {i}: 位置 {vehicle.position}, 速度 {vehicle.speed:.2f} m/s, 碰撞 {vehicle.crashed}")

        # 检查终止条件
        if any(vehicle.crashed for vehicle in env.road.vehicles):
            print("检测到碰撞，仿真结束。")
            break

        # 控制推进速度（与动画同步）
        time.sleep(dt)

    print("\n仿真推进完成。")
    env.close()  # 关闭渲染窗口

if __name__ == "__main__":
    test_step()