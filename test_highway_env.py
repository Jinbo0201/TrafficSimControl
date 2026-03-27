import gymnasium as gym
import highway_env  # 导入以注册环境
import numpy as np

def test_highway_env():
    """
    测试 HighwayEnv 环境的代码。
    环境的输入（动作）在范围内随机生成。
    """
    # 创建环境
    env = gym.make('highway-v0', render_mode='human')  # 启用渲染以显示界面

    # 重置环境，获取初始观察
    obs, info = env.reset()
    print("初始观察:", obs)
    print("初始信息:", info)

    # 运行几个步骤，使用随机动作
    num_steps = 50
    total_reward = 0

    for step in range(num_steps):
        # 随机生成动作（假设动作空间是离散的）
        action = env.action_space.sample()  # 随机采样动作

        # action = 4

        print(f"步骤 {step + 1}: 选择动作 {action}")

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"观察: {obs}")
        print(f"奖励: {reward}")
        print(f"终止: {terminated}, 截断: {truncated}")
        print(f"信息: {info}")
        print("-" * 50)

        if terminated or truncated:
            print("Episode ended.")
            break

    print(f"总奖励: {total_reward}")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    test_highway_env()