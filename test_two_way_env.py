import gymnasium as gym
import highway_env  # 导入以注册环境
import numpy as np

def test_two_way_env():
    """
    测试 TwoWayEnv 环境的代码。
    环境的输入（动作）在范围内随机生成。
    """
    # 创建环境
    env = gym.make('two-way-v0', render_mode='human')  # 启用渲染以显示画面

    # 重置环境，获取初始观察
    obs, info = env.reset()
    env.render()  # 渲染初始画面
    print("初始观察:", obs)
    print("初始信息:", info)

    # 运行几个步骤，使用随机动作
    num_steps = 100
    total_reward = 0

    for step in range(num_steps):
        # 随机生成动作（假设动作空间是离散的，范围根据环境配置）
        action = env.action_space.sample()  # 随机采样动作

        action = 1

        

        print(f"步骤 {step + 1}: 选择动作 {action}")

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()  # 渲染画面 vscode远程编程不支持看画面
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
    test_two_way_env()