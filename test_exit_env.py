import os
import gymnasium as gym
import highway_env

# 如果在无头环境下运行，可启用下面这一行（在本地带界面时可注释掉）：
# os.environ['SDL_VIDEODRIVER'] = 'dummy'


def test_exit_env():
    """测试 ExitEnv 环境，并开启界面显示。"""
    # 创建环境并启用渲染
    env = gym.make('exit-v0', render_mode='human')

    obs, info = env.reset()
    print("初始观察:", obs)
    print("初始信息:", info)

    num_steps = 200
    total_reward = 0.0

    for step in range(num_steps):
        action = env.action_space.sample()  # 随机动作，可替换为策略动作

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(f"step={step+1}, action={action}, reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

        # 如果你要持续可视化，减少打印频率
        if (step + 1) % 20 == 0:
            print(f"已完成 {step + 1} 步，总奖励 {total_reward:.3f}")

        if terminated or truncated:
            print("Episode 结束。")
            break

    print("最终总奖励:", total_reward)
    env.close()


if __name__ == '__main__':
    test_exit_env()