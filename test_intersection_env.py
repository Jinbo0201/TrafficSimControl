import gymnasium as gym
import highway_env


def test_intersection_env():
    """测试 IntersectionEnv 环境，并开启界面显示。"""
    env = gym.make('intersection-v0', render_mode='human')

    obs, info = env.reset()
    print('初始观察:', obs)
    print('初始信息:', info)

    total_reward = 0.0
    max_steps = 200

    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 20 == 0:
            print(f'step={step+1}, reward={reward:.3f}, total_reward={total_reward:.3f}, terminated={terminated}, truncated={truncated}')

        if terminated or truncated:
            print('Episode 结束')
            break

    print('最终总奖励:', total_reward)
    env.close()


if __name__ == '__main__':
    test_intersection_env()