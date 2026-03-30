import gymnasium as gym
import highway_env


def test_merge_auto_env():
    """测试 merge-auto-v0 环境：自动驾驶，无需来自外部的动作控制。"""
    env = gym.make("merge-auto-v0", render_mode="human")

    obs, info = env.reset()
    print("初始观察: ", obs)
    print("初始 info: ", info)

    max_steps = 200

    for step in range(max_steps):
        # 不需要任何具体动作，传 None 让环境内置行为自行推进
        obs, reward, terminated, truncated, info = env.step(None)

        if step % 20 == 0:
            print(f"step={step}, reward={reward:.3f}, terminated={terminated}, truncated={truncated}")

        if terminated or truncated:
            print("Episode 结束, step=", step)
            break

    print("测试结束，总步数:", step + 1)
    env.close()


if __name__ == '__main__':
    test_merge_auto_env()
