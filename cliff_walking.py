import gymnasium as gym

env = gym.make('CliffWalking-v0', render_mode="human")
state, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()