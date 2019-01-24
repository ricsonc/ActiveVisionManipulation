import gym

env_id = "HalfCheetah-v1"
env = gym.make(env_id)
done = False
env.reset()
while(not done):
    a = env.action_space.sample()
    _, _, done, _ = env.step(a)
    env.render()