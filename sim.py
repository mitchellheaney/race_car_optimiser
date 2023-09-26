from optimiser import BSWCEnv
from stable_baselines3.common.env_checker import check_env
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = BSWCEnv()
print(env.observation_space.sample())
env.reset()
check_env(env, warn=True)

episodes = 5
for episode in range(episodes):
    state, info = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done_sim, trunc, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000)

model.save('PPO')
evaluate_policy(model, env, n_eval_episodes=10, render=True)
