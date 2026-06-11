##
## EPITECH PROJECT, 2026
## START_TREK
## File description:
## main
##

import os
import pandas as pd
import gymnasium as gym
import numpy as np
import yaml
from ppo_torch import Agent
from gymnasium.wrappers import RecordVideo

with open('config.yaml', 'r') as file :
    config = yaml.safe_load(file)

def train():
    N = config['train']['N']
    batch_size = config['train']['batch_size']
    n_epochs = config['train']['n_epochs']
    alpha = config['train']['alpha']
    n_train = config['train']['n_train']
    best_score = -np.inf
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    crashed = False

    if os.path.exists('eval.csv'):
        df = pd.read_csv('eval.csv')
    else:
        df = pd.DataFrame(columns=['score', 'average score', 'crashed ?'])
    os.makedirs('ppo_lunar_lander_videos', exist_ok=True)
    os.makedirs('tmp/ppo_with_wind', exist_ok=True)
    env = gym.make(config['env']['name'],
                   enable_wind=config['env']['wind'],
                   wind_power=config['env']['wind_power'],
                   turbulence_power=config['env']['turbulence_power'],
                   render_mode=config['env']['render_mode'])

    env = RecordVideo(env, video_folder=config['env']['video_folder'],
                      episode_trigger=lambda ep: ep % 10 == 0)

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  chkpt_dir='tmp/ppo_with_wind', policy_clip=config['agent']['policy_clip'], gamma=config['agent']['gamma'],
                  gae_lambda=config['agent']['gae_lambda'],
                  fc1_dims=config['agent']['fc1_dims'],
                  fc2_dims=config['agent']['fc2_dims'])

    if os.path.exists('tmp/ppo_with_wind/actor_torch_ppo') and \
            os.path.exists('tmp/ppo_with_wind/critic_torch_ppo'):
        with open('tmp/ppo_with_wind/best_score.txt', 'r') as file:
            best_score = float(file.read())
        agent.load_models()

    for i in range(n_train):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
            with open('tmp/ppo_with_wind/best_score.txt', 'w') as file:
                file.write(str(best_score))
        if reward == -100 :
            crashed = True
        else:
            crashed = False
        new_row = pd.DataFrame([{'score' : round(score, 2),
                                 'average score' : round(avg_score, 2),
                                 'crashed ?' : crashed}])
        df = pd.concat([df, new_row], ignore_index=True)
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
        df.to_csv('eval.csv', index=False)

if __name__ == '__main__':
    train()