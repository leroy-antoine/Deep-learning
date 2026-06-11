##
## EPITECH PROJECT, 2026
## START_TREK
## File description:
## eval
##

import os
import yaml
import gymnasium as gym
import numpy as np
from ppo_torch import Agent
from gymnasium.wrappers import RecordVideo


with open('config.yaml', 'r') as file :
    config = yaml.safe_load(file)

def evaluate(n_eval_episodes: int = config['eval']['n_episodes']):
    os.makedirs('ppo_lunar_lander_videos/', exist_ok=True)

    env = gym.make(config['env']['name'],
                   enable_wind=config['env']['wind'],
                   wind_power=config['env']['wind_power'],
                   turbulence_power=config['env']['turbulence_power'],
                   render_mode=config['env']['render_mode'])

    env = RecordVideo(env, video_folder=config['env']['video_folder'],
                      episode_trigger=lambda ep: ep % 10 == 0)

    agent = Agent(n_actions=env.action_space.n, batch_size=64, alpha=0.00003,
                  n_epochs=5, input_dims=env.observation_space.shape, chkpt_dir='tmp/ppo_with_wind', policy_clip=0.1
    )
    if not (os.path.exists('tmp/ppo_with_wind/actor_torch_ppo') and
            os.path.exists('tmp/ppo_with_wind/critic_torch_ppo')):
        print("ERROR : tmp/ppo_with_wind/actor_torch_ppo and "
              "tmp/ppo_with_wind/critic_torch_ppo directories doesn't exist.")
        return
    agent.load_models()
    score_history = []
    n_landed    = 0
    n_crashed   = 0

    for i in range(n_eval_episodes):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            action, _, _ = agent.choose_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward

        if terminated:
            if reward == -100:
                end_reason = "CRASHED"
                n_crashed += 1
            else:
                end_reason = "LANDED"
                n_landed += 1

        score_history.append(score)
        print(f"{end_reason}")
    env.close()
    avg  = np.mean(score_history)
    std  = np.std(score_history)
    best = np.max(score_history)
    worst = np.min(score_history)
    print("-" * 50)
    print("EVALUATION :")
    print(f"Number of episodes  : {n_eval_episodes}")
    print(f"Average score       : {avg:.1f} ± {std:.1f}")
    print(f"Best score / Worst  : {best:.1f} / {worst:.1f}")
    print(f"Percentage landed   : {round(n_landed / n_eval_episodes * 100)}%")
    print(f"Percentage crashed  : {round(n_crashed / n_eval_episodes * 100)}%")

if __name__ == '__main__':
    evaluate()