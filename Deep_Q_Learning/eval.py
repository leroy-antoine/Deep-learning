
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
import pandas as pd
from dql_torch import Agent
from gymnasium.wrappers import RecordVideo

with open("Deep_Q_Learning/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def evaluate(n_eval_episodes: int = config["eval"]["n_episodes"]):
    os.makedirs(config["env"]["video_folder"], exist_ok=True)
    env = gym.make(config["env"]["name"], render_mode=config["env"]["render_mode"])
    env = RecordVideo(
        env,
        video_folder=config["env"]["video_folder"],
        episode_trigger=lambda ep: ep % 10 == 0,
    )
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(
        state_size,
        action_size,
        lr=config["agent"]["lr"],
        gamma=config["agent"]["gamma"],
        tau=config["agent"]["tau"],
        batch_size=config["memory"]["batch_size"],
        buffer_size=config["memory"]["buffer_size"],
        update_every=config["training"]["update_every"],
    )
    checkpoint_path = config["checkpoint"]["path"]
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at '{checkpoint_path}'.")
        return
    agent.load(checkpoint_path)
    agent.local_qnetwork.eval()
    score_history = []
    n_landed = 0
    n_crashed = 0
    for i in range(n_eval_episodes):
        state, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        if terminated:
            if reward == -100:
                end_reason = "CRASHED"
                n_crashed += 1
            else:
                end_reason = "LANDED"
                n_landed += 1
        else:
            end_reason = "TRUNCATED"
        score_history.append(score)
        print(f"Episode {i:>3} | Score: {score:>8.1f} | {end_reason}")
    env.close()
    avg   = np.mean(score_history)
    std   = np.std(score_history)
    best  = np.max(score_history)
    worst = np.min(score_history)

    print("-" * 50)
    print("EVALUATION :")
    print(f"Number of episodes  : {n_eval_episodes}")
    print(f"Average score       : {avg:.1f} ± {std:.1f}")
    print(f"Best score / Worst  : {best:.1f} / {worst:.1f}")
    print(f"Percentage landed   : {round(n_landed  / n_eval_episodes * 100)}%")
    print(f"Percentage crashed  : {round(n_crashed / n_eval_episodes * 100)}%")

if __name__ == "__main__":
    evaluate()

