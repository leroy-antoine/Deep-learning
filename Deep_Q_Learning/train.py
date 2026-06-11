##
## EPITECH PROJECT, 2026
## START_TREK
## File description:
## main
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

def train():
    if os.path.exists("Deep_Q_Learning/eval.csv"):
        df = pd.read_csv("Deep_Q_Learning/eval.csv")
    else:
        df = pd.DataFrame(columns=["score", "average score", "landed ?"])
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
    if os.path.exists(config["checkpoint"]["path"]):
        agent.load(config["checkpoint"]["path"])
        if os.path.exists("Deep_Q_Learning/epsilon.txt"):
            with open("Deep_Q_Learning/epsilon.txt", "r") as f:
                epsilon = float(f.read())
        else:
            epsilon = config["epsilon"]["start"]
    else:
        epsilon = config["epsilon"]["start"]
    scores = []
    for episode in range(config["training"]["n_episodes"]):
        state, _ = env.reset()
        score = 0
        for t in range(config["training"]["max_t"]):
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        epsilon = max(
            config["epsilon"]["min"],
            config["epsilon"]["decay"] * epsilon,
        )
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        landed = terminated and reward != -100
        new_row = pd.DataFrame([{
            "score": round(score, 2),
            "average score": round(avg_score, 2),
            "landed ?": landed,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        print(
            f"Episode {episode:>4} | "
            f"Score: {score:>8.2f} | "
            f"Avg: {avg_score:>8.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )
    agent.save(config["checkpoint"]["path"])
    with open("Deep_Q_Learning/epsilon.txt", "w") as f:
        f.write(str(epsilon))
    df.to_csv("Deep_Q_Learning/eval.csv", index=False)
    env.close()

if __name__ == "__main__":
    train()
