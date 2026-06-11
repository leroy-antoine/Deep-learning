import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import pickle
from config import load_config
import os

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("videos", exist_ok=True)

config = load_config()
ENV_NAME = config["env"]["name"]

MODEL_PATH = config["model"]["path"]

N_BINS = tuple(config["discretization"]["bins"])

LOWER_BOUNDS = config["discretization"]["lower_bounds"]
UPPER_BOUNDS = config["discretization"]["upper_bounds"]

VIDEO_ENABLED = config["video"]["enabled"]
VIDEO_FOLDER = config["video"]["folder"]
RENDER_MODE = config["video"]["render_mode"]

EVAL_EPISODES = config["evaluation"]["episodes"]

def get_bin(value, low, high, bins):

    value = min(high, max(low, value))

    if value == high:
        return bins - 1

    return int((value - low) / (high - low) * bins)


def discretize_state(state):

    return tuple(
        get_bin(value, low, high, bins)
        for value, low, high, bins in zip(
            state,
            LOWER_BOUNDS,
            UPPER_BOUNDS,
            N_BINS
        )
    )

def load_q_table():

    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    return data["q_table"]

def run_episode(env, q_table):

    state, _ = env.reset()

    state_discrete = discretize_state(state)

    terminated = False
    truncated = False

    total_reward = 0

    while not terminated and not truncated:

        action = np.argmax(q_table[state_discrete])

        new_state, reward, terminated, truncated, _ = env.step(action)

        state_discrete = discretize_state(new_state)

        total_reward += reward

    return total_reward


def evaluate(episodes=EVAL_EPISODES, record_video=VIDEO_ENABLED):
    render_mode = (
        RENDER_MODE
        if record_video
        else "human"
    )
    env = gym.make(
        ENV_NAME,
        render_mode=render_mode
    )

    if record_video:

        env = RecordVideo(
            env,
            video_folder=VIDEO_FOLDER,
            episode_trigger=lambda episode: True
        )

    q_table = load_q_table()

    for episode in range(episodes):

        total_reward = run_episode(env, q_table)

        print(
            f"Episode {episode + 1} | "
            f"Reward: {total_reward:.2f}"
        )

    env.close()

if __name__ == "__main__":

    evaluate(
        episodes=5,
        record_video=True
    )