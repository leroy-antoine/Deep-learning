import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import csv
from config import load_config

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("videos", exist_ok=True)

config = load_config()
ENV_NAME = config["env"]["name"]

MODEL_PATH = config["model"]["path"]

N_BINS = tuple(config["discretization"]["bins"])

LOWER_BOUNDS = config["discretization"]["lower_bounds"]
UPPER_BOUNDS = config["discretization"]["upper_bounds"]

LEARNING_RATE = config["q_learning"]["learning_rate"]
DISCOUNT_FACTOR = config["q_learning"]["discount_factor"]

EPSILON_START = config["epsilon"]["start"]
EPSILON_DECAY = config["epsilon"]["decay"]
MIN_EPSILON = config["epsilon"]["min"]

SAVE_EVERY = config["training"]["save_every"]

TRAINING_EPISODES = config["training"]["episodes"]

SOLVED_REWARD = config["training"]["solved_reward"]

PLOT_PATH = config["plots"]["training_curve"]


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

def create_empty_q_table(action_count):
    return np.zeros(N_BINS + (action_count,))


def load_model(action_count):
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        print("Loaded existing model.")
        return data["q_table"], data["epsilon"]
    print("Creating new model.")
    q_table = create_empty_q_table(action_count)
    return q_table, EPSILON_START


def save_model(q_table, epsilon):
    data = {
        "q_table": q_table,
        "epsilon": epsilon
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

def choose_action(env, q_table, state_discrete, epsilon, rng):

    if rng.random() < epsilon:
        return env.action_space.sample()

    return np.argmax(q_table[state_discrete])


def update_q_table(
    q_table,
    state_discrete,
    action,
    reward,
    new_state_discrete
):
    old_value = q_table[state_discrete][action]

    future_max = np.max(q_table[new_state_discrete])

    new_value = (
        old_value
        + LEARNING_RATE
        * (
            reward
            + DISCOUNT_FACTOR * future_max
            - old_value
        )
    )

    q_table[state_discrete][action] = new_value

def run_episode(env, q_table, epsilon, rng):

    state, _ = env.reset()

    state_discrete = discretize_state(state)

    terminated = False
    truncated = False

    total_reward = 0

    while not terminated and not truncated:

        action = choose_action(
            env,
            q_table,
            state_discrete,
            epsilon,
            rng
        )

        new_state, reward, terminated, truncated, _ = env.step(action)

        new_state_discrete = discretize_state(new_state)

        update_q_table(
            q_table,
            state_discrete,
            action,
            reward,
            new_state_discrete
        )

        state_discrete = new_state_discrete

        total_reward += reward

    return total_reward

def save_rewards_csv(rewards, filename="plots/training_rewards.csv"):

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["episode", "reward"])

        for episode, reward in enumerate(rewards):
            writer.writerow([episode, reward])

    print(f"Saved CSV to {filename}")

def plot_rewards(rewards_per_episode):

    mean_rewards = [
        np.mean(rewards_per_episode[max(0, i - 100):(i + 1)])
        for i in range(len(rewards_per_episode))
    ]

    plt.plot(mean_rewards)

    plt.xlabel("Episode")
    plt.ylabel("Mean Reward (last 100)")
    plt.title("Training Performance")

    plt.savefig(PLOT_PATH)

    plt.show()


def train(episodes=TRAINING_EPISODES):

    env = gym.make(ENV_NAME)

    rng = np.random.default_rng()

    q_table, epsilon = load_model(env.action_space.n)

    rewards_per_episode = []

    for episode in range(episodes):

        total_reward = run_episode(
            env,
            q_table,
            epsilon,
            rng
        )

        rewards_per_episode.append(total_reward)

        mean_reward = np.mean(rewards_per_episode[-100:])

        if episode % 100 == 0:

            print(
                f"Episode {episode} | "
                f"Reward: {total_reward:.1f} | "
                f"Mean(100): {mean_reward:.1f} | "
                f"Epsilon: {epsilon:.4f}"
            )

        # Save periodically
        if episode % SAVE_EVERY == 0:

            save_model(q_table, epsilon)

            print("Model saved.")

        # Stop if solved
        if mean_reward > SOLVED_REWARD:

            print("Environment solved!")

            break

        # Decay epsilon
        epsilon = max(
            MIN_EPSILON,
            epsilon * EPSILON_DECAY
        )

    env.close()

    save_model(q_table, epsilon)

    print("Training complete.")
    save_rewards_csv(rewards_per_episode)
    plot_rewards(rewards_per_episode)

if __name__ == "__main__":
    train()