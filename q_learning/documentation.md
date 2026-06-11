# Q-Learning

This is the Q-Learning part of Start Trek. In this part, you will find all the `.py` files needed to run the Q-Learning algorithm on the Lunar Lander environment using Gymnasium.

---

# What is Q-Learning ?

Q-Learning is a value-based Reinforcement Learning (RL) algorithm. It is considered an off-policy algorithm because it learns the optimal policy independently from the actions currently being taken by the agent.

It was introduced by Christopher Watkins in 1989 and became one of the foundational algorithms of Reinforcement Learning.

Unlike Deep Q Learning (DQL), standard Q-Learning does not use neural networks. Instead, it stores all learned values inside a structure called a **Q-table**.

The objective of the algorithm is simple:

Learn which action gives the highest long-term reward for every possible state.

---

# How does it works ?

Q-Learning is based on repeated interaction between an agent and an environment.

The algorithm follows these steps:

1. The agent observes the current state.
2. The state is discretized into bins to reduce the size of the state space.
3. The agent selects an action using an epsilon-greedy policy.
4. The action is executed in the environment.
5. The environment returns:
   - the next state
   - a reward
   - whether the episode ended
6. The Q-table is updated using the Bellman Equation.
7. Restart from step 1 until the episode ends.

Over time, the agent progressively learns which actions maximize future rewards.

---

# Deep explanation of each part

---

# A : The Q Value

The Q-value `Q(s, a)` represents the expected cumulative reward obtained by taking action `a` in state `s` and then following the best possible policy afterward.

The goal of Q-Learning is to estimate these Q-values as accurately as possible.

The update rule is:

```python
Q(s, a) = Q(s, a) + alpha * (
    reward + gamma * max(Q(next_state)) - Q(s, a)
)
```

Where:

- `alpha` = learning rate
- `gamma` = discount factor
- `reward` = immediate reward
- `max(Q(next_state))` = best future reward estimate

This formula progressively improves the agent's knowledge of the environment.

---

# B : The Q-Table

Unlike DQL, Q-Learning does not use neural networks.

Instead, all learned values are stored inside a multidimensional NumPy array called the **Q-table**.

Each entry represents:

```text
(state, action) -> expected reward
```

For Lunar Lander, the environment state is continuous, meaning values can theoretically be infinite.

To make Q-Learning possible, the continuous state space is discretized into bins.

Example:

```python
N_BINS = (8, 8, 8, 8, 8, 8, 2, 2)
```

This converts continuous observations into discrete indices usable by the Q-table.

The Q-table is saved using Python `pickle` so training can resume later without losing progress.

---

# C : State Discretization

The Lunar Lander environment provides floating-point observations:

```text
x_position
y_position
x_velocity
y_velocity
angle
angular_velocity
left_leg_contact
right_leg_contact
```

Since Q-Learning cannot directly handle continuous spaces efficiently, each value is mapped into discrete intervals called bins.

Example:

```python
x_velocity = 0.08
```

might become:

```python
bin_index = 5
```

The final discretized state becomes a tuple:

```python
(4, 6, 3, 5, 2, 4, 1, 0)
```

This tuple is then used as an index inside the Q-table.

Discretization is one of the most important parts of tabular Q-Learning because it controls:

- memory usage
- learning speed
- precision of learned behaviors

---

# D : Epsilon-Greedy Policy

The agent needs to balance:

- exploration
- exploitation

To achieve this, the algorithm uses an epsilon-greedy policy.

With probability epsilon:

```text
Choose a random action
```

With probability `1 - epsilon`:

```text
Choose the best known action
```

At the beginning of training:

```python
epsilon = 1.0
```

meaning the agent explores almost completely randomly.

Over time, epsilon decays:

```python
epsilon *= decay
```

allowing the agent to progressively exploit learned knowledge.

A minimum epsilon is kept to avoid complete loss of exploration.

---

# E : Reward Maximization

The objective of the agent is to maximize cumulative rewards.

In Lunar Lander:

- smooth landings give positive rewards
- crashing gives large negative rewards
- fuel usage slightly penalizes the agent
- successful landing gives a large reward bonus

The Q-table progressively learns which actions maximize total future reward instead of immediate reward only.

This long-term optimization is controlled by the discount factor gamma.

---

# Hyperparameters

---

# A : env

- `name` = Gymnasium environment used by the agent.
- `video.render_mode` = rendering mode used during evaluation.
- `video.folder` = folder where evaluation videos are stored.

---

# B : q_learning

- `learning_rate` = controls how quickly the Q-table updates.
  - High values = faster learning but less stable.
  - Low values = slower but more stable learning.

- `discount_factor` = importance of future rewards.
  - Close to `1` = agent values long-term rewards.
  - Close to `0` = agent focuses only on immediate rewards.

---

# C : epsilon

- `start` = initial exploration rate.
- `min` = minimum exploration rate.
- `decay` = multiplicative epsilon decay after each episode.

Example:

```python
epsilon = epsilon * decay
```

A value close to `1` produces slower decay and longer exploration.

---

# D : discretization

- `bins` = number of intervals used for each state variable.
- `lower_bounds` = minimum accepted values.
- `upper_bounds` = maximum accepted values.

Increasing the number of bins increases precision but also increases:

- memory usage
- training duration

---

# E : training

- `episodes` = total number of training episodes.
- `save_every` = interval between automatic model saves.
- `solved_reward` = reward threshold considered as solving the environment.

---

# F : evaluation

- `episodes` = number of evaluation episodes executed after training.

---

# G : model

- `path` = location where the trained Q-table is saved using pickle.

If the model already exists when training starts, it is automatically loaded so training can continue from previous progress.

---

# H : plots

- `training_curve` = path where the training reward graph is saved.

The graph displays the moving average reward over the last 100 episodes.

---

# Project Structure

```text
project/
│
├── train.py
├── eval.py
├── config.py
├── config.yaml
│
├── models/
├── plots/
└── videos/
```

---

# Reproducibility

The project includes a reproduction script:

```bash
./reproduce.sh
```

This script automatically:

1. trains the model
2. evaluates the trained agent
3. regenerates plots
4. regenerates videos
5. exports CSV metrics

This guarantees reproducibility of the experimental results.

---

# Conclusion

Q-Learning solves Reinforcement Learning problems by progressively estimating the value of actions for every state of the environment.

Unlike Deep Q Learning, it does not require neural networks, replay buffers, or GPUs. Instead, it relies on a discretized representation of the environment stored inside a Q-table.

The main strengths of Q-Learning are:

- simplicity
- interpretability
- stability on small state spaces

However, its main limitation is scalability: very large or continuous environments quickly make the Q-table too large to manage efficiently.

Despite these limitations, Q-Learning remains one of the most important and educational Reinforcement Learning algorithms and serves as the foundation for more advanced approaches such as Deep Q Learning (DQL).