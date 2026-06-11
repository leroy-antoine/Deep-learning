# Deep Q Learning

This is the DQL's part of Start Trek. In this part, you will find all the .py files needed to run the DQL algorithm


# What is Deep Q Learning (DQL)

DQL is a value-based Reinforcement Learning (RL) algorithm. It is an off-policy algorithm, meaning it can learn from experiences that were collected under a different policy than the one currently being trained.

It was introduced by DeepMind in 2013 and became famous for beating human-level performance on Atari games using only raw pixels as input.

# How does it works ?

DQL is built around the concept of Q-values: for each state, the network estimates how much total reward the agent can expect by taking each possible action.

1. The agent observes the current state and picks an action using an epsilon-greedy policy (mostly the best known action, sometimes a random one).
2. The action is executed in the environment, and the transition (state, action, reward, next_state, done) is stored in a Replay Memory.
3. Every update_every steps, a random mini-batch is sampled from the memory and used to train the local Q-network.
4. The target Q-network is slowly updated toward the local network using a soft update.
5. Restart to step 1.

# Deep explanation of each part

# A : The Q Value

The Q-value Q(s, a) represents the expected cumulative reward when taking action a in state s and following the optimal policy afterward.
The network learns to predict Q-values for every action at once, and the agent simply picks the action with the highest Q-value.

# B : Neural networks

There are two neural networks in DQL : the Local Q-network and the Target Q-network.

- The Local Q-network is the one being actively trained. It takes the current state as input and outputs a Q-value for each possible action.
- The Target Q-network has the same architecture but its weights are updated slowly (via soft update with parameter tau). It is used to compute stable Q-value targets during training, preventing the moving-target problem that would occur if we used the same network for both prediction and target.

These two networks work together to stabilize training.

# C : The Replay Memory

The Replay Memory (or Experience Replay Buffer) stores past transitions seen by the agent. At each training step, a random mini-batch is sampled from it.
This breaks the temporal correlation between consecutive experiences, which would otherwise cause the network to overfit to recent events and forget older ones.
The buffer has a fixed capacity: when full, the oldest experiences are discarded.

# D : Epsilon-Greedy Policy

Since the agent needs to explore the environment to discover good strategies, it uses an epsilon-greedy policy:

- With probability epsilon, it picks a random action (exploration).
- With probability 1 - epsilon, it picks the action with the highest Q-value (exploitation).

Epsilon starts high (full exploration) and decays over time so the agent gradually shifts toward exploiting what it has learned.

# E : Soft Update

Instead of copying the local network weights directly into the target network (which would be too abrupt), DQL uses a soft update:
```python
target_weights = tau * local_weights + (1 - tau) * target_weights
```
With a small tau (e.g. 0.001), the target network changes very slowly, keeping training stable.

# Hyperparamters

# A : env

- name = the Gymnasium environment used (here, Lunar Lander v3)
- render_mode = the render mode for Gymnasium, I advise you not to touch it.
- video_folder = the folder where you want to store the videos.

# B : agent

- gamma = discount factor applied to future rewards. Close to 1: the agent values long-term rewards. Close to 0: the agent is short-sighted.
- tau = soft update coefficient for the target network. A small value (0.001) keeps the target network stable.
- lr = learning rate of the optimizer (Adam). Lower values: more stable but slower learning.

# C : memory

- buffer_size = maximum number of transitions stored in the Replay Memory. Older experiences are discarded when the buffer is full.
- batch_size = number of transitions sampled from the memory at each training step.

# D : epsilon

- start = initial value of epsilon (1.0 = fully random at the beginning).
- min = minimum value epsilon can decay to. The agent will always keep a small amount of exploration.
- decay = multiplicative decay factor applied after each episode. A value close to 1 (e.g. 0.999) decays slowly.

# E : training

- n_episodes = total number of training episodes.
- max_t = maximum number of steps per episode before it is forced to end.
- update_every = number of steps between each training update of the local Q-network.

# F : eval

- n_episodes = number of episodes used to evaluate the trained model. 100 is enough to get stable statistics.

# G : checkpoint

- path = path where the trained model weights are saved (e.g. dql.pth). If a checkpoint already exists when training starts, it will be loaded automatically to resume training.

# Conclusion

DQL solves a fundamental challenge in reinforcement learning: how to train a neural network to predict Q-values in a stable way?
The answer lies in three complementary mechanisms:

- Experience Replay breaks temporal correlations and allows the agent to reuse past data efficiently.
- The Target Network provides stable training targets and prevents the moving-target problem.
- Epsilon-Greedy decay balances exploration and exploitation over the course of training.

Together, these mechanisms allow DQL to learn complex behaviors directly from raw state observations, making it one of the foundational algorithms of modern deep reinforcement learning.
