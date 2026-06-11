
# Proximal Policy Optimization

This is the PPO's part of Start Trek. In this part, you will find all the .py files needed to run the PPO algorithm


# What is Proximal Policy Optimization (PPO)

PPO is a policy gradient Renforcement Learning (RL) algorithm. 
It is an on-policy and actor critic algorithm.

Since 2018, PPO s the default RL algorithm at OpenAI. 
PPO has been applied to many areas, such as controlling a robotic arm, beating professional players at Dota 2 (OpenAI Five), and playing Atari games.

# How does it works ?

PPO is quite a simple algorithm. 
1. First, it will collect datas for N steps. The neural network will take actions and return the percentage of chance of each actions during these steps.
2. Then, the other neural network will evaluate each state, and the algorithm will calculate the Advantage.
3. Finally, the policy will be updated after the clipping of the values. The Actor neural network will see its actions probabilities changed, and the Critic network will adapt its estimations.
4. Restart to step 1.

# Deep explanation of each part

# A : The policy

Since the PPO is an on-policy algorithm, we need to understand and implement a policy. 
The policy is simply the function wich for each state will give us the probabilities of perform each actions. 
For example, for the Lunar Lander algorithm, you will have something like [right : 10%, left : 70%, main : 10%, nothing : 10%] at a state N.

# B : Neural networks

There are two neural networks in PPO : the "Actor Network" and the "Critic Network".

- The Actor Network (AN) takes the decisions. It observes the environment, then gives the probabilities of realize each possible actions.
Here, we trained it with Lunar Lander, so it will give something like [right : 10%, left : 70%, main : 10%, nothing : 10%].

- The Critic Network (CN) evaluate how many points it will accumulate before the end of the episode. To process that "score", it will take into account its actual state.

Theses neural networks works in parallel.

# C : The Advantage

The Advantage is the center of PPO algorithm. It answers the question "Was this action better or worse than what was predicted by the Critic Network?"
If the Advantage is positive, then it means that the action take by the AN is better than what was predicted, so we encourage it.
Otherwise, we discourage it.
So, if the Advantage is positive, the probability of taking the right decision increases, or the other way if the Advantage is negative.

If you want to learn more about PPOs Advantage, I recommend you to read this documentation :
https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d

# D : The Clip

Clipping and Advantage are the reasons why the PPO is game changer. 

PPO algorithm measure how much the new policy is far away from the last policy. It's a ratio calculated this way : new probability to perform each action / last probability to perform each action.


If the new policy is to far away from the last one, The clipping algorithm will smooth out any changes made to the old policy so that the new policy remains within a range of ±20%, depending on the "policy_clip" hyperparamter.

For example, if the last policy gave us these probabilities : [right : 10%, left : 70%, main : 10%, nothing : 10%], and if the new one is [right : 20%, left : 60%, main : 10%, nothing : 10%] before the clipping, then, after clipping, it will look like that : [right : 12%, left : 68%, main : 10%, nothing : 10%].
Clipping prevents big gaps between the politics, so that the datas are smoother and more coherent.

# Hyperparamters

# A : env

- name = the Gymnasium environment used (here, Lunar Lander v3)
- wind = true or false value, so that you can easily add the wind parameter to the Gymansium environment.
- wind_power = a float, you can change its value to lower or increase the power of the wind.
- turbulence_power = same as wind_power, but for the turbulence.
- render_mode = the render mode for Gymnasium, I advise you not to touch it.
- video_folder = the folder where you want to store the videos.

# B : agent

- gamma = it's the factor of discount of the advantage calculus.
- gae_lambda = it's the lambda of the GAE Advantage algorithm. Close to 1 = high variance, Close to 0 = less variance but a lot of bias, close to 1 = more variance but no bias.
- policy_clip = clipping ration. Its value is "0.1", so the new policy can't more than/lesser than 10% different than the other.
- fc1_dims = number of neurons of the first layer of the neural networks.
- fc2_dims = number of neurons of the second layer of the neural networks.

If you lower fc1 or fc2, the model will train faster, but its performances will drop. If you increase these values, the network could theoricaly learn more complex behavior, but the training will be much slower and the model might overfit.  
I recommend you to not change these values (256x256), since it's considered as the "standard choice" for Lunar Lander.

# C : train

- N = the number of steps before each update (called collect horizon).
- batch_size = mini batches dividing the N by batch_size. For the actual values it would be 2048 / 64 = 32 mini batches per epoch.
- n_epochs = number of passes over the data collected with each update. PPO reuses the data multiple times, unlike algorithms such as A2C.
- alpha = the learning rate of Actor and Critic. Lower values = more stabilized learning. Bigger values = learns more but more changes on the models.
- n_train = total number of training sessions.

# D : eval

- n_episodes = number of episodes to evaluate the trained model. 100 is enough to get stables stats.

# Conclusion

PPO is a powerful algorithm which solves the fundamental problem of on-policy algorithms: how to learn effectively without straying too far from the current policy?

The answer lies in three complementary mechanisms:
- GAE calculates a precise Advantage with a good bias-variance trade-off thanks to 'gae_lambda'.
- Clipping prevents overly abrupt updates thanks to 'policy_clip'.
- Multi-epoch mini-batches allow us to extract maximum information from each data collection without destabilizing the learning process.

It is this combination that has made it OpenAI’s go-to algorithm since 2018, capable of adapting to everything from a robotic arm to a LunarLander agent operating in wind and turbulence.
