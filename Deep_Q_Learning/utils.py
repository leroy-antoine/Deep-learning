import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def learn(
    local_qnetwork,
    target_qnetwork,
    optimizer,
    experiences,
    gamma
):
    states, actions, rewards, next_states, dones = experiences
    next_q = target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + gamma * next_q * (1 - dones)
    q_expected = local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(
        target_model.parameters(),
        local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data +
            (1.0 - tau) * target_param.data
        )

def plot_learning_curve(scores, file_name):
    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i - 100):i + 1])
    plt.plot(running_avg)
    plt.title("Running average (100 episodes)")
    plt.savefig(file_name)