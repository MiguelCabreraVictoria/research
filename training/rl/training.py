import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from models.rl.agent import Agent
from utils.config_loader import LLM_CONFIG


# Se aplica una semilla para reproducibilidad
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

loss_history = []
avg_rewards = []
avg_rewards_bandit_1 = []
avg_rewards_bandit_2 = []




num_epochs = 100
bandit_1 = ["bean", "chickpea"]
bandit_2 = ["lentil", "rice"]
allowed_words = bandit_1 + bandit_2

prompts_bandit_1 = [
    "Which is better chickpea or bean?",
    "Do you like chickpea or bean more?",
    "Which one do you prefer bean or chickpea?",
    "What is better bean or chickpea?",
    "If you had to choose, chickpea or bean?",
    "Pick one: bean or chickpea.",
    "bean vs chickpea: which do you think is better?",
]

prompts_bandit_2 = [
    "Which is better lentil or rice?",
    "Do you like lentil or rice more?",
    "Which one do you prefer rice or lentil?",
    "What is better rice or lentil?",
    "If you had to choose, lentil or rice?",
    "Pick one: lentil or rice.",
    "lentil vs rice: which do you think is better?",
]

checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "model_and_optimizer.pth")


agent = Agent(checkpoints_path=checkpoint_path, allowed_words=allowed_words)

optimizer = torch.optim.AdamW(
    agent.model.parameters(),
    lr=LLM_CONFIG['training']['learning_rate'],
    weight_decay=LLM_CONFIG['training']['weight_decay']
)
optimizer.load_state_dict(agent.checkpoints['optimizer_state_dict'])



for epoch in range(num_epochs):
    agent.model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_reward = 0.0
    total_reward_bandit_1 = 0.0
    total_reward_bandit_2 = 0.0
    n1 = len(prompts_bandit_1)
    n2 = len(prompts_bandit_2)
    n_prompts = 0

    for prompt in prompts_bandit_1:
        action, log_prob, _ = agent.act(prompt)
        first_option_factor = np.random.choice([1, 0], p=[0.75, 0.25])
        second_option_factor = np.random.choice([1, 0], p=[0.25, 0.75])
        base_reward = 10.0
        if action == bandit_1[0]:
            reward = first_option_factor * base_reward
        else:
            reward = second_option_factor * base_reward

        loss = agent.policy_loss(log_prob, reward)
        total_loss += loss
        total_reward += reward
        total_reward_bandit_1 += reward
        n_prompts += 1

    for prompt in prompts_bandit_2:
        action, log_prob, _ = agent.act(prompt)
        first_option_factor = np.random.choice([1, 0], p=[0.75, 0.25])
        second_option_factor = np.random.choice([1, 0], p=[0.25, 0.75])
        base_reward = 1.0
        if action == bandit_2[0]:
            reward = first_option_factor * base_reward
        else:
            reward = second_option_factor * base_reward
        loss = agent.policy_loss(log_prob, reward)
        total_loss += loss
        total_reward += reward
        total_reward_bandit_2 += reward
        n_prompts += 1

    total_prompts = n1 + n2
    total_loss /= n_prompts
    avg_reward = total_reward / total_prompts

    loss_history.append(total_loss.item())
    avg_rewards.append(avg_reward)
    avg_rewards_bandit_1.append(total_reward_bandit_1 / n1)
    avg_rewards_bandit_2.append(total_reward_bandit_2 / n2)


    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Total loss: {total_loss.item():.4f}, Average reward: {avg_reward:.4f}")

plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Epoch')
plt.title('Training Loss Over Epochs - Policy Gradient')
plt.grid()
plt.savefig(os.path.join(os.getcwd(), "images","rl","training_loss.png"))

plt.plot(avg_rewards, label="Average Reward (All Bandits)", color='purple')
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("Overall Average Reward Over Epochs")
plt.grid(True)
plt.savefig(os.path.join("images", "rl", "avg_total_reward.png"))

plt.plot(avg_rewards_bandit_1, label="Bandit 1 (bean vs chickpea)", color='green')
plt.plot(avg_rewards_bandit_2, label="Bandit 2 (lentil vs rice)", color='orange')
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("Average Reward per Bandit Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("images", "rl", "reward_per_bandit.png"))

# test_prompts = [
#     "Which is better bean or lentil?",
#     "Which is better chickpea or rice?",
# ]

# for prompt in test_prompts:
#     action, log_prob, probs = agent.act(prompt)
#     print(f"Prompt: {prompt} | Action: {action} | Probabilities: {probs}")

save_path = os.path.join(os.getcwd(), "checkpoints", "rl_model.pth")
print("Guardando checkpoints actualizados")
torch.save({
    'model_state_dict': agent.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    "epochs": num_epochs
}, save_path)