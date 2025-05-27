import os
import torch
import matplotlib.pyplot as plt
from models.rl.agent import Agent
from utils.config_loader import LLM_CONFIG

loss_history = []



num_epochs = 50
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
    n_prompts = 0

    for prompt in prompts_bandit_1:
        action, log_prob, _ = agent.act(prompt)
        reward = 1.0 if action in bandit_1 else -1.0
        loss = agent.policy_loss(log_prob, reward)
        total_loss += loss
        n_prompts += 1

    for prompt in prompts_bandit_2:
        action, log_prob, _ = agent.act(prompt)
        reward = 1.0 if action in bandit_2 else -0.1
        loss = agent.policy_loss(log_prob, reward)
        total_loss += loss
        n_prompts += 1

    total_loss /= n_prompts
    loss_history.append(total_loss.item())

 
    total_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Total loss: {total_loss.item():.4f}")

plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Epoch')
plt.title('Training Loss Over Epochs - Policy Gradient')
plt.savefig(os.path.join(os.getcwd(), "images","rl","training_loss.png"))

# test_prompts = [
#     "Which is better bean or lentil?",
#     "Which is better chickpea or rice?",
# ]

# for prompt in test_prompts:
#     action, log_prob, probs = agent.act(prompt)
#     print(f"Prompt: {prompt} | Action: {action} | Probabilities: {probs}")

save_path = os.path.join(os.getcwd(), "checkpoints", "rl_model.pth")
torch.save({
    'model_state_dict': agent.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    "epochs": num_epochs
}, save_path)
