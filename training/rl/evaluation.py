import os 
import torch 
from models.rl.agent import Agent
from utils.config_loader import LLM_CONFIG

allowed_words = ["chickpea", "bean", "lentil", "rice"]

test_prompts = [
    "Which is better bean or lentil?",
    "Which is better chickpea or rice?",
    "If you had to pick between rice or bean, what would you choose?",
    "Do you prefer lentil or chickpea?"
]

checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "rl_model.pth")

agent = Agent(checkpoints_path=checkpoint_path, allowed_words=allowed_words)
checkpoint = torch.load(checkpoint_path, map_location=agent.device)
agent.model.load_state_dict(checkpoint["model_state_dict"])

for prompt in test_prompts:
    action, log_prob, probs = agent.act(prompt)
    print(f"Prompt: {prompt}\nAction: {action}\nProbabilities: {probs}")


    print(f"Prompt: {prompt}")
    print(f"→ Action chosen: {action}")
    print("-" * 50)