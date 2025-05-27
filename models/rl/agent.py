import torch
import torch.nn.functional as F
import tiktoken
from models.llm.model.gpt_model import GPTModel
from utils.config_loader import LLM_CONFIG

class Agent:
    def __init__(self, checkpoints_path, allowed_words, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tiktoken.get_encoding(LLM_CONFIG['dataset']["encoder"])

        self.model = GPTModel(LLM_CONFIG['model'])
        self.checkpoints = torch.load(checkpoints_path, map_location=self.device)

        self.model.load_state_dict(self.checkpoints['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.allowed_words = allowed_words
        self.allowed_ids = [self.tokenizer.encode(word)[0] for word in allowed_words]

    def generate_response(self, prompt):
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        output = self.model(input_tensor)
        
        return output[0, -1, :]  # ultimo token

    def sample_action(self, logits, prompt):
        prompt = prompt.lower()
        valid_word_indices = [i for i, word in enumerate(self.allowed_words) if word in prompt]

        if not valid_word_indices:
            valid_word_indices = list(range(len(self.allowed_words)))
        
        valid_ids = [self.allowed_ids[i] for i in valid_word_indices]
        allowed_logits = logits[valid_ids]
        probs = F.softmax(allowed_logits, dim=0)
        log_probs = torch.log(probs)
        idx = torch.multinomial(probs, num_samples=1).item()
        chosen_word = self.allowed_words[valid_word_indices[idx]]
        return chosen_word, log_probs[idx], probs

    def act(self, prompt):
        logits = self.generate_response(prompt)
        action, log_prob, probs = self.sample_action(logits, prompt)
        return action, log_prob, probs

    def policy_loss(self, log_prob, reward):
        # Algoritmo REINFORCE
        return -log_prob * reward
