import os
import torch
import tiktoken
from utils.config_loader import LLM_CONFIG
from models.llm.model.gpt_model import GPTModel
import torch.nn.functional as F


"""
Cargar el modelo y el tokenizador
Preparacion del prompt
Inferencia y calculo de probabilidades
 - Generar logits
 - Extraer logits de palabras permitidas
 - Se puede usar softmax para obtener probabilidades
"""

checkpoint_path = os.path.join(os.getcwd(), "checkpoints", "model_and_optimizer.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)

model = GPTModel(LLM_CONFIG['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding(LLM_CONFIG['dataset']["encoder"])

prompt = "Which is better chickpea or bean ? "
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids], device=device)

# Palabras permitidas y sus IDs
allowed_words = ["chickpea", "bean", "lentil", "rice"]
allowed_ids = [tokenizer.encode(word)[0] for word in allowed_words]

with torch.no_grad():
    output = model(input_tensor)
    logits = output[0, -1, :]  # (vocab_size,)

    # Palabra más probable de TODO el vocabulario
    best_token_id = torch.argmax(logits).item()
    best_token = tokenizer.decode([best_token_id])
    print(f"Token más probable (sin restricción): {best_token}")

    allowed_logits = logits[allowed_ids]
    probs = F.softmax(allowed_logits, dim=0)

    for word, prob in zip(allowed_words, probs):
        print(f"{word}: {prob.item():.4f}")

    # Muestreo 
    sampled_idx = torch.multinomial(probs, num_samples=1).item()
    sampled_word = allowed_words[sampled_idx]

print(f"Prompt: {prompt}")
print(f"Sampled response: {sampled_word}")


