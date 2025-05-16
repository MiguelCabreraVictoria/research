import torch
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, cfg):
        self.input_ids = []
        self.target_ids = []

        # Tokenizar todo el texto de entrada en una lista de IDs
        tokens_ids = tokenizer.encode(txt)

        max_len = cfg["max_length"]
        stride = cfg["stride"]
        total_len = len(tokens_ids)

        # Crear las ventanas con sliding window
        for i in range(0, total_len - max_len, stride):
            input_chunk = tokens_ids[i : i + max_len]
            # Target es el siguiente token para cada token en input
            target_chunk = tokens_ids[i + 1 : i + max_len + 1]

            # Puede pasar que el target_chunk sea un token menos si estamos al final
            # Por eso verificamos y ajustamos para que input y target tengan la misma longitud
            if len(target_chunk) < max_len:
                target_chunk.append(tokenizer.eot_token_id if hasattr(tokenizer, "eot_token_id") else 0)  # o cualquier token PAD

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]