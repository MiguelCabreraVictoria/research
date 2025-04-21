import torch
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, cfg):
        self.input_ids: list = []
        self.target_ids: list = []

        # tokenizer (Byte Pair Encoding)
        tokens_ids = tokenizer.encode(txt)

        # sliding window
        for i in range(0, len(tokens_ids) - cfg["max_length"], cfg["stride"]):
            input_chunks = tokens_ids[i: i + cfg["max_length"]]
            target_chunks = tokens_ids[i + 1: i + cfg["max_length"] + 1]

            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
