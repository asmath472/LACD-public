import torch
from torch.utils.data import Dataset

class GNNNLIDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article1_idx, article2_idx, input_ids, attention_mask, label = self.dataset[idx]
        return {
            'article1_idx': torch.tensor(article1_idx, dtype=torch.long),
            'article2_idx': torch.tensor(article2_idx, dtype=torch.long),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float)
        }