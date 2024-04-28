import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, values):
        super(Dataset, self).__init__()
        self.values = values
    def __len__(self):
        return len(self.values)
    def __getitem__(self, index):
        return self.values[index]

