import torch

class Perceptron(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.flatten = torch.nn.Flatten() # convert matrix to array
        self.device = device # use provided device
        self.step = lambda x: 1 if x >= 0 else -1 # perceptron step function
