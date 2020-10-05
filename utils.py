import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def make_dataloader(X, y, batch_size):
    data = np.concatenate([X, y], axis=1)
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=batch_size)
    return train_loader