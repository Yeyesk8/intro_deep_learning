import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset
import numpy as np

# Load dataset
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
dataset = dataset['train']

# Define the PyTorch-compatible wrapper
class TFToTorchDataset(Dataset):
    def __init__(self, tf_dataset):
        self.data = list(tf_dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = image.numpy()
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # HWC -> CHW and normalize
        label = torch.tensor(label).long()
        return image, label

# Wrap the dataset
torch_dataset = TFToTorchDataset(dataset)

# Use with DataLoader
dataloader = torch.utils.data.DataLoader(torch_dataset, batch_size=32, shuffle=True)