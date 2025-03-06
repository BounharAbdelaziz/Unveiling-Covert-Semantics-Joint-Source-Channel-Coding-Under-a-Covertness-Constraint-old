import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

class IIDDataset(Dataset):

    def __init__(self, k=32, n_samples=60000, p=0.65) -> None:
        # Size of each sequence
        self.k = k
        # Number of iid sequences
        self.n_samples = n_samples
        # Parameter of the Bernoulli distribution
        self.p = p
        # where to save the dataset
        self.path_saving = f'./datasets/k_{k}_n_{n_samples}_p_{p}_iid_dataset.csv'
        # Load the dataset if it exists, otherwise generate it
        if not os.path.exists(self.path_saving):
            print(f'[INFO] Dataset not found at {self.path_saving}, generating a new one...')
            self.data = self.generate_dataset()
        else:
            print(f'[INFO] Dataset found at {self.path_saving}, loading it...')
            self.data = pd.read_csv(self.path_saving).to_numpy()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        dummy_label = 0 # just so that the DataLoader loop works
        return torch.tensor(sample, dtype=torch.float), dummy_label

    def generate_bernoulli_sequence_torch(self):
        """
        Generate a sequence of size k from a Bernoulli distribution with parameter p using PyTorch.
        """
        return torch.bernoulli(torch.full((self.k,), self.p)).int()

    def generate_dataset_torch(self):
        """
        Generate a dataset of N iid sequences of size k from a Bernoulli distribution with parameter p using PyTorch.
        """
        dataset = torch.zeros(self.n_samples, self.k, dtype=torch.int)
        for i in range(self.n_samples):
            dataset[i] = self.generate_bernoulli_sequence_torch()

        # Print some information about the dataset
        print(f'[INFO] Dataset shape: {dataset.shape},')
        return dataset
    
    def save_dataset_to_csv(self, dataset, filename='iid_dataset.csv'):
        """
        Save the dataset to a CSV file.
        """
        df = pd.DataFrame(dataset.numpy())
        df.to_csv(filename, index=False)

    def generate_dataset(self):
        """
        Generate a dataset of N iid sequences of size k from a Bernoulli distribution with parameter p.
        """
        # Generate the dataset
        iid_dataset_torch = self.generate_dataset_torch()
        self.save_dataset_to_csv(iid_dataset_torch, self.path_saving)

        return iid_dataset_torch