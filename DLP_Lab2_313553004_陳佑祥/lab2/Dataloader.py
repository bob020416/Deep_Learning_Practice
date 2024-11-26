import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MIBCI2aDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = self._getFeatures(features_path)
        self.labels = self._getLabels(labels_path)
        if len(self.features) != len(self.labels):
            raise ValueError("The number of features does not match the number of labels.")

    def _getFeatures(self, filePath):
        data = []
        for file_name in os.listdir(filePath):
            if file_name.endswith('.npy'):
                file_path = os.path.join(filePath, file_name)
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Feature file not found: {file_path}")
                array = np.load(file_path)
                data.append(array)
        if not data:
            raise ValueError(f"No data found in directory: {filePath}")
        return np.concatenate(data, axis=0)

    def _getLabels(self, filePath):
        labels = []
        for file_name in os.listdir(filePath):
            if file_name.endswith('.npy'):
                file_path = os.path.join(filePath, file_name)
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Label file not found: {file_path}")
                array = np.load(file_path)
                labels.append(array)
        if not labels:
            raise ValueError(f"No labels found in directory: {filePath}")
        return np.concatenate(labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

