import os

from torch.utils.data import Dataset
import torch


class LabelledCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 label_path_template,
                 feature_path_template,
                 file_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.feature_path_template = feature_path_template
        self.handler = file_handler

    def __len__(self):
        return self.n_sims
    
    def __getitem__(self, sim_idx):
        label_path = self.label_path_template.format(sim_idx=sim_idx)
        label = self.handler.read(label_path)
        label_tensor = torch.as_tensor(label)

        features = []
        for freq in self.freqs:
            feature_path = self.feature_path_template.format(sim_idx=sim_idx, freq=freq)
            features.append(self.handler.read(feature_path))
        features_tensor = [torch.as_tensor(f) for f in features]
        features_tensor = torch.cat(features_tensor, dim=0)
        return features_tensor, label_tensor, sim_idx


class TestCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 feature_path_template,
                 file_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.handler = file_handler

    def __len__(self):
        return self.n_sims
    
    def __getitem__(self, sim_idx):
        features = []
        for freq in self.freqs:
            feature_path = self.feature_path_template.format(sim_idx=sim_idx, freq=freq)
            features.append(self.handler.read(feature_path))
        features_tensor = [torch.as_tensor(f) for f in features]
        features_tensor = torch.cat(features_tensor, dim=0)
        return features_tensor, sim_idx
