import os

from torch.utils.data import Dataset
import torch


class LabelledCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 detectors,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.detectors = detectors
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler

    def __len__(self):
        return self.n_sims
    
    def __getitem__(self, sim_idx):
        label_path = self.label_path_template.format(sim_idx=sim_idx)
        label = self.label_handler.read(label_path)
        label_tensor = torch.as_tensor(label)

        feature_path_template = self.feature_path_template.format(sim_idx=sim_idx, det="{det}")
        features = self.feature_handler.read(feature_path_template)
        features_tensor = tuple([torch.as_tensor(f) for f in features.values()])
        features_tensor = torch.cat(features_tensor, dim=0)
        return features_tensor, label_tensor, sim_idx


class TestCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 detectors,
                #  label_path_template,
                #  label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.detectors = detectors
        # self.label_path_template = label_path_template
        # self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler

    def __len__(self):
        return self.n_sims
    
    def __getitem__(self, sim_idx):
        # label_path = self.label_path_template.format(sim_idx=sim_idx)
        # label = self.label_handler.read(label_path)
        # label_tensor = torch.as_tensor(label)

        feature_path_template = self.feature_path_template.format(sim_idx=sim_idx, det="{det}")
        features = self.feature_handler.read(feature_path_template)
        features_tensor = tuple([torch.as_tensor(f) for f in features.values()])
        features_tensor = torch.cat(features_tensor, dim=0)
        return features_tensor, sim_idx