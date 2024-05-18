from typing import List, Callable, Optional
import os

from torch.utils.data import Dataset
import torch
from core.asset_handlers.asset_handlers_base import GenericHandler


class TrainCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims: int,
                 freqs: List[int],
                 map_fields: str,
                 feature_path_template: str,
                 file_handler: GenericHandler,
                 label_path_template: str = None,
                 transforms: Optional[List[Callable]]=[]
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.feature_path_template = feature_path_template
        self.handler = file_handler
        self.n_map_fields:int = len(map_fields)
        self.transforms = transforms

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.handler,
                                     n_map_fields=self.n_map_fields,
                                     sim_idx=sim_idx)
        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.handler,
                               n_map_fields=self.n_map_fields,
                               sim_idx=sim_idx)
        data = (features, label)
        if self.transforms:
            try:
                for transform in self.transforms:
                    data = transform(data)
            except AttributeError:
                data = transform(data)
        return data


class TestCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims: int,
                 freqs: List[int],
                 map_fields: str,
                 feature_path_template: str,
                 file_handler: GenericHandler,
                 transforms: Optional[List[Callable]]=[]
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.handler = file_handler
        self.n_map_fields:int = len(map_fields)
        self.transforms = transforms

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.handler,
                                     n_map_fields=self.n_map_fields,
                                     sim_idx=sim_idx)
        data = features
        if self.transforms:
            try:
                for transform in self.transforms:
                    data = transform(data)
            except AttributeError:
                data = transform(data)
        return data, sim_idx


def _get_features_idx(freqs, path_template, handler, n_map_fields, sim_idx):
    features = []
    for freq in freqs:
        feature_path = path_template.format(sim_idx=sim_idx, freq=freq)
        feature_data = handler.read(feature_path)
        # Assume that we run either I or IQU
        features.append(feature_data[:n_map_fields, :])
    features_tensor = [torch.as_tensor(f) for f in features]
    features_tensor = torch.cat(features_tensor, dim=0)
    return features_tensor


def _get_label_idx(path_template, handler, n_map_fields, sim_idx):
    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    if label.shape[0] == 3 and n_map_fields == 1:
        label = label[0, :]
    label_tensor = torch.as_tensor(label)
    return label_tensor