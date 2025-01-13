import os
import logging

import numpy as np
from torch.utils.data import Dataset
import torch

from cmbml.utils.patch_healpix import make_pixel_index_lut


logger = logging.getLogger(__name__)


class TrainCMBMap2PatchDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 which_patch_dict: dict,
                 lut, 
                 transform=None
                 ):
        # TODO: Adopt similar method as in parallel operations to allow 
        #       this to use num_workers and transforms
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.which_patch_dict = which_patch_dict

        self.transform = transform

        self.patch_lut = lut
        logger.debug(f"Patch LUT shape: {self.patch_lut.shape}")

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        patch_id = self.which_patch_dict[sim_idx]
        r_ids = self.patch_lut[patch_id]
        features = _get_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.feature_handler,
                                     n_map_fields=self.n_map_fields,
                                     sim_idx=sim_idx, 
                                     r_ids=r_ids)

        label = _get_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_map_fields=self.n_map_fields,
                               sim_idx=sim_idx,
                               r_ids=r_ids)
        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.stack(features_tensor, dim=0)

        label_tensor = torch.as_tensor(label)
        label_tensor = label_tensor.unsqueeze(0)
        return features_tensor, label_tensor, sim_idx, patch_id  # For debugging
        # return features_tensor, label  # For regular use


class TrainCMBPrePatchDataset(Dataset):
    """
    For use with pre-processed patches, already scaled.
    """
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        self.label_path_template = label_path_template
        self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)

    def __len__(self):
        return self.n_sims

    def __getitem__(self, sim_idx):
        features = _get_patch_features_idx(freqs=self.freqs,
                                     path_template=self.feature_path_template,
                                     handler=self.feature_handler,
                                     n_map_fields=self.n_map_fields,
                                     sim_idx=sim_idx)

        label = _get_patch_label_idx(path_template=self.label_path_template,
                               handler=self.label_handler,
                               n_map_fields=self.n_map_fields,
                               sim_idx=sim_idx)

        if self.transform:
            features = self.transform(np.array(features))
            label = self.transform(np.array(label))

        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.stack(features_tensor, dim=0)

        label_tensor = torch.as_tensor(label)
        # Double unsqueeze to match the shape of the other label tensors
        label_tensor = label_tensor.unsqueeze(0)
        # return features_tensor, label_tensor, sim_idx  # For debugging
        return features_tensor, label_tensor  # For regular use


class TestCMBPatchDataset(Dataset):
    """
    Creates a dataset, iterating over patches for a single simulation.
    """
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 feature_path_template,
                 feature_handler,
                 lut,
                 transform=None
                 ):
        # TODO: Adopt similar method as in parallel operations to allow 
        #       this to use num_workers and transforms
        self.n_sims = n_sims
        self.freqs = freqs
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        self.patch_lut = lut
        logger.debug(f"Patch LUT built. Shape: {self.patch_lut.shape}")

        self._sim_idx = None
        self._current_map_data = None

        self.transform = transform

    def __len__(self):
        # We use the number of patches defined as the length of the dataset
        return self.patch_lut.shape[0]

    @property
    def sim_idx(self):
        return self._sim_idx

    @sim_idx.setter
    def sim_idx(self, value):
        """Setter for sim_idx. Reads the map file when the value is set."""
        if value != self._sim_idx:  # Avoid re-reading if the same value is set
            self._sim_idx = value
            features = []
            for freq in self.freqs:
                feature_path = self.feature_path_template.format(sim_idx=self.sim_idx, freq=freq)
                feature_data = self.feature_handler.read(feature_path)
                # Strip out the other map fields and the astropy units information:
                feature_data = feature_data[0].value  # TODO: Implement multiple fields
                features.append(feature_data)
            self._current_map_data = features
            logger.debug(f"Loaded map data for simulation index {value}")

    @property
    def current_map_data(self):
        return self._current_map_data

    def __getitem__(self, patch_id):
        r_ids = self.patch_lut[patch_id]
        features = [m[r_ids] for m in self.current_map_data]

        if self.transform:
            features = self.transform(np.array(features))

        features_tensor = tuple([torch.as_tensor(f) for f in features])
        features_tensor = torch.stack(features_tensor, dim=0)

        return features_tensor, self.sim_idx, patch_id  # For debugging
        # return features_tensor, label  # For regular use


def _get_features_idx(freqs, path_template, handler, n_map_fields, sim_idx, r_ids):
    # TODO: Implement multiple fields
    if n_map_fields > 1:
        raise NotImplementedError("This function only supports one map field at a time.")

    features = []
    for freq in freqs:
        feature_path = path_template.format(sim_idx=sim_idx, freq=freq)
        feature_data = handler.read(feature_path)
        feature_data = feature_data[0]  # TODO: Implement multiple fields
        feature_data = feature_data[r_ids]
        feature_data = feature_data.value
        features.append(feature_data)
    return features


def _get_label_idx(path_template, handler, n_map_fields, sim_idx, r_ids):
    # TODO: Implement multiple fields
    if n_map_fields > 1:
        raise NotImplementedError("This function only supports one map field at a time.")

    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    label = label[0]  # TODO: Implement multiple fields
    label = label[r_ids]
    label = label.value
    return label


def _get_patch_features_idx(freqs, path_template, handler, n_map_fields, sim_idx):
    # TODO: Implement multiple fields
    if n_map_fields > 1:
        raise NotImplementedError("This function only supports one map field at a time.")

    features = []
    for freq in freqs:
        feature_path = path_template.format(sim_idx=sim_idx, freq=freq)
        feature_data = handler.read(feature_path)
        feature_data = feature_data[0]  # TODO: Implement multiple fields
        features.append(feature_data)
    return features


def _get_patch_label_idx(path_template, handler, n_map_fields, sim_idx):
    # TODO: Implement multiple fields
    if n_map_fields > 1:
        raise NotImplementedError("This function only supports one map field at a time.")

    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    label = label[0]  # TODO: Implement multiple fields
    return label
