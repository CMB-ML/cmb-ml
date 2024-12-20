import os
import logging

from torch.utils.data import Dataset
import torch

from cmbml.utils.patch_healpix import make_pixel_index_lut


logger = logging.getLogger(__name__)


class TrainCMBPatchDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields: str,
                 label_path_template,
                 label_handler,
                 feature_path_template,
                 feature_handler,
                 which_patch_dict: dict,
                 nside_obs,
                 nside_patches,
                 lut
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
        logger.debug("Building patch LUT...")
        self.patch_lut = lut
        logger.debug(f"Patch LUT built. Shape: {self.patch_lut.shape}")

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


class TestCMBMapDataset(Dataset):
    def __init__(self, 
                 n_sims,
                 freqs,
                 map_fields,
                #  label_path_template,
                #  label_handler,
                 feature_path_template,
                 feature_handler,
                 ):
        self.n_sims = n_sims
        self.freqs = freqs
        # self.label_path_template = label_path_template
        # self.label_handler = label_handler
        self.feature_path_template = feature_path_template
        self.feature_handler = feature_handler
        self.n_map_fields:int = len(map_fields)
        raise NotImplementedError("This class is not implemented yet.")

    # def __len__(self):
    #     return self.n_sims
    
    # def __getitem__(self, sim_idx):
    #     # label_path = self.label_path_template.format(sim_idx=sim_idx)
    #     # label = self.label_handler.read(label_path)
    #     # label_tensor = torch.as_tensor(label)

    #     features = _get_features_idx(freqs=self.freqs,
    #                                  path_template=self.feature_path_template,
    #                                  handler=self.feature_handler,
    #                                  n_map_fields=self.n_map_fields,
    #                                  sim_idx=sim_idx)


    #     # feature_path_template = self.feature_path_template.format(sim_idx=sim_idx, det="{det}")
    #     # features = self.feature_handler.read(feature_path_template)
    #     features_tensor = tuple([torch.as_tensor(f) for f in features])
    #     features_tensor = torch.cat(features_tensor, dim=0)
    #     return features_tensor, sim_idx
    

def _get_features_idx(freqs, path_template, handler, n_map_fields, sim_idx, r_ids):
    # TODO: Implement multiple fields
    if n_map_fields > 1:
        raise ValueError("This function only supports one map field at a time.")

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
        raise ValueError("This function only supports one map field at a time.")

    label_path = path_template.format(sim_idx=sim_idx)
    label = handler.read(label_path)
    label = label[0]  # TODO: Implement multiple fields
    label = label[r_ids]
    label = label.value

    

    return label
