import torch

from utils.planck_instrument import Instrument
from .scale_tasks_helper import TaskTarget


def min_max_scale(data, min_v:torch.tensor, max_v:torch.tensor):
    return (data - min_v) / (max_v - min_v)


def min_max_unscale(data, min_v, max_v):
    return data * (max_v - min_v) + min_v


def find_min_max(task_target: TaskTarget, freqs, map_fields):
    """
    Acts on a single simulation (TaskTarget) to find the max and min values
        for each detector and field.
    """
    cmb = task_target.cmb_asset
    cmb_data = cmb.handler.read(cmb.path)
    obs = task_target.obs_asset

    res = {'cmb': {}}
    for i, map_field in enumerate(map_fields):
        res['cmb'][map_field] = {'min_v': cmb_data[i,:].min(), 
                                 'max_v': cmb_data[i,:].max()}

    for freq in freqs:
        obs_data = obs.handler.read(str(obs.path).format(freq=freq))
        res[freq] = {}
        # In case a simulation hsa 3 map fields, but the current settings are for just 1
        for i, _ in zip(range(obs_data.shape[0]), map_fields):
            map_field = map_fields[i]
            res[freq][map_field] = {'min_v': obs_data[i,:].min(), 'max_v': obs_data[i,:].max()}
    return res


def sift_min_max_results(results_list, instrument: Instrument, map_fields:str):
    """
    Sifts through aggregated results to find the min and max over all simulations
    Uses sift_for_detector
    """
    summary = {'cmb': {}}
    for det in instrument.dets.values():
        freq = det.nom_freq
        summary[freq] = {}
        for map_field in det.fields:
            summary[freq][map_field] = {}
            min_v, max_v = sift_for_detector(results_list, freq=freq, map_field=map_field)
            summary[freq][map_field]['min_val'] = min_v
            summary[freq][map_field]['max_val'] = max_v
    for map_field in map_fields:
        summary['cmb'][map_field] = {}
        min_v, max_v = sift_for_detector(results_list, freq='cmb', map_field=map_field)
        summary['cmb'][map_field]['min_val'] = min_v
        summary['cmb'][map_field]['max_val'] = max_v
    return summary


def sift_for_detector(results_list, freq, map_field):
    """
    Gets the min and max for a single detector
    """
    min_vals = [d[freq][map_field]['min_v'] for d in results_list]
    max_vals = [d[freq][map_field]['max_v'] for d in results_list]
    return min(min_vals), max(max_vals)


