import numpy as np

from utils.planck_instrument import Instrument
from .scale_tasks_helper import TaskTarget


def abs_max_scale(data, abs_max):
    return data / abs_max * 0.5 + 0.5


def abs_max_unscale(data, abs_max):
    return (data - 0.5) * 2 * abs_max


def find_abs_max(task_target: TaskTarget, freqs, map_fields):
    """
    Acts on a single simulation (TaskTarget) to find the maximum absolute value 
        for each detector and field.
    """
    cmb = task_target.cmb_asset
    cmb_data = cmb.handler.read(cmb.path)
    obs = task_target.obs_asset

    res = {'cmb': {}}
    for i, map_field in enumerate(map_fields):
        abs_max = np.max(np.abs(cmb_data[i,:]))
        res['cmb'][map_field] = {'abs_max': abs_max}

    for freq in freqs:
        obs_data = obs.handler.read(str(obs.path).format(freq=freq))
        res[freq] = {}
        # In case a simulation hsa 3 map fields, but the current settings are for just 1
        for i, _ in zip(range(obs_data.shape[0]), map_fields):
            map_field = map_fields[i]
            abs_max = np.max(np.abs(obs_data[i,:]))
            res[freq][map_field] = {'abs_max': abs_max}
    return res


def sift_abs_max_results(results_list, instrument: Instrument, map_fields:str):
    """
    Sifts through aggregated results to find the max over all simulations
    Uses sift_for_detector
    """
    summary = {'cmb': {}}
    for det in instrument.dets.values():
        freq = det.nom_freq
        summary[freq] = {}
        for map_field in det.fields:
            summary[freq][map_field] = {}
            max_v = sift_for_detector(results_list, freq=freq, map_field=map_field)
            summary[freq][map_field]['abs_max'] = max_v
    for map_field in map_fields:
        summary['cmb'][map_field] = {}
        max_v = sift_for_detector(results_list, freq='cmb', map_field=map_field)
        summary['cmb'][map_field]['abs_max'] = max_v
    return summary


def sift_for_detector(results_list, freq, map_field):
    """
    Gets the max for a single detector
    """
    max_vals = [d[freq][map_field]['abs_max'] for d in results_list]
    return max(max_vals)


