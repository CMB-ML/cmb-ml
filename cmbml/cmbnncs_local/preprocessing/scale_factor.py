import numpy as np

from cmbml.utils.planck_instrument import Instrument
from .scale_tasks_helper import TaskTarget


def factor_scale(map_data, scale):
    n_dets = scale.shape[0]
    scale = scale.view(n_dets, 1, 1)
    return map_data / scale


def factor_unscale(map_data, scale):
    n_dets = scale.shape[0]
    scale = scale.view(n_dets, 1, 1)
    return map_data * scale


# def find_abs_max(task_target: TaskTarget, freqs, map_fields):
#     """
#     Pull value from config yaml. 5, per CMBNNCS paper.
#     """
#     # TODO: load from config here for similar process as Petroff
#     return res
