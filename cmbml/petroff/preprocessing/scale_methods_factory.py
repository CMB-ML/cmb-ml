from .pytorch_transform_absmax_scale import (
    TrainAbsMaxScaleMap,
    TrainAbsMaxUnScaleMap,
    TestAbsMaxScaleMap,
    TestAbsMaxUnScaleMap
)
from .pytorch_transform_minmax_scale import (
    TrainMinMaxScaleMap,
    TrainMinMaxUnScaleMap,
    TestMinMaxScaleMap,
    TestMinMaxUnScaleMap
)

from .scale_find_absmax import find_abs_max, sift_abs_max_results
from .scale_find_minmax import find_min_max, sift_min_max_results


# Define a dictionary to map method, dataset, and scale type to the corresponding class
scale_class_map = {
    ('absmax', 'train', 'scale'): TrainAbsMaxScaleMap,
    ('absmax', 'train', 'unscale'): TrainAbsMaxUnScaleMap,
    ('absmax', 'test', 'scale'): TestAbsMaxScaleMap,
    ('absmax', 'test', 'unscale'): TestAbsMaxUnScaleMap,
    ('minmax', 'train', 'scale'): TrainMinMaxScaleMap,
    ('minmax', 'train', 'unscale'): TrainMinMaxUnScaleMap,
    ('minmax', 'test', 'unscale'): TestMinMaxUnScaleMap,
    ('minmax', 'test', 'scale'): TestMinMaxScaleMap
}

def get_scale_class(method, dataset, scale):
    """ Retrieve the scale method class based on method, dataset, and scale. """
    key = (method, dataset, scale)
    if key in scale_class_map:
        return scale_class_map[key]
    else:
        raise ValueError(f"No scale method found for method='{method}', dataset='{dataset}', scale='{scale}'.")


scale_method_map = {
    ('absmax', 'scan'): find_abs_max,
    ('absmax', 'sift'): sift_abs_max_results,
    ('minmax', 'scan'): find_min_max,
    ('minmax', 'sift'): sift_min_max_results
}

def get_sim_scanner(method):
    key = (method, 'scan')
    if key in scale_method_map:
        return scale_method_map[key]
    else:
        raise ValueError(f"No scale method found for method='{method}'")

def get_sim_sifter(method):
    key = (method, 'sift')
    if key in scale_method_map:
        return scale_method_map[key]
    else:
        raise ValueError(f"No scale method found for method='{method}'")
