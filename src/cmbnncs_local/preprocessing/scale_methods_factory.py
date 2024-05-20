from .transform_factor_scale import (
    TrainAbsMaxScaleMap,
    TrainAbsMaxUnScaleMap,
    TestAbsMaxScaleMap,
    TestAbsMaxUnScaleMap
)


# Define a dictionary to map method, dataset, and scale type to the corresponding class
scale_class_map = {
    ('factor', 'train', 'scale'): TrainAbsMaxScaleMap,
    ('factor', 'train', 'unscale'): TrainAbsMaxUnScaleMap,
    ('factor', 'test', 'scale'): TestAbsMaxScaleMap,
    ('factor', 'test', 'unscale'): TestAbsMaxUnScaleMap,
}

def get_scale_class(method, dataset, scale):
    """ Retrieve the scale method class based on method, dataset, and scale. """
    key = (method, dataset, scale)
    if key in scale_class_map:
        return scale_class_map[key]
    else:
        raise ValueError(f"No scale method found for method='{method}', dataset='{dataset}', scale='{scale}'.")
