from .transform_factor_scale import (
    TrainFactorScaleMap,
    TrainFactorUnScaleMap,
    TestFactorScaleMap,
    TestFactorUnScaleMap
)


# Define a dictionary to map method, dataset, and scale type to the corresponding class
scale_class_map = {
    ('factor', 'train', 'scale'): TrainFactorScaleMap,
    ('factor', 'train', 'unscale'): TrainFactorUnScaleMap,
    ('factor', 'test', 'scale'): TestFactorScaleMap,
    ('factor', 'test', 'unscale'): TestFactorUnScaleMap,
}

def get_scale_class(method, dataset, scale):
    """ Retrieve the scale method class based on method, dataset, and scale. """
    key = (method, dataset, scale)
    if key in scale_class_map:
        return scale_class_map[key]
    else:
        raise ValueError(f"No scale method found for method='{method}', dataset='{dataset}', scale='{scale}'.")
