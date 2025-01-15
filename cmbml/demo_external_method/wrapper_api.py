# from some_library import SomeClass


class SomeClass:
    """
    Part of an external library that we want to use
    """
    def __init__(self,
                 param_A,
                 param_B,
                 param_C,
                 mask) -> None:
        self.param_A = param_A
        self.param_B = param_B
        self.param_C = param_C
        self.mask = mask
    
    def run(self, input_maps):
        # This will process the input maps and return a result
        return None


def run_wrapped_api(input_maps, param_A, param_B, param_C, mask):
    something = SomeClass(param_A=param_A, param_B=param_B, param_C=param_C, mask=mask)
    result = something.run(input_files=input_maps)
    return result
