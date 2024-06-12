import numpy as np
import healpy as hp


class ReorderTransform(object):
    """
    Reorders Healpy Maps between ring and nested orderings.

    The default storage of maps is ring ordering. Hierarchical processes
    (e.g. DeepSphere, Petroff) expect nest ordering. A good visual explanation is
    available at https://healpy.readthedocs.io/en/1.11.0/tutorial.html

    Does not work on PyTorch tensors.

    Args:
        from_ring (bool): If the input maps will be in ring ordering.
    """
    def __init__(self, r2n: bool):
        self.r2n = r2n
        if r2n:
            self.hp_call = dict(r2n = True)
        else:
            self.hp_call = dict(n2r = True)

    def __call__(self, map_data: np.ndarray) -> np.ndarray:
        out_map = hp.reorder(map_data, **self.hp_call)
        return out_map
