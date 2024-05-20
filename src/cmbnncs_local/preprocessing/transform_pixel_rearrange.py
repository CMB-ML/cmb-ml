import numpy as np
import healpy as hp

from cmbnncs.spherical import sphere2piecePlane, piecePlanes2spheres


# class Sphere2RectTransform(object):
#     """
#     Uses CMBNNCS method to rearrange top-level pixels into a rectangle.

#     Native function assumes exactly 1D arrays for map data.

#     Args:
#         from_ring (bool): If the input maps will be in ring ordering.
#     """
#     def __call__(self, map_data: np.ndarray) -> np.ndarray:
#         map_data = map_data.squeeze()
#         map_data_piece_plane = sphere2piecePlane(map_data)
#         return map_data_piece_plane

def sphere2rect(map_data: np.ndarray) -> np.ndarray:
    map_data = map_data.squeeze()
    map_data_piece_plane = sphere2piecePlane(map_data)
    return map_data_piece_plane


# class Rect2SphereTransform(object):
#     """
#     Uses CMBNNCS method to rearrange top-level pixels into a rectangle.

#     Native function assumes exactly 2D arrays for map data.

#     Args:
#         from_ring (bool): If the input maps will be in ring ordering.
#     """
#     def __call__(self, map_data: np.ndarray) -> np.ndarray:
#         map_data = map_data.squeeze()
#         map_data_piece_plane = piecePlanes2spheres(map_data)
#         return map_data_piece_plane

def rect2sphere(map_data: np.ndarray) -> np.ndarray:
    map_data = map_data.squeeze()
    map_data_piece_plane = piecePlanes2spheres(map_data)
    return map_data_piece_plane
