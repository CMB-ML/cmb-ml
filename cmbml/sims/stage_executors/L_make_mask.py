"""
We need to make a mask for the power spectrum analysis.
This is a simple task, but it is important to ensure consistent results.
"""
import logging

import healpy as hp
import numpy as np
from omegaconf import DictConfig
import pymaster as nmt
import pysm3.units as u

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class MaskCreatorExecutor(BaseStageExecutor):
    """
    MaskCreatorExecutor is responsible for generating the mask file at appropriate resolution.

    Attributes:
        out_mask (Asset): The output asset for the mask.
        in_mask (Asset): The input asset for the mask.
        nside_out (int): The nside for the output mask.
        mask_threshold (float): The threshold for the mask.
    Methods:
        execute() -> None:
            Executes the mask generation process.
        get_mask() -> None:
            Retrieves the mask from the input asset.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='make_mask')
        if cfg.map_fields != 'I':
            raise NotImplementedError("MaskCreatorExecutor only supports Temperature maps.")

        self.out_mask: Asset    = self.assets_out['mask']
        self.out_mask_sm: Asset = self.assets_out['mask_sm']
        out_mask_handler: HealpyMap

        self.in_mask: Asset = self.assets_in['mask']
        in_mask_handler: HealpyMap

        self.nside_out = cfg.scenario.nside
        self.mask_threshold = self.cfg.model.analysis.mask_threshold

        self.mask_apo_size = u.Quantity(self.cfg.model.analysis.mask_sm_apo_size, u.arcmin)
        self.mask_apo_type = self.cfg.model.analysis.mask_sm_apo_type

        if self.mask_apo_type not in ["C1", "C2", "Smooth", 
                                      "hp.smoothing", 
                                      "standardization_is_for_chumps"]:
            raise NotImplementedError(f"Apodization type {self.mask_apo_type} is not supported.")
        if self.mask_apo_type not in ["C1", "C2", "Smooth"]:
            logger.warning(f'Mask apodization "{self.mask_apo_type}" will included masked areas in analysis! This is bad.')

    def execute(self) -> None:
        """
        Runs the mask generation process.
        """
        mask = self.get_masks()
        mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        self.out_mask.write(data=mask)

        if self.mask_apo_type in ["C1", "C2", "Smooth"]:
            mask_sm = use_namaster(mask, self.mask_apo_size, self.mask_apo_type)
        elif self.mask_apo_type == "hp.smoothing":
            mask_sm = use_hp_smoothing(mask, self.mask_apo_size)
        elif self.mask_apo_type == "standardization_is_for_chumps":
            mask_sm = use_alm_conv(mask, self.mask_apo_size)
        else:
            raise NotImplementedError(f"Apodization type {self.mask_apo_type} is not supported.")
        self.out_mask_sm.write(data=mask_sm)

    def get_masks(self):
        """
        Retrieves the mask from the input asset.
        """
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        try:
            mask = mask.value   # HealpyMap returns a Quantity
        except AttributeError:  # Mask is not a Quantity (weird)
            pass
        return mask

def use_namaster(mask, mask_apo_size, mask_apo_type):
    mask_apo_size_degree = mask_apo_size.to(u.degree).value
    apo_mask = nmt.mask_apodization(mask, mask_apo_size_degree, apotype=mask_apo_type)
    return apo_mask

def use_hp_smoothing(mask, mask_apo_size):
    # Not recommended! This includes masked regions of the map in analysis!
    mask_apo_size_rad = mask_apo_size.to(u.rad).value
    apo_mask = hp.smoothing(mask, fwhm=mask_apo_size_rad)
    return apo_mask

def use_alm_conv(mask, mask_apo_size):
    # This method is the same as hp.smoothing, but this is the best I can get from Physics
    # Not recommended! This includes masked regions of the map in analysis!
    mask_apo_size_arcmin = mask_apo_size.to(u.arcmin)
    apo_mask = conv_to_beam(mask, mask_apo_size_arcmin)
    return apo_mask

def conv_to_beam(map, fwhm_new, fwhm_orig = 0,lmax=None):
    # Method taken from Physics' "PyILC Demonstration.ipynb"
    #Convolves from a certain beam to a new beam. If fwhm_orig left blank will assumed inititally deconvolved
    Nside = hp.get_nside(map)
    if lmax is None:
        lmax = 3*Nside-1
    newbeam = hp.gauss_beam(np.radians(fwhm_new/60),lmax=lmax)
    origbeam = hp.gauss_beam(np.radians(fwhm_orig/60),lmax = lmax)
    alm = hp.map2alm(map,lmax = lmax)
    alm_c = hp.almxfl(alm, newbeam/origbeam)
    map_c = hp.alm2map(alm_c,nside=Nside)
    return map_c
