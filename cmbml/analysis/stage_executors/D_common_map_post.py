import logging

import healpy as hp
import numpy as np
from omegaconf import DictConfig
import pysm3.units as u
from tqdm import tqdm

from cmbml.core.asset_handlers import HealpyMap
from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.utils.physics_beam import make_beam, make_target_beams
from cmbml.utils.physics_mask import downgrade_mask


logger = logging.getLogger(__name__)


class CommonPostExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str: str, source_beam_spec) -> None:
        super().__init__(cfg, stage_str)

        self.out_cmb_map: Asset = self.assets_out["cmb_map"]
        out_cmb_map_handler: HealpyMap

        self.in_cmb_map: Asset = self.assets_in["cmb_map"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        in_mask_handler: HealpyMap
        in_cmb_map_handler: HealpyMap

        self.nside_out = cfg.scenario.nside
        self.lmax = cfg.model.analysis.lmax

        self.mask_threshold = cfg.model.analysis.mask_threshold
        self.use_pixel_weights = False
        self.mask_b4_deconv = cfg.model.analysis.post_map_mask_b4_deconv
        self.mask_b4_remove_dipole = cfg.model.analysis.post_map_mask_b4_remove_dipole
        self.do_remove_dipole = cfg.model.analysis.post_map_remove_dipole

        self.source_beam_spec = source_beam_spec
        self.target_beams_cfg = cfg.model.analysis.target_beam_fwhm
        self.source_beam = None
        self.target_beams = None
        self.mask = None

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_mask()
        self.source_beam = make_beam(self.source_beam_spec, lmax=self.lmax)
        self.target_beams = make_target_beams(self.target_beams_cfg, lmax=self.lmax)
        self.default_execute()

    def get_mask(self):
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
        return mask

    def process_split(self, split: Split) -> None:
        logger.info(
            f"Running {self.__class__.__name__} process_split() for split: {split.name}."
        )
        epochs = self.model_epochs if self.model_epochs else [""]
        for epoch in epochs:
            for sim in tqdm(split.iter_sims()):
                with self.name_tracker.set_contexts(dict(epoch=epoch, sim_num=sim)):
                    self.process_sim()

    def process_sim(self) -> None:
        cmb_map = self.in_cmb_map.read()
        if cmb_map.shape[0] == 3 and self.map_fields == "I":
            cmb_map = cmb_map[0]
        post_map = cmb_map.copy()

        # Preserved for compatibility; normally False for full-sky analysis.
        if self.mask_b4_deconv:
            post_map = hp.ma(post_map)
            post_map.mask = np.logical_not(self.mask)

        source_profile = np.asarray(
            self.source_beam.beam[:self.lmax + 1], dtype=float
        )
        if source_profile.size != self.lmax + 1:
            raise ValueError("Source beam does not cover lmax")

        # Lazy cache: transform only if a target differs from the source beam.
        alm_in = None
        units = [getattr(cmb_map, "unit", None) for _ in self.map_fields]
    
        for beam_name, target_beam in self.target_beams:
            target_profile = target_beam.beam

            if np.array_equal(source_profile, target_profile):
                # Preserve pixels exactly when source and target beams match,
                # including NoBeam -> NoBeam and PyILC 20.6' -> 20.6'.
                output_map = post_map.copy()
            else:
                if not np.all(np.isfinite(source_profile)) or np.any(source_profile == 0):
                    raise ValueError("Source beam cannot be safely inverted")

                # Always transform to the *specified* target resolution.
                transfer = target_profile / source_profile
                if not np.all(np.isfinite(transfer)):
                    raise ValueError("Invalid source-to-target beam transfer function")

                if alm_in is None:
                    alm_in = hp.map2alm(post_map, lmax=self.lmax)
                alm_out = hp.almxfl(alm_in, transfer)
                output_map = hp.alm2map(alm_out, nside=self.nside_out)

            if self.do_remove_dipole:
                output_map = self.remove_dipole(output_map)

            with self.name_tracker.set_context("beam_name", beam_name):
                self.out_cmb_map.write(data=output_map, column_units=units)

    def remove_dipole(self, cmb_map: np.ndarray) -> np.ndarray:
        """Fit monopole/dipole (optionally masked); subtract from full sky.

        Preserves units if the input is an Astropy Quantity.
        Assumes HEALPix RING ordering.
        """
        # Extract numerical values, preserving the original units
        is_quantity = isinstance(cmb_map, u.Quantity)
        map_values = cmb_map.value if is_quantity else cmb_map

        if self.mask_b4_remove_dipole:
            fit_map = hp.ma(map_values.copy())
            fit_map.mask = np.logical_not(self.mask)
        else:
            fit_map = map_values

        mono, dipole = hp.fit_dipole(fit_map)

        nside = hp.get_nside(map_values)
        pixels = np.arange(hp.nside2npix(nside))
        x, y, z = hp.pix2vec(nside, pixels)

        fitted_signal = mono + dipole[0] * x + dipole[1] * y + dipole[2] * z

        # Subtract, then restore units if appropriate
        result = map_values - fitted_signal
        return result * cmb_map.unit if is_quantity else result


class CommonRealPostExecutor(CommonPostExecutor):
    """Postprocess full-sky realization maps."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(
            cfg,
            stage_str="common_post_map_real",
            source_beam_spec=cfg.scenario.cmb_beam
        )


class CommonPredPostExecutor(CommonPostExecutor):
    """Postprocess prediction maps."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(
            cfg,
            stage_str="common_post_map_pred",
            source_beam_spec=cfg.model.analysis.beam_type
        )
