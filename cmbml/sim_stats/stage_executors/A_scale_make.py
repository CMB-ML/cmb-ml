import logging

import healpy as hp
import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.core.asset_handlers import NumpyMap, HealpyMap, Config
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils.physics_units import convert_field_str_to_Unit
from cmbml.core.asset_select_features import (
    FeatureAssetCollection,
    FeatureAssetNames,
)
from cmbml.sim_stats.utils.stat_convention import (
    STAT_MIN,
    STAT_MAX,
    STAT_MEAN,
    STAT_STDV,
    N_STATS,
)


logger = logging.getLogger(__name__)


STAT_NAMES = FeatureAssetNames(
    cmb="cmb_stat",
    obs="obs_stat",
)


class FindPxMapStatisticsExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="get_data_stats")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_map_stat = FeatureAssetCollection.select(
            self.assets_out,
            choices=["cmb", "obs"],
            instrument=self.instrument,
            name_map=STAT_NAMES,
        )

        self.out_stats: Asset = self.assets_out["scalar_stats"]
        self.out_config: Asset = self.assets_out["stats_config"]
        out_map_stat_handler: NumpyMap
        out_stats_handler: Config

        self.in_features = FeatureAssetCollection.select(
            self.assets_in,
            choices=["cmb", "obs"],
            instrument=self.instrument,
        )
        in_map_handler: HealpyMap

        self.nside = cfg.scenario.nside
        self.units = cfg.scenario.units
        self.unit = convert_field_str_to_Unit(self.units)

        if self.map_fields != "I":
            raise NotImplementedError(
                "Map statistics currently support temperature-only maps."
            )

    def execute(self) -> None:
        self.ensure_splits()

        split: Split = self.splits["train"]
        n_sims = split.n_sims

        if n_sims < 2:
            raise ValueError(
                "Need at least two simulations to compute statistics."
            )

        self.check_output_paths()

        npix = hp.nside2npix(self.nside)

        feature_names = [
            channel for channel, _ in self.in_features.feature_iter()
        ]

        logger.info(f"Computing statistics for {n_sims} simulations.")
        logger.info(f"Features: {feature_names}")
        logger.info(f"Nside: {self.nside}")
        logger.info(f"Units: {self.units}")

        min_maps = {
            name: np.full(npix, np.inf, dtype=np.float64)
            for name in feature_names
        }
        max_maps = {
            name: np.full(npix, -np.inf, dtype=np.float64)
            for name in feature_names
        }
        mean_maps = {
            name: np.zeros(npix, dtype=np.float64)
            for name in feature_names
        }
        m2_maps = {
            name: np.zeros(npix, dtype=np.float64)
            for name in feature_names
        }

        count = 0
        sim_indices = []

        for sim in tqdm(split.iter_sims(), total=n_sims):
            context = dict(sim_num=sim, split=split.name)

            with self.name_tracker.set_contexts(context):
                count += 1
                sim_indices.append(int(sim))

                for name, asset in self.in_features.feature_iter():
                    x = asset.read()

                    x = x.to_value(self.unit)
                    x = np.asarray(x, dtype=np.float64)

                    if x.shape == (1, npix):
                        x = x[0]

                    if x.shape != (npix,):
                        raise ValueError(
                            f"Unexpected shape for {name}: {x.shape}"
                        )

                    if not np.isfinite(x).all() or hp.mask_bad(x).any():
                        raise ValueError(
                            f"Invalid map values in {asset.path}"
                        )

                    np.minimum(min_maps[name], x, out=min_maps[name])
                    np.maximum(max_maps[name], x, out=max_maps[name])

                    delta = x - mean_maps[name]
                    mean_maps[name] += delta / count
                    delta2 = x - mean_maps[name]
                    m2_maps[name] += delta * delta2

        if count < 2:
            raise ValueError("Fewer than two simulations processed.")

        # Global statistics
        global_dict = {}

        for name in feature_names:
            global_min = min_maps[name].min()
            global_max = max_maps[name].max()
            global_mean = mean_maps[name].mean()

            global_m2 = (
                m2_maps[name].sum()
                + count * np.sum(
                    (mean_maps[name] - global_mean) ** 2
                )
            )

            global_stdv = np.sqrt(global_m2 / (count * npix))

            global_dict[name] = dict(
                vmin=float(global_min),
                vmax=float(global_max),
                mean=float(global_mean),
                stdv=float(global_stdv),
            )

        # Local statistics
        for name, asset in self.out_map_stat.feature_iter():
            out_maps = np.empty((N_STATS, npix), dtype=np.float64)

            out_maps[STAT_MIN] = min_maps[name]
            out_maps[STAT_MAX] = max_maps[name]
            out_maps[STAT_MEAN] = mean_maps[name]
            out_maps[STAT_STDV] = np.sqrt(m2_maps[name] / count)

            logger.info(f"Writing local statistics for {name}.")
            asset.write(data=out_maps)

        # Write global statistics
        logger.info("Writing global statistics.")
        self.out_stats.write(data=global_dict)

        # Write provenance and completion marker last
        stats_config = dict(
            schema_version=1,
            dataset=self.name_tracker.dataset_name,
            nside=self.nside,
            map_fields=self.map_fields,
            ordering="RING",
            units=self.units,
            source=dict(
                split=split.name,
                n_sims=count,
                sim_indices=sim_indices,
                detectors=[
                    int(freq) for freq in self.instrument.dets.keys()
                ],
                include_cmb=True,
            ),
            statistics=dict(
                global_stats=True,
                per_pixel_stats=True,
                row_order=["min", "max", "mean", "stdv"],
                ddof=0,
                accumulation_dtype="float64",
                output_dtype="float64",
            ),
            preprocessing=dict(
                mask=None,
                smoothing=None,
                remove_monopole=False,
                remove_dipole=False,
            ),
        )

        self.out_config.write(data=stats_config)

        logger.info("Map statistics generation completed.")

    def check_output_paths(self) -> None:
        paths = [
            self.out_stats.path,
            self.out_config.path,
        ]

        for _, asset in self.out_map_stat.feature_iter():
            paths.append(asset.path)

        existing = [path for path in paths if path.exists()]

        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing statistics:\n"
                + "\n".join(str(path) for path in existing)
            )
