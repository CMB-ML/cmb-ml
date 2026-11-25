from typing import NamedTuple
from pathlib import Path
import logging
from multiprocessing import Pool

import healpy as hp
import numpy as np

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import (
    BaseStageExecutor,
    GenericHandler,
    Asset,
)
from cmbml.core.asset_handlers import (
    NumpyMap,
    HealpyMap,
    Config,
)
from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.utils.physics_units import convert_field_str_to_Unit
from cmbml.core.asset_select_features import FeatureAssetCollection


logger = logging.getLogger(__name__)


class FrozenAsset(NamedTuple):
    path: Path
    handler: GenericHandler


class TaskTarget(NamedTuple):
    asset_in: FrozenAsset
    asset_out: FrozenAsset
    dtype: np.dtype
    ordering: str
    units: str
    nside: int


class PreprocessExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, stage_str="data_preprocess")

        self.instrument: Instrument = make_instrument(cfg=cfg)

        self.out_features = FeatureAssetCollection.select(
            self.assets_out,
            choices=["cmb", "obs"],
            instrument=self.instrument,
        )
        out_map_handler: NumpyMap

        self.in_features = FeatureAssetCollection.select(
            self.assets_in,
            choices=["cmb", "obs"],
            instrument=self.instrument,
        )
        in_map_handler: HealpyMap

        self.out_config: Asset = self.assets_out["preprocess_config"]
        out_config_handler: Config

        self.dtype = np.dtype(
            self.get_stage_element("dtype")
        ).newbyteorder("=")

        self.ordering = self.get_stage_element("ordering")
        self.num_processes = self.get_stage_element("num_processes")

        self.units = cfg.scenario.units
        self.nside = cfg.scenario.nside

        if self.map_fields != "I":
            raise NotImplementedError(
                "Preprocessing currently supports temperature-only maps."
            )

        if self.ordering not in ("RING", "NESTED"):
            raise ValueError(f"Unsupported ordering: {self.ordering}")

        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        if self.num_processes < 1:
            raise ValueError("num_processes must be positive.")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.ensure_splits()

        tasks = self.build_tasks()

        if not tasks:
            raise ValueError("No preprocessing tasks were generated.")

        self.check_output_paths(tasks)

        first_task = tasks.pop(0)
        self.try_a_task(parallel_preprocess, first_task)

        self.run_all_tasks(parallel_preprocess, tasks)

        # Write provenance and completion marker last.
        self.out_config.write(data=dict(
            schema_version=1,
            dataset=self.name_tracker.dataset_name,
            nside=self.nside,
            map_fields=self.map_fields,
            units=self.units,
            ordering=self.ordering,
            dtype=self.dtype.name,
            byte_order="native",
            normalization=None,
            detectors=[
                int(freq) for freq in self.instrument.dets.keys()
            ],
            include_cmb=True,
            splits={
                split.name: list(split.iter_sims())
                for split in self.splits
            },
        ))

        logger.info("Preprocessing completed.")

    def build_tasks(self):
        tasks = []

        for split in self.splits:
            for sim in split.iter_sims():
                context = dict(
                    split=split.name,
                    sim_num=sim,
                )

                with self.name_tracker.set_contexts(contexts_dict=context):
                    in_features = self.freeze_features(self.in_features)
                    out_features = self.freeze_features(self.out_features)

                    if in_features.keys() != out_features.keys():
                        raise ValueError(
                            "Input/output feature sets do not match."
                        )

                    for name in in_features:
                        tasks.append(
                            TaskTarget(
                                asset_in=in_features[name],
                                asset_out=out_features[name],
                                dtype=self.dtype,
                                ordering=self.ordering,
                                units=self.units,
                                nside=self.nside,
                            )
                        )

        return tasks

    def freeze_features(self, feature_collection):
        frozen = {}

        for name, asset in feature_collection.feature_iter():
            frozen[name] = self.make_frozen_asset(asset)

        return frozen

    @staticmethod
    def make_frozen_asset(asset: Asset):
        return FrozenAsset(path=asset.path, handler=asset.handler)

    def check_output_paths(self, tasks):
        paths = [self.out_config.path]
        paths.extend(task.asset_out.path for task in tasks)

        for path in paths:
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {path}"
                )

    def try_a_task(self, process, task: TaskTarget):
        process(task)
        logger.info("First map preprocessed without errors.")

    def run_all_tasks(self, process, tasks):
        logger.info(
            f"Running preprocessing on {len(tasks)} tasks "
            f"with {self.num_processes} workers."
        )

        if self.num_processes == 1:
            for task in tqdm(tasks):
                process(task)
            return

        with Pool(processes=self.num_processes) as pool:
            task_iterator = tqdm(
                pool.imap_unordered(process, tasks),
                total=len(tasks),
            )

            for _ in task_iterator:
                pass


def parallel_preprocess(task_target: TaskTarget):
    tt = task_target
    in_asset = tt.asset_in

    in_map = in_asset.handler.read(in_asset.path)

    prepped_map = preprocess_map(
        map_data=in_map,
        dtype=tt.dtype,
        ordering=tt.ordering,
        units=tt.units,
        nside=tt.nside,
    )

    out_asset = tt.asset_out
    out_asset.handler.write(
        path=out_asset.path,
        data=prepped_map,
    )


def preprocess_map(map_data, dtype, ordering, units, nside):
    m = map_data[0] if map_data.ndim > 1 else map_data

    unit = convert_field_str_to_Unit(units)
    m = m.to_value(unit)

    expected_shape = (hp.nside2npix(nside),)

    if m.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {m.shape}."
        )

    if not np.isfinite(m).all() or hp.mask_bad(m).any():
        raise ValueError("Map contains invalid values.")

    m = np.asarray(m, dtype=dtype)

    if ordering == "NESTED":
        m = hp.reorder(m, r2n=True)

    return np.ascontiguousarray(m, dtype=dtype)
