from pathlib import Path
from omegaconf import DictConfig
from typing import List, Dict, Any
import hydra
import logging


logger = logging.getLogger(__name__)


class DatasetFiles():
    def __init__(self, conf: DictConfig, create_missing_dest: bool=False) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.dataset_root = Path(conf.local_system.datasets_root)
        self.dataset_name: str = conf.dataset_name
        self.detectors: List[int] = conf.simulation.detector_freqs

        # For use in SplitFolderLocator
        self.split_structures: Dict[str, Any] = dict(conf.splits)

        # For use in SimFolderLocator
        self.sim_folder_prefix: str = conf.file_system.sim_folder_prefix
        self.sim_str_num_digits: int = conf.file_system.sim_str_num_digits
        self.sim_loc_structure: str = conf.file_system.structure

        self.sim_config_fn: str = conf.file_system.sim_config_fn
        self.cmb_map_fid_fn: str = conf.file_system.cmb_map_fid_fn
        self.cmb_ps_fid_fn: str = conf.file_system.cmb_ps_fid_fn
        self.cmb_ps_der_fn: str = conf.file_system.cmb_ps_der_fn
        self.obs_map_fn: str = conf.file_system.obs_map_fn

    def get_split(self, split):
        try: 
            assert split in self.split_structures.keys()
        except AssertionError:
            raise ValueError(f"Split {split} is not specified in configuration files.")
        return SplitFiles(parent_dfl=self, split_name=split)


class SplitFiles():
    def __init__(self,
                 parent_dfl: DatasetFiles, 
                 split_name: str) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.dfl = parent_dfl
        self.split_name = split_name

        split_structure = self.dfl.split_structures[split_name]
        self.ps_fidu_fixed: bool = split_structure.ps_fidu_fixed
        self.n_sims: int = split_structure.n_sims

    def get_sim(self, sim_num):
        try:
            f"{sim_num:03d}"
        except Exception as e:
            raise ValueError(f"sim_num {sim_num} should be an integer.")
        try:
            assert sim_num < self.n_sims
        except AssertionError:
            raise ValueError(f"Sim {sim_num} is outside the range (max: {self.n_sims}) specified in configuration files.")
        return SimFiles(parent_sfl=self, sim_num=sim_num)


class SimFiles():
    def __init__(self,
                 parent_sfl: SplitFiles, 
                 sim_num: int) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        self.sfl = parent_sfl
        self.dfl = parent_sfl.dfl

        sim_folder_prefix = self.dfl.sim_folder_prefix
        str_num_digits = self.dfl.sim_str_num_digits

        sim_path_build_dict = dict(
            dataset_name = self.dfl.dataset_name,
            split_name = self.sfl.split_name,
            sim_folder = f"{sim_folder_prefix}{sim_num:0{str_num_digits}d}"
        )

        sim_loc_path = self.dfl.sim_loc_structure.format(**sim_path_build_dict)
        
        self.sim_path = self.dfl.dataset_root / sim_loc_path
        self.ps_fidu_fixed = self.sfl.ps_fidu_fixed

        self.sim_config_fn = self.dfl.sim_config_fn

        self.cmb_map_fid_fn = self.dfl.cmb_map_fid_fn
        self.cmb_ps_fid_fn = self.dfl.cmb_ps_fid_fn
        self.cmb_ps_der_fn = self.dfl.cmb_ps_der_fn
        self.obs_map_fn = self.dfl.obs_map_fn

    @property
    def sim_config_path(self):
        return self.sim_path / self.sim_config_fn

    @property
    def cmb_map_fid_path(self):
        return self.sim_path / self.cmb_map_fid_fn

    @property
    def cmb_ps_fid_path(self):
        if self.ps_fidu_fixed:
            return self.sim_path.parent / self.cmb_ps_fid_fn
        else:
            return self.sim_path / self.cmb_ps_fid_fn

    @property
    def cmb_ps_der_path(self):
        return self.sim_path / self.cmb_ps_der_fn

    def obs_map_path(self, detector:int):
        try:
            assert detector in self.sfl.dfl.detectors
        except AssertionError:
            raise ValueError(f"Detector {detector} not found in configuration.")
        return self.sim_path / self.obs_map_fn.format(det=detector)


class NoiseGenericFiles:
    def __init__(self, conf: DictConfig) -> None:
        self.noise_dir: Path = None
        self.detectors: List[int] = conf.simulation.detector_freqs

    @staticmethod
    def _det_str(detector):
        return f"{detector:03d}"

    def _get_noise_fn(self, detector: int) -> str:
        raise NotImplementedError("This is an abstract class")

    def _assume_detector_in_conf(self, detector) -> None:
        try:
            assert detector in self.detectors
        except AssertionError:
            raise ValueError(f"Detectors {detector} not in config file ({self.detectors}).")

    def _get_path(self, noise_fn: str):
        noise_src_path = self.noise_dir / noise_fn
        return noise_src_path


class NoiseSrcFiles(NoiseGenericFiles):
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(conf)
        self.noise_dir = Path(conf.local_system.noise_files)

        self.noise_files: Dict[str, str] = dict(conf.simulation.noise)

    def _get_noise_det_conf_key(self, detector: int):
        return f"det{self._det_str(detector)}"

    def _get_noise_fn(self, detector: int) -> str:
        det_conf_key = self._get_noise_det_conf_key(detector)
        try:
            noise_fn = self.noise_files[det_conf_key]
        except KeyError:
            raise ValueError(f"Configuration has no {det_conf_key} in simulation/noise.yaml")
        return noise_fn
    
    def get_path_for(self, detector):
        self._assume_detector_in_conf(detector)
        noise_fn = self._get_noise_fn(detector)
        noise_src_path = self._get_path(noise_fn)
        return noise_src_path


class NoiseCacheFiles(NoiseGenericFiles):
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(conf)
        self.noise_dir = Path(conf.local_system.noise_cache_dir)

        self.cache_noise_fn_template: str = conf.file_system.noise_cache_fn_template
        self.create_noise_cache: bool = conf.create_noise_cache
        self.nside: int = conf.simulation.nside

    def _get_noise_fn(self, detector: int, field: str) -> str:
        det = self._det_str(detector)
        nside = self.nside
        noise_fn = self.cache_noise_fn_template.format(det=det, field_char=field, nside=nside)
        return noise_fn

    @staticmethod
    def _assume_valid_field(field: str) -> None:
        try:
            assert field in "TQU"
        except:
            raise ValueError(f"field must be one of T, Q, or U")

    def get_path_for(self, detector: int, field: str):
        self._assume_detector_in_conf(detector)
        self._assume_valid_field(field)
        noise_fn = self._get_noise_fn(detector, field)
        return self._get_path(noise_fn)


class WMAPFiles:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.wmap_chains_dir = Path(conf.local_system.wmap_chains)


class InstrumentFiles:
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.instr_table_path = Path(conf.local_system.instr_table)