from pathlib import Path
from omegaconf import DictConfig
import hydra
import logging


logger = logging.getLogger(__name__)


class FileLocator:
    def __init__(self, create_missing_dest: bool=True) -> None:
        self.create_missing_dest = create_missing_dest

    @staticmethod
    def is_path_accessible(p: Path) -> bool:
        res = False
        try:
            assert p.exists()
            res = True
        except AttributeError as e:
            logger.error(f"References to locations in the file system should be pathlib Paths. Found {p}, type {type(p)}")
            raise e
        except AssertionError:
            logger.error(f"Cannot find {p}")
            raise FileNotFoundError(f"Cannot find {p}")
        return res


class DatasetFileLocator(FileLocator):
    def __init__(self, conf: DictConfig, create_missing_dest: bool=False) -> None:
        logger.info(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(create_missing_dest)
        self.conf = conf
        self.dataset_root = Path(conf.local_system.datasets_root)
        self.dataset_name = conf.dataset_name
        self.detectors = conf.simulation.detector_freqs

        # For use in SplitFolderLocator
        self.split_structures = dict(conf.splits)

        # For use in SimFolderLocator
        self.sim_folder_prefix = conf.file_system.sim_folder_prefix
        self.sim_str_num_digits = conf.file_system.sim_str_num_digits
        self.sim_loc_structure = conf.file_system.structure

        self.cmb_map_fid_fn = conf.file_system.cmb_map_fid_fn
        self.cmb_ps_fid_fn = conf.file_system.cmb_ps_fid_fn
        self.cmb_ps_der_fn = conf.file_system.cmb_ps_der_fn
        self.obs_map_fn = conf.file_system.obs_map_fn

    def get_split(self, split):
        try: 
            assert split in self.split_structures.keys()
        except AssertionError:
            raise ValueError(f"Split {split} is not specified in configuration files.")
        return SplitFolderLocator(parent_dfl=self, split_name=split)


class SplitFolderLocator(FileLocator):
    def __init__(self,
                 parent_dfl: DatasetFileLocator, 
                 split_name: str) -> None:
        logger.info(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(parent_dfl.create_missing_dest)
        self.dfl = parent_dfl
        self.split_name = split_name

        split_structure = self.dfl.split_structures[split_name]
        self.ps_fidu_fixed = split_structure.ps_fidu_fixed
        self.n_sims = split_structure.n_sims

    def get_sim(self, sim_num):
        try:
            f"{sim_num:03d}"
        except Exception as e:
            raise ValueError(f"sim_num {sim_num} should be an integer.")
        try:
            assert sim_num < self.n_sims
        except AssertionError:
            raise ValueError(f"Sim {sim_num} is outside the range (max: {self.n_sims}) specified in configuration files.")
        return SimFileLocator(parent_sfl=self, sim_num=sim_num)


class SimFileLocator(FileLocator):
    def __init__(self,
                 parent_sfl: DatasetFileLocator, 
                 sim_num: int) -> None:
        logger.info(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(create_missing_dest=parent_sfl.create_missing_dest)
        self.sfl = parent_sfl
        self.dfl = parent_sfl.dfl

        sim_folder_prefix = self.dfl.sim_folder_prefix  #
        str_num_digits = self.dfl.sim_str_num_digits    #

        sim_path_build_dict = dict(
            dataset_name = self.dfl.dataset_name,
            split_name = self.sfl.split_name,
            sim_folder = f"{sim_folder_prefix}{sim_num:0{str_num_digits}d}"
        )

        sim_loc_path = self.dfl.sim_loc_structure.format(**sim_path_build_dict)  #
        
        self.sim_path = self.dfl.dataset_root / sim_loc_path
        self.ps_fidu_fixed = self.sfl.ps_fidu_fixed

        self.cmb_map_fid_fn = self.dfl.cmb_map_fid_fn
        self.cmb_ps_fid_fn = self.dfl.cmb_ps_fid_fn
        self.cmb_ps_der_fn = self.dfl.cmb_ps_der_fn
        self.obs_map_fn = self.dfl.obs_map_fn

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


class AssetFileLocator(FileLocator):
    def __init__(self, conf: DictConfig) -> None:
        logger.info(f"Running {__name__} in {__file__}")
        super().__init__(create_missing_dest=False)
        self.dataset_root = Path(conf.local_system.datasets_root)
        self.dest_dir_exists(self.dataset_root)


