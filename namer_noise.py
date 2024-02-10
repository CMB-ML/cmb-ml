from typing import List, Dict
import logging
from pathlib import Path

from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class NoiseGenericFilesNamer:
    def __init__(self, conf: DictConfig, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.detectors: List[int] = conf.simulation.detector_freqs
        self.assume_root_dir_exists()

    def assume_root_dir_exists(self) -> None:
        try:
            assert self.root_dir.exists()
        except:
            raise FileNotFoundError(f"Cache root directory does not exist: {self.root_dir}")

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
        noise_src_path = self.root_dir / noise_fn
        return noise_src_path


class NoiseSrcFilesNamer(NoiseGenericFilesNamer):
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(conf, root_dir=conf.local_system.noise_src_dir)

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


class NoiseCacheFilesNamer(NoiseGenericFilesNamer):
    def __init__(self, conf: DictConfig) -> None:
        logger.debug(f"Running {self.__class__.__name__} in {__name__}")
        super().__init__(conf, root_dir=conf.local_system.noise_cache_dir)

        self.cache_noise_fn_template: str = conf.file_system.noise_cache_fn_template
        self.nside: int = conf.simulation.nside_out

    def _get_noise_fn(self, detector: int, field: str) -> str:
        det = self._det_str(detector)
        noise_fn = self.cache_noise_fn_template.format(det=det, field_char=field, nside=self.nside)
        return noise_fn

    @staticmethod
    def _assume_valid_field(field: str) -> None:
        try:
            assert field in "TQU"
        except:
            raise ValueError(f"field must be one of T, Q, or U")

    def get_path_for(self, detector: int, field: str) -> Path:
        self._assume_detector_in_conf(detector)
        self._assume_valid_field(field)
        noise_fn = self._get_noise_fn(detector, field)
        return self._get_path(noise_fn)
