import logging

from omegaconf import DictConfig
from tqdm import tqdm

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import Config, HealpyMap

from get_data.utils.get_from_shared_link import download_shared_link_info


logger = logging.getLogger(__name__)


class GetFromBoxBaseExecutor(BaseStageExecutor):
    """
    GetAssetsExecutor downloads assets needed for running CMB-ML.
    """
    def __init__(self, cfg: DictConfig, stage_str:str) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='download_sims')

        self.temp_tar_dir: Asset = self.assets_out['temp_tar_dir']
        self.dataset_dir:  Asset = self.assets_out['dataset_dir']
        # For reference:
        out_maps: HealpyMap

        self.in_shared_links: Asset = self.assets_in['shared_links']
        in_links: Config

        if not self.in_shared_links.path.exists():
            raise FileNotFoundError(f"Shared links file not found at {self.in_shared_links.path}. " \
                                    "Please be sure that it's been copied from '<this_repo>/assets/CMB-ML'")

        self.shared_links = None  # This will be loaded in the execute method

    def execute(self) -> None:
        """
        Executes the downloading process.
        """
        raise NotImplementedError("This method must be implemented in a subclass.")

    def process_split(self, split):
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        
        # If, later, we want to select particular sims, replace the following line
        sim_iter = split.iter_sims()

        with tqdm(total=len(sim_iter), desc=f"Processing {split.name} split") as pbar:
            for sim_num in sim_iter:
                with self.name_tracker.set_context("sim_num", sim_num):
                    self.process_sim(split, sim_num)
                    pbar.update(1)

    def process_sim(self, split, sim_num):
        key = self.get_key(split=split.name, sim_num=sim_num)
        self.download_by_key(key)

    def download_by_name(self, name):
        key = self.get_key(name=name)
        self.download_by_key(key)

    def get_key(self, split=None, sim_num=None, name=None):
        if name is not None:
            key = name
        elif split is None or sim_num is None:
            raise ValueError("split and sim_num must be provided if name is None.")
        else:
            context = dict(split=split, sim_num=sim_num)
            with self.name_tracker.set_contexts(context):
                key = f"{split}_{self.name_tracker.sim_name()}"
        return key

    def download_by_key(self, key):
        shared_link = self.shared_links[key]
        download_shared_link_info(shared_link, 
                                  self.temp_tar_dir.path, 
                                  self.dataset_dir.path)


class GetDatasetExecutor(GetFromBoxBaseExecutor):
    """
    GetAssetsExecutor downloads assets needed for running CMB-ML.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='download_sims')

    def execute(self) -> None:
        """
        Downloads the dataset.
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        self.shared_links = self.in_shared_links.read()
        
        # Download all sims with default_execute
        self.default_execute()

        # Download Logs and Noise Model
        self.download_by_name("Logs")
        self.download_by_name("NoiseCache")


class GetNoiseModelExecutor(GetFromBoxBaseExecutor):
    """
    GetAssetsExecutor downloads assets needed for the noise model.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='download_sims')

    def execute(self) -> None:
        """
        Downloads summary files for the noise model. Those can be regenerated instead, using
        cmbml/sims/D_make_average_map -> MakePlanckAverageNoiseExecutor
        cmbml/sims/E_make_noise_models -> MakePlanckNoiseModelExecutor
        """
        logger.debug(f"Running {self.__class__.__name__} execute() method.")
        self.shared_links = self.in_shared_links.read()
        
        self.download_by_name("NoiseModel")
