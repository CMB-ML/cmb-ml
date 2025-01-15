from typing import Dict, Union, List

from omegaconf import OmegaConf # ListConfig, DictConfig

from astropy import units as u


class ConfigMaker:
    def __init__(self, cfg, planck_deltabandpass, use_dets=None) -> None:
        self.cfg = cfg
        self.planck_deltabandpass = planck_deltabandpass
        self.use_dets = use_dets
        self.detector_freqs: List[int] = None
        self.bandwidths: List[float] = None
        self.set_ordered_detectors()
        self.ilc_cfg_hydra_yaml = self.cfg.model.pyilc
        self.template = {}
        self.compose_template()

    def set_ordered_detectors(self) -> None:
        """
        Get the detector frequencies and bandwidths in descending order of bandwidth.
        This is required for PyILC and demonstrates how settings in the common
        configurations can be rewritten for a particular piece of software.
        """
        # Pull the detector frequencies from the config
        detector_freqs = self.cfg.scenario.detector_freqs
        # Convert to strings for lookup in the planck_deltabandpass table
        band_strs = {det: f"{det}" for det in detector_freqs}
        
        # Get the FWHM values from the planck_deltabandpass table
        table = self.planck_deltabandpass
        fwhm_s = {det: table.loc[det_str]["fwhm"] for det, det_str in band_strs.items()}
        
        # Sort the detectors by bandwidth
        sorted_det_bandwidths = sorted(fwhm_s.items(), key=lambda item: item[1], reverse=True)
        self.detector_freqs = [int(det) for det, bandwidth in sorted_det_bandwidths]
        self.bandwidths = [bandwidth.value for det, bandwidth in sorted_det_bandwidths]

    def compose_template(self):
        """
        Put together the constant configuration information.
        """
        # Some configurations are set directly in the cfg.model.pyilc yaml file
        ilc_cfg = self.ilc_cfg_hydra_yaml
        ilc_cfg = OmegaConf.to_container(ilc_cfg, resolve=True)

        # Other configurations come from the common configurations
        cfg_dict = dict(
            freqs_delta_ghz = self.detector_freqs,
            N_freqs = len(self.detector_freqs),
            N_side = self.cfg.scenario.nside,
            beam_FWHM_arcmin = self.bandwidths,
        )

        # Merge the two dictionaries
        for k, v in ilc_cfg.items():
            cfg_dict[k] = v

        # Convert any astropy quantities to their values
        for k in list(cfg_dict.keys()):
            if isinstance(cfg_dict[k], u.Quantity):
                cfg_dict[k] = cfg_dict[k].value

        self.template = cfg_dict

    def make_config(self, output_path, input_paths: List[str], mask_path=None):
        """
        Create the configuration file for the run.

        input_paths may be List[str] or List[Path]
        """
        this_template = self.template.copy()
        this_template["freq_map_files"] = input_paths
        this_template["output_dir"] = str(output_path) + r"/"
        if mask_path is not None:
            # The yaml library doesn't like to print square brackets or spaces; 
            #     we escape [] for now and fix it in the write() method
            #     we also do not include a space after the comma
            #     this works, but deviates from pyilc's instructions.
            this_template["mask_before_covariance_computation"] = f'\[{mask_path},0\]'
        return this_template
