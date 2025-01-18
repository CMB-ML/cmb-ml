from typing import Dict, Union, List


from omegaconf import OmegaConf # ListConfig, DictConfig

from astropy import units as u


class ILCConfigMaker:
    def __init__(self, cfg, deltabandpass, use_dets=None) -> None:
        self.cfg = cfg
        self.deltabandpass = deltabandpass
        self.use_dets = use_dets
        self.detector_freqs: List[int] = None
        self.bandwidths: List[float] = None
        self.set_ordered_detectors()
        self.ilc_cfg_hydra_yaml = self.cfg.model.pyilc
        self.template = {}
        self.compose_template()

    def set_ordered_detectors(self) -> None:
        if self.use_dets is None:
            detector_freqs = self.cfg.scenario.detector_freqs
        else:
            detector_freqs = self.use_dets
        band_strs = {det: f"{det}" for det in detector_freqs}
        
        table = self.deltabandpass
        fwhm_s = {det: table.loc[det_str]["fwhm"] for det, det_str in band_strs.items()}
        
        sorted_det_bandwidths = sorted(fwhm_s.items(), key=lambda item: item[1], reverse=True)
        self.detector_freqs = [int(det) for det, bandwidth in sorted_det_bandwidths]
        self.bandwidths = [bandwidth.value for det, bandwidth in sorted_det_bandwidths]

    def compose_template(self):
        # Convert OmegaConf to dictionary for later use with yaml library write()
        ilc_cfg = self.ilc_cfg_hydra_yaml
        ilc_cfg = OmegaConf.to_container(ilc_cfg, resolve=True)
        distinct_cfg = OmegaConf.to_container(self.cfg.model.pyilc.distinct, resolve=True)

        # Get items set in the common configurations
        cfg_dict = dict(
            freqs_delta_ghz = self.detector_freqs,
            N_freqs = len(self.detector_freqs),
            N_side = self.cfg.scenario.nside,
        )

        ignore_keys = ["config_maker", "distinct"]
        special_keys = self.special_keys()

        # Merge the two dictionaries, but only include the keys that are not in ignore_keys
        for k, v in ilc_cfg.items():
            if k not in ignore_keys:
                cfg_dict[k] = special_keys[k]() if k in special_keys else v

        # Add the distinct keys to the dictionary
        cfg_dict.update(distinct_cfg)

        # Convert all astropy quantities to their values
        for k in list(cfg_dict.keys()):
            if isinstance(cfg_dict[k], u.Quantity):
                cfg_dict[k] = cfg_dict[k].value

        self.template = cfg_dict

    def special_keys(self):
        return {
            "beam_files": self.get_beam_files,
            "beam_FWHM_arcmin": self.get_beam_fwhm_vals,
            "freq_bp_files": self.get_freq_bp_files,
        }

    def get_beam_files(self):
        raise NotImplementedError()

    def get_beam_fwhm_vals(self):
        return self.bandwidths

    def get_freq_bp_files(self):
        raise NotImplementedError()

    def make_config(self, output_path, input_paths: List[str], mask_path=None):
        """
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
