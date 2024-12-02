from typing import Dict, Union, List


from omegaconf import OmegaConf # ListConfig, DictConfig

from astropy import units as u


class ILCConfigMaker:
    def __init__(self, cfg, planck_deltabandpass, use_dets=None) -> None:
        self.cfg = cfg
        self.planck_deltabandpass = planck_deltabandpass
        self.use_dets = use_dets
        self.ilc_cfg_hydra_yaml = self.cfg.model.pyilc
        
        # Placeholders
        self.detector_freqs: List[int] = None
        self.bandwidths: List[float] = None
        self.template = {}
        self.ilc_type = None
        self.set_ilc_type(cfg.model.pyilc.wavelet_type)
        self.set_ordered_detectors()
        self.compose_template()

    def set_ilc_type(self, ILC_type: str) -> None:
        if ILC_type == "CosineNeedlets":
            self.ilc_type = "CNILC"
        elif ILC_type == "TopHatHarmonic":
            self.ilc_type = "HILC"
        elif ILC_type == "GaussianNeedlets":
            self.ilc_type = "GNILC"
        else:
            raise ValueError(f"Unknown ILC type: {ILC_type}")

    def set_ordered_detectors(self) -> None:
        """
        PyILC requires detectors to be ordered by decreasing bandwidth.
        """
        if self.use_dets is None:
            detector_freqs = self.cfg.scenario.detector_freqs
        else:
            detector_freqs = self.use_dets
        band_strs = {det: f"{det}" for det in detector_freqs}
        
        table = self.planck_deltabandpass
        fwhm_s = {det: table.loc[det_str]["fwhm"] for det, det_str in band_strs.items()}
        
        sorted_det_bandwidths = sorted(fwhm_s.items(), key=lambda item: item[1], reverse=True)
        self.detector_freqs = [int(det) for det, bandwidth in sorted_det_bandwidths]
        self.bandwidths = [bandwidth.value for det, bandwidth in sorted_det_bandwidths]

    def compose_template(self):
        """
        We handle a few different kinds of keys in the template:
        (1) Some are used for Hydra, and we ignore them
        (2) Some appear in the model.pyilc yaml exactly
        (3) Some need special handling

        If they don't require special handling, we just include or exclude 
        them in the template.

        For the special handling ones:
        (A) Some are common to all ways PyILC is run
        (B) Some are distinct to the type of PyILC run (e.g., CNILC or HILC)
        (C) Some are optional and independent of the method
            - Include the name of the key with a null value in the model.pyilc yaml
        """
        # The dictionary we'll use
        cfg_dict = {}

        # Get the keys to ignore (1)
        ignore_keys = self.ilc_cfg_hydra_yaml.ignore_keys

        # Get all keys in our model.pyilc yaml
        cfg_hydra = self.ilc_cfg_hydra_yaml
        # Convert OmegaConf to dictionary for later use with yaml library write()
        cfg_hydra = OmegaConf.to_container(cfg_hydra, resolve=True)

        # Put special common keys (3A) in the dictionary (initialize)
        cfg_dict = dict(
            freqs_delta_ghz = self.detector_freqs,
            N_freqs = len(self.detector_freqs),
            N_side = self.cfg.scenario.nside,
        )

        # Add special distinct keys (3B)
        if self.ilc_type == "CNILC":
            cfg_dict["N_scales"] = len(cfg_hydra["ellpeaks"]) + 1
            cfg_dict["ELLMAX"] = cfg_hydra["ellpeaks"][-1] - 1 # Last ellpeak is ellmax + 1

        # Prepare keys that require special handling (3C)
        special_keys = self.special_keys()

        # Add all the (2) and (3C) keys from the model.pyilc yaml to the dictionary
        for k, v in cfg_hydra.items():
            if k not in ignore_keys:
                #             (3C)                                       (2)
                cfg_dict[k] = special_keys[k]() if k in special_keys else v

        # Convert any astropy quantities to values
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

    def make_freq_map_paths(self, input_template):
        paths = []
        for detector in self.detector_freqs:
            det_str = f"{detector}"
            paths.append(str(input_template).format(det=det_str))
        return paths
