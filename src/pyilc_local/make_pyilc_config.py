from typing import Dict, Union, List

from omegaconf import ListConfig, DictConfig

from astropy import units as u


class ILCConfigMaker:
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
        ilc_cfg = self.ilc_cfg_hydra_yaml
        cfg_dict = dict(
            freqs_delta_ghz = self.detector_freqs,
            N_freqs = len(self.detector_freqs),
            N_side = self.cfg.scenario.nside,
        )
        ignore_keys = ["config_maker", "distinct"]
        special_keys = self.special_keys()

        for k in ilc_cfg:
            if k in ignore_keys:
                continue
            elif k in special_keys.keys():
                cfg_dict[k] = special_keys[k]()
            else:
                cfg_dict[k] = ilc_cfg[k]

        distinct_cfg = dict(self.cfg.model.pyilc.distinct)
        for k in distinct_cfg:
            if isinstance(distinct_cfg[k], ListConfig):
                val = list(distinct_cfg[k])
            else:
                val = distinct_cfg[k]
            cfg_dict[k] = val

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

    def make_config(self, output_path, input_paths: List[str]):
        """
        input_paths may be List[str] or List[Path]
        """
        this_template = self.template.copy()
        this_template["freq_map_files"] = input_paths
        this_template["output_dir"] = str(output_path) + r"/"
        return this_template

    def make_freq_map_paths(self, input_template):
        paths = []
        for detector in self.detector_freqs:
            det_str = f"{detector}"
            paths.append(str(input_template).format(det=det_str))
        return paths


# class HarmonicILC(ILCConfigMaker):
#     def compose_template(self):
#         super().compose_template()


# class CosineNeedletILC(ILCConfigMaker):
#     def compose_template(self):
#         super().compose_template()
#         distinct_cfg = dict(self.cfg.model.distinct)
#         for k in distinct_cfg:
#             if isinstance(distinct_cfg[k], ListConfig):
#                 val = list(distinct_cfg[k])
#             else:
#                 val = distinct_cfg[k]
#             self.template[k] = val


# class GaussianNeedletILC(ILCConfigMaker):
#     def compose_template(self):
#         super().compose_template()
#         distinct_cfg = dict(self.cfg.model.distinct)
#         for k in distinct_cfg:
#             if isinstance(distinct_cfg[k], ListConfig):
#                 val = list(distinct_cfg[k])
#             else:
#                 val = distinct_cfg[k]
#             self.template[k] = val
