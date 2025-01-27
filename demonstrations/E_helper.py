# Do not do this! This is just for demonstration purposes!

import numpy as np

import healpy as hp
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from cmbml.core import BaseStageExecutor, Asset
from cmbml.core.asset_handlers import TextPowerSpectrum, HealpyMap


class MakePSExecutor(BaseStageExecutor):
    def __init__(self, cfg):
        super().__init__(cfg, stage_str="ps_setup")

        self.out_cmb_ps: Asset = self.assets_out["cmb_ps"]
        out_cmb_ps_handler: TextPowerSpectrum

    def execute(self):
        ell = np.arange(200)
        # Do not do this! This is just for demonstration purposes!
        cheap_model = [ 1.51935454e-13,
                       -1.33044280e-10,
                        4.87473463e-08,
                       -9.68198860e-06,
                        1.12257186e-03, 
                       -7.62816561e-02,
                        3.00276536e+00, 
                       -4.49411282e+01,
                        1.04893659e+03]
        ps = np.poly1d(cheap_model)
        self.out_cmb_ps.write(data=ps(ell))


class InitPS2MapExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="ps2map")

        self.out_cmb_map = self.assets_out["cmb_map"]
        self.in_cmb_ps = self.assets_in["cmb_ps"]

    def execute(self):
        print("This is an example helper")
        return


def make_map_from_ps(ps, nside):
    return hp.synfast(ps, nside)


# Do not do this! This is just for demonstration purposes!
# Use functions and classes and not the global namespace!
with initialize(version_base=None, config_path="../cfg"):
    cfg = compose(config_name='config_demoE_framework')
    prelim = MakePSExecutor(cfg)
    prelim.execute()
    helper = InitPS2MapExecutor(cfg)


###########################################################
# Do not do this! This is just for demonstration purposes!
# helper.name_tracker.context['sim_num'] = 0  # This is a bad idea!
###########################################################


# It causes physical pain to see this!
assets_in = helper.assets_in
assets_out = helper.assets_out
out_map_asset = assets_out["cmb_map"]
in_ps_asset   = assets_in["cmb_ps"]
name_tracker = helper.name_tracker


# But it's been done for the sake of a simpler notebook.
# Please, go learn stuff there, instead.
