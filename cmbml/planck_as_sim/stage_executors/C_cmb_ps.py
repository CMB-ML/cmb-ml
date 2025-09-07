import logging

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Asset
from cmbml.utils.planck_ps_text_add_l01 import add_missing_multipoles


logger = logging.getLogger(__name__)


class CMBPSConvertExecutor(BaseStageExecutor):
    """
    Simply copies the Planck power spectrum such that the rest of the
    CMB-ML pipeline handles it natively.
    """
    def __init__(self, cfg: DictConfig) -> None:
        # The following stage_str must match the pipeline yaml
        super().__init__(cfg, stage_str='convert_ps')

        self.out_cmb_ps: Asset = self.assets_out['cmb_ps']
        self.in_cmb_ps: Asset = self.assets_in['cmb_ps']


    def execute(self) -> None:
        """
        Executes the noise cache generation process.
        """
        in_cmb_ps = self.in_cmb_ps.read()

        out_cmb_ps = add_missing_multipoles(in_cmb_ps)

        context = dict(
            split="Test",
            sim_num=0
        )
        with self.name_tracker.set_contexts(context):
            self.out_cmb_ps.write(data=out_cmb_ps)

