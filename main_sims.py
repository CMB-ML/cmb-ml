import logging

import hydra

from src.core import (
                      PipelineContext,
                      LogMaker
                      )
from core.A_check_hydra_configs import HydraConfigCheckerExecutor
from src.sims import (
    SimsHydraConfigCheckerExecutor,
    NoiseCacheExecutor,
    ConfigExecutor,
    TheoryPSExecutor,
    SimCreatorExecutor
)
from src.analysis import (
    ShowSimsExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_32_sim")
# @hydra.main(version_base=None, config_path="cfg", config_name="config_128_sim")
# @hydra.main(version_base=None, config_path="cfg", config_name="config_512_sim")
# @hydra.main(version_base=None, config_path="cfg", config_name="config_2048_sim")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    # pipeline_context.add_pipe(SimsHydraConfigCheckerExecutor)
    # pipeline_context.add_pipe(NoiseCacheExecutor)
    # pipeline_context.add_pipe(ConfigExecutor)
    # pipeline_context.add_pipe(TheoryPSExecutor)
    pipeline_context.add_pipe(SimCreatorExecutor)
    # pipeline_context.add_pipe(ShowSimsExecutor)

    pipeline_context.run_pipeline()

    logger.info("Simulation pipeline completed.")
    log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    make_all_simulations()
