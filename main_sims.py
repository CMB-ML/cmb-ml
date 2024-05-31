import logging

import hydra

from src.core import (
                      PipelineContext,
                      LogMaker
                      )
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from src.sims import (
    HydraConfigSimsCheckerExecutor,
    NoiseCacheExecutor,
    ConfigExecutor,
    TheoryPSExecutor,
    SimCreatorExecutor
)
from src.analysis import ShowSimsExecutor


logger = logging.getLogger(__name__)


# @hydra.main(version_base=None, config_path="cfg", config_name="config_sim_t")
@hydra.main(version_base=None, config_path="cfg", config_name="config_sim_one_off")
def run_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(HydraConfigSimsCheckerExecutor)
    pipeline_context.add_pipe(NoiseCacheExecutor)
    pipeline_context.add_pipe(ConfigExecutor)
    pipeline_context.add_pipe(TheoryPSExecutor)
    pipeline_context.add_pipe(SimCreatorExecutor)
    # TODO: Put this back in the pipeline yaml; fix/make executor
    # pipeline_context.add_pipe(ShowSimsExecutor)

    pipeline_context.prerun_pipeline()

    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Simulation pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    run_simulations()
