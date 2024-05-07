import logging

import hydra

from src.core import (
                      PipelineContext,
                      LogMaker
                      )

from src.sims import (
    HydraConfigCheckerExecutor,
    NoiseCacheExecutor,
    ConfigExecutor,
    TheoryPSExecutor,
    SimCreatorExecutor
)


logger = logging.getLogger(__name__)
logging.getLogger('pysm3').setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_scripts_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(NoiseCacheExecutor)
    pipeline_context.add_pipe(ConfigExecutor)
    pipeline_context.add_pipe(TheoryPSExecutor)
    pipeline_context.add_pipe(SimCreatorExecutor)

    pipeline_context.run_pipeline()

    logger.info("Simulation pipeline completed.")
    log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    make_all_simulations()
