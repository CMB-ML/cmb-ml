import logging

import hydra

from src.core import (
                      PipelineContext,
                      LogMaker
                      )
from src.cmbnncs_local import (
                         PreprocessMakeScaleExecutor,
                         PreprocessExecutor,
                         ParallelPreprocessExecutor,
                        #  TrainingExecutor,
                        #  PredictionExecutor,
                        #  PostprocessExecutor,
                         )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(PreprocessMakeScaleExecutor)
    # pipeline_context.add_pipe(PreprocessExecutor)
    pipeline_context.add_pipe(ParallelPreprocessExecutor)
    # pipeline_context.add_pipe(TrainingExecutor)

    pipeline_context.run_pipeline()

    logger.info("Simulation pipeline completed.")
    log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    make_all_simulations()
