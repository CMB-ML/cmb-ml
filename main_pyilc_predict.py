import logging

import hydra

from utils.check_env_var import validate_environment_variable
from core import (
                      PipelineContext,
                      LogMaker
                      )
from src.sims import MaskCreatorExecutor
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from pyilc_local.B_predict_executor import PredictionExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc_t")
def run_pyilc_predictions(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(MaskCreatorExecutor)
    pipeline_context.add_pipe(PredictionExecutor)

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
    validate_environment_variable("CMB_SIMS_LOCAL_SYSTEM")
    run_pyilc_predictions()
