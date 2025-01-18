"""
This script runs a pipeline for cleaning CMB signal using PyILC.

The pipeline consists of the following steps:
1. Checking configurations
2. Creating needed precursor assets (masks)
3. Generating predictions using PyILC

Because settings in PyILC cause conflicts for Matplotlib, analysis is performed in `main_pyilc_analysis.py`.

The script uses the Hydra library for configuration management.

Usage:
    python main_pyilc_predict.py

Author: 
Date: June 11, 2024
"""
import logging
import hydra

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.sims import MaskCreatorExecutor
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.pyilc_local.B_predict_executor import PredictionExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc")
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
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    validate_environment_variable("CMB_ML_LOCAL_SYSTEM")
    run_pyilc_predictions()
