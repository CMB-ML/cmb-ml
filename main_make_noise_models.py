"""
This script runs a simulation pipeline for generating simulations based on a configuration.

The pipeline consists of several steps, including checking configurations, 
creating needed precursor assets (noise cache, configs, and theory power spectra), 
and creating the simulations.

The script uses the Hydra library for configuration management.

To run the simulation pipeline, execute the `run_simulations()` function.

Example usage:
    python main_sims.py

Note: This script requires the project to be installed, with associated libraries in pyproject.toml.
Note: This script may require the environment variable "CMB_SIMS_LOCAL_SYSTEM" to be set,
        or for appropriate settings in your configuration for local_system.

Author: 
Date: June 11, 2024
"""

import logging
import hydra
from cmbml.core import PipelineContext, LogMaker
# from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims_noise import (
    GetPlanckNoiseSimsExecutor,
    MakePlanckAverageNoiseExecutor,
    MakePlanckNoiseModelExecutor,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_noise_models")
def run_simulations(cfg):
    """
    Runs the pipeline for generating anisotropic correlated noise models.

    Args:
        cfg: The configuration object.
    """
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(GetPlanckNoiseSimsExecutor)
    pipeline_context.add_pipe(MakePlanckAverageNoiseExecutor)
    pipeline_context.add_pipe(MakePlanckNoiseModelExecutor)

    pipeline_context.prerun_pipeline()

    had_exception = False
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        had_exception = True
        logger.exception("An exception occurred during the pipeline.", exc_info=e)
        raise e
    finally:
        if had_exception:
            logger.error("Pipeline failed.")
        else:
            logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()

if __name__ == "__main__":
    run_simulations()