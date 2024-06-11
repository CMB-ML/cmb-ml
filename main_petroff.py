"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using Petroff's network.

WARNING: It is a work-in-progress; set aside to handle other priorities.

The pipeline consists of the following steps:
1. Training the model
2. Predicting the cleaned CMB signal
3. Converting predictions to common form for comparison across models
4. Generating per-pixel analysis results for each simulation
5. Generating per-pixel summary statistics for each simulation
6. Converting the theory power spectrum to a format that can be used for analysis
# 7. Generating per-ell power spectrum analysis results for each simulation
# 8. Generating per-ell power spectrum summary statistics for each simulation

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_pyilc_predict.py

Note: This script requires the project to be installed, with associated libraries in pyproject.toml.
Note: This script may require the environment variable "CMB_SIMS_LOCAL_SYSTEM" to be set,
        or for appropriate settings in your configuration for local_system.

Author: James Amato
Date: June 11, 2024
"""
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.petroff import (
                     PreprocessMakeExtremaExecutor,
                     PreprocessExecutor,
                     CheckTransformsExecutor,
                     TrainingExecutor,
                     TrainingOnPreprocessedExecutor,
                     PredictionExecutor
                     )
from cmbml.analysis import (
                      PetroffShowSimsPostExecutor,
                      PixelAnalysisExecutor,
                      PixelSummaryExecutor,
                      ConvertTheoryPowerSpectrumExecutor,
                    #   MakePredPowerSpectrumExecutor,
                      )



logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_petroff_t")
def run_petroff(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    # Write hydra's full composed config to a log
    logging.getLogger("hydraconf").info(OmegaConf.to_yaml(HydraConfig.get()))

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    # pipeline_context.add_pipe(HydraConfigPetroffCheckerExecutor)

    pipeline_context.add_pipe(PreprocessMakeExtremaExecutor)
    # pipeline_context.add_pipe(CheckTransformsExecutor)
    # pipeline_context.add_pipe(PreprocessExecutor)
    # pipeline_context.add_pipe(TrainingOnPreprocessedExecutor)
    pipeline_context.add_pipe(TrainingExecutor)
    pipeline_context.add_pipe(PredictionExecutor)
    # pipeline_context.add_pipe(PetroffShowSimsPostExecutor)

    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)
    pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
    # pipeline_context.add_pipe(MakePredPowerSpectrumExecutor)
    # pipeline_context.add_pipe(ShowSinglePsFigExecutor)

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
    validate_environment_variable("CMB_SIMS_LOCAL_SYSTEM")
    run_petroff()
