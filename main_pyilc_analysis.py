"""
This script runs a pipeline for analysis of the cleaned CMB signal generated with PyILC.

The pipeline consists of the following steps:
1. Generating per-pixel analysis results for each simulation
2. Generating per-pixel summary statistics for each simulation
3. Converting the theory power spectrum to a format that can be used for analysis
4. Generating per-ell power spectrum analysis results for each simulation
5. Generating per-ell power spectrum summary statistics for each simulation

And also generating various analysis figures, throughout.

Because settings in PyILC cause conflicts for Matplotlib, this analysis is performed separately from `main_pyilc_predict.py`.

The script uses the Hydra library for configuration management.

Usage:
    python main_pyilc_predict.py

Note: This script requires the project to be installed, with associated libraries in pyproject.toml.
Note: This script may require the environment variable "CMB_SIMS_LOCAL_SYSTEM" to be set,
        or for appropriate settings in your configuration for local_system.

Author: James Amato
Date: June 11, 2024
"""
from functools import partial
import logging

import hydra

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor

from cmbml.analysis import   (
                            NILCShowSimsPostExecutor,
                            CommonRealPostExecutor,
                            CommonPyILCPredPostExecutor,
                            CommonNILCShowSimsPostExecutor,
                            PixelAnalysisExecutor,
                            PixelSummaryExecutor,
                            ConvertTheoryPowerSpectrumExecutor,
                            MakeTheoryPSStats,
                            PyILCMakePSExecutor,
                            PixelSummaryFigsExecutor,
                            PSAnalysisExecutor,
                            PowerSpectrumAnalysisExecutorSerial,
                            PowerSpectrumSummaryExecutor,
                            PowerSpectrumSummaryFigsExecutor,
                            ShowOnePSExecutor,
                            PostAnalysisPsFigExecutor)

logger = logging.getLogger(__name__)

# config_pyilc_t_HILC_backup

@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc_t")
def run_pyilc_analysis(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    pipeline_context.add_pipe(CommonRealPostExecutor)
    pipeline_context.add_pipe(CommonPyILCPredPostExecutor)
    pipeline_context.add_pipe(CommonNILCShowSimsPostExecutor)

    # pipeline_context.add_pipe(PixelAnalysisExecutor)
    # pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(PixelSummaryFigsExecutor)

    # # Not needed in every analysis pipeline, but needed in one
    # pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)  # Moved to main_convert_theory.py due to working_dir conflict.
    # pipeline_context.add_pipe(MakeTheoryPSStats)
    
    # # # PyILC's Predictions as Power Spectra Anaylsis
    # pipeline_context.add_pipe(MaskCreatorExecutor)
    # pipeline_context.add_pipe(PyILCMakePSExecutor)
    # pipeline_context.add_pipe(ShowOnePSExecutor)


    # pipeline_context.add_pipe(PowerSpectrumAnalysisExecutorSerial)
    # pipeline_context.add_pipe(PSAnalysisExecutor)
    # pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)
    # pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)
    # pipeline_context.add_pipe(PostAnalysisPsFigExecutor)

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
    run_pyilc_analysis()
