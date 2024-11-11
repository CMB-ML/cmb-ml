"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using CMBNNCS.

The pipeline consists of the following steps:
1. Preprocessing the data
2. Training the model
3. Predicting the cleaned CMB signal
4. Postprocessing the predictions
5. Converting predictions to common form for comparison across models
6. Generating per-pixel analysis results for each simulation
7. Generating per-pixel summary statistics for each simulation
8. Converting the theory power spectrum to a format that can be used for analysis
9. Generating per-ell power spectrum analysis results for each simulation
10. Generating per-ell power spectrum summary statistics for each simulation

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

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor
from cmbml.cmbnncs_local import (
                         HydraConfigCMBNNCSCheckerExecutor,
                         PreprocessMakeScaleExecutor,
                         PreprocessExecutor,
                        #  NonParallelPreprocessExecutor,
                         CheckTransformsExecutor,
                         TrainingExecutor,
                         PredictionExecutor,
                         PostprocessExecutor,
                         NonParallelPostprocessExecutor
                         )

from cmbml.analysis import (ShowSimsPrepExecutor, 
                            CommonRealPostExecutor,
                            CommonCMBNNCSPredPostExecutor,
                            CommonCMBNNCSShowSimsPostExecutor,
                            CMBNNCSShowSimsPredExecutor, 
                            # CMBNNCSShowSimsPostExecutor,
                            PixelAnalysisExecutor,
                            PixelSummaryExecutor,
                            ConvertTheoryPowerSpectrumExecutor,
                            MakeTheoryPSStats,
                            CMBNNCSMakePSExecutor,
                            PixelSummaryFigsExecutor,
                            PSAnalysisExecutor,
                            PowerSpectrumSummaryExecutor,
                            PowerSpectrumSummaryFigsExecutor,
                            PostAnalysisPsFigExecutor,
                            ShowOnePSExecutor)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs_unet8")
def run_cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(HydraConfigCMBNNCSCheckerExecutor)

    # pipeline_context.add_pipe(PreprocessMakeScaleExecutor)
    # # pipeline_context.add_pipe(NonParallelPreprocessExecutor)  # For demonstration only
    # pipeline_context.add_pipe(PreprocessExecutor)
    # pipeline_context.add_pipe(ShowSimsPrepExecutor)

    # pipeline_context.add_pipe(TrainingExecutor)

    # pipeline_context.add_pipe(PredictionExecutor)
    # pipeline_context.add_pipe(CMBNNCSShowSimsPredExecutor)
    # pipeline_context.add_pipe(PostprocessExecutor)
    # pipeline_context.add_pipe(NonParallelPostprocessExecutor)  # For demonstration only

    # pipeline_context.add_pipe(MaskCreatorExecutor)

    # In the following, "Common" means "Apply the same postprocessing to all models"; requires a mask
    # Apply to the target (CMB realization)
    pipeline_context.add_pipe(CommonRealPostExecutor)
    # Apply to CMBNNCS's predictions
    pipeline_context.add_pipe(CommonCMBNNCSPredPostExecutor)

    # Show results of cleaning
    pipeline_context.add_pipe(CommonCMBNNCSShowSimsPostExecutor)

    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)
    pipeline_context.add_pipe(PixelSummaryFigsExecutor)

    # These two do not need to run individually for all models (but they're fast, so it doesn't matter unless you're actively changing them)
    pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
    pipeline_context.add_pipe(MakeTheoryPSStats)

    # # # CMBNNCS's Predictions as Power Spectra Anaylsis
    # pipeline_context.add_pipe(CMBNNCSMakePSExecutor)
    pipeline_context.add_pipe(PSAnalysisExecutor)
    pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)
    pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)
    pipeline_context.add_pipe(PostAnalysisPsFigExecutor)
    # pipeline_context.add_pipe(ShowOnePSExecutor)  # Used for debugging; does not require full set of theory ps for simulations

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
    run_cmbnncs()
