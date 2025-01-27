"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using <TODO: Name it>

The pipeline consists of the following steps:
1. <TODO: Steps here>

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_<TODO: Name it>.py
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
from cmbml.demo_patch_nn import (
    MakeLutExecutor,
    ChoosePatchesExecutor,
    FindDatasetStatsParallelExecutor,
    PreprocessPatchesExecutor,
    TrainingExecutor,
    PredictExectutor
    )

from cmbml.analysis import (
    CommonRealPostExecutor,
    CommonNNPredPostExecutor,
    CommonNNShowSimsPostExecutor,
    PixelAnalysisExecutor,
    PixelSummaryExecutor,
    PixelSummaryFigsExecutor,
    NNMakePowerSpectrumExecutor,
    PowerSpectrumAnalysisExecutor,
    PowerSpectrumSummaryExecutor,
    PowerSpectrumSummaryFigsExecutor,
    PostAnalysisPsFigExecutor,
    )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_patch_nn")
def run_cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # Cleaning: Preprocessing
    pipeline_context.add_pipe(MakeLutExecutor)
    pipeline_context.add_pipe(MaskCreatorExecutor)
    pipeline_context.add_pipe(ChoosePatchesExecutor)
    pipeline_context.add_pipe(FindDatasetStatsParallelExecutor)
    pipeline_context.add_pipe(PreprocessPatchesExecutor)

    # Cleaning: Training and Prediction
    pipeline_context.add_pipe(TrainingExecutor)
    pipeline_context.add_pipe(PredictExectutor)

    # Analysis: processing the maps in a common way
    pipeline_context.add_pipe(CommonRealPostExecutor)
    pipeline_context.add_pipe(CommonNNPredPostExecutor)

    # Analysis: showing the maps (pixel-level result)
    pipeline_context.add_pipe(CommonNNShowSimsPostExecutor)

    # Analysis: getting & presenting pixel-level statistics
    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)
    pipeline_context.add_pipe(PixelSummaryFigsExecutor)

    # Analysis: generating the power spectra and power spectra statistics
    pipeline_context.add_pipe(NNMakePowerSpectrumExecutor)
    pipeline_context.add_pipe(PowerSpectrumAnalysisExecutor)
    pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)
    pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)
    pipeline_context.add_pipe(PostAnalysisPsFigExecutor)

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
