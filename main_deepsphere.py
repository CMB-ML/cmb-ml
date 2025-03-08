"""
This script runs a pipeline for training and prediction of the cleaned CMB signal using Adams' Bayesian DeepSphere Model.

The pipeline consists of the following steps:
2. Training the model
3. Predicting the cleaned CMB signal

Usage:
    python main_deepsphere.py
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
# from cmbml.cmbnncs_local import (
#                          HydraConfigCMBNNCSCheckerExecutor,
#                          PreprocessMakeScaleExecutor,
#                          PreprocessExecutor,
#                          NonParallelPreprocessExecutor,
#                          TrainingExecutor,
#                          PredictionExecutor,
#                          PostprocessExecutor,
#                          NonParallelPostprocessExecutor
#                          )

from cmbml.deepsphere_unet import (
                            DeterministicTrainingExecutor,
                            PredictionExecutor,
                            BayesianTrainingExecutor,
                            BayesianPredictionExecutor
                            )

# from cmbml.analysis import (ShowSimsPrepExecutor, 
#                             CommonRealPostExecutor,
#                             CommonCMBNNCSPredPostExecutor,
#                             CommonCMBNNCSShowSimsPostExecutor,
#                             CommonCMBNNCSShowSimsPostIndivExecutor,
#                             CMBNNCSShowSimsPredExecutor, 
#                             # CMBNNCSShowSimsPostExecutor,
#                             PixelAnalysisExecutor,
#                             PixelSummaryExecutor,
#                             ConvertTheoryPowerSpectrumExecutor,
#                             MakeTheoryPSStats,
#                             CMBNNCSMakePSExecutor,
#                             PixelSummaryFigsExecutor,
#                             PowerSpectrumAnalysisExecutor,
#                             PowerSpectrumSummaryExecutor,
#                             PowerSpectrumSummaryFigsExecutor,
#                             PostAnalysisPsFigExecutor,
#                             ShowOnePSExecutor)

from cmbml.sims import MaskCreatorExecutor

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


@hydra.main(version_base=None, config_path="cfg", config_name="config_deepsphere")
def run_deepsphere(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # pipeline_context.add_pipe(DeterministicTrainingExecutor)
    # pipeline_context.add_pipe(PredictionExecutor)

    # pipeline_context.add_pipe(BayesianTrainingExecutor)
    pipeline_context.add_pipe(BayesianPredictionExecutor)


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
    run_deepsphere()
