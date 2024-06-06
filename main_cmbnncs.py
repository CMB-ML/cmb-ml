from functools import partial
import logging

import hydra

from src.utils.check_env_var import validate_environment_variable
from src.core import (
                      PipelineContext,
                      LogMaker
                      )
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from src.sims import MaskCreatorExecutor
from src.cmbnncs_local import (
                         HydraConfigCMBNNCSCheckerExecutor,
                         PreprocessMakeScaleExecutor,
                         PreprocessExecutor,
                         CheckTransformsExecutor,
                         TrainingExecutor,
                         PredictionExecutor,
                         PostprocessExecutor
                         )

from src.analysis import   (ShowSimsPrepExecutor, 
                            CommonRealPostExecutor,
                            CommonCMBNNCSPredPostExecutor,
                            CommonCMBNNCSShowSimsPostExecutor,
                            CMBNNCSShowSimsPredExecutor, 
                            CMBNNCSShowSimsPostExecutor,
                            PixelAnalysisExecutor,
                            PixelSummaryExecutor,
                            ConvertTheoryPowerSpectrumExecutor,
                            MakeTheoryPSStats,
                            CMBNNCSMakePSExecutor,
                            PixelSummaryFigsExecutor,
                            PSAnalysisExecutor,
                            PowerSpectrumAnalysisExecutorSerial,
                            PowerSpectrumSummaryExecutor,
                            PowerSpectrumSummaryFigsExecutor,
                            PostAnalysisPsFigExecutor)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs_t")
def run_cmbnncs(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(HydraConfigCMBNNCSCheckerExecutor)

    # pipeline_context.add_pipe(PreprocessMakeScaleExecutor)
    # pipeline_context.add_pipe(PreprocessExecutor)
    # pipeline_context.add_pipe(ShowSimsPrepExecutor)

    # pipeline_context.add_pipe(TrainingExecutor)

    # pipeline_context.add_pipe(PredictionExecutor)
    # pipeline_context.add_pipe(CMBNNCSShowSimsPredExecutor)

    # pipeline_context.add_pipe(PostprocessExecutor)
    ## pipeline_context.add_pipe(CMBNNCSShowSimsPostExecutor)
    # pipeline_context.add_pipe(CommonRealPostExecutor)
    # pipeline_context.add_pipe(CommonCMBNNCSPredPostExecutor)
    pipeline_context.add_pipe(CommonCMBNNCSShowSimsPostExecutor)

    # pipeline_context.add_pipe(PixelAnalysisExecutor)
    # pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(PixelSummaryFigsExecutor)

    # # Not needed in every analysis pipeline, but needed in one
    # pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
    # pipeline_context.add_pipe(MakeTheoryPSStats)

    # CMBNNCS's Predictions as Power Spectra Anaylsis
    # pipeline_context.add_pipe(MaskCreatorExecutor)
    # pipeline_context.add_pipe(CMBNNCSMakePSExecutor)
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
        logger.info("Simulation pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    validate_environment_variable("CMB_SIMS_LOCAL_SYSTEM")
    run_cmbnncs()
