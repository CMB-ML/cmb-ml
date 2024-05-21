import logging

import hydra

from core import (
                      PipelineContext,
                      LogMaker
                      )
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from src.cmbnncs_local import (
                         HydraConfigCMBNNCSCheckerExecutor,
                         PreprocessMakeScaleExecutor,
                         PreprocessExecutor,
                         CheckTransformsExecutor,
                         TrainingExecutor,
                         PredictionExecutor,
                         PostprocessExecutor
                         )
from analysis import (ShowSimsPrepExecutor, 
                      ShowSimsPredExecutor, 
                      CMBNNCSShowSimsPostExecutor,
                      PixelAnalysisExecutor,
                      PixelSummaryExecutor,
                      ConvertTheoryPowerSpectrumExecutor,
                      MakePredPowerSpectrumExecutor,
                      ShowSinglePsFigExecutor
                      )


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbnncs_t")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    pipeline_context.add_pipe(HydraConfigCMBNNCSCheckerExecutor)

    pipeline_context.add_pipe(PreprocessMakeScaleExecutor)
    # pipeline_context.add_pipe(PreprocessExecutor)
    # pipeline_context.add_pipe(CheckTransformsExecutor)  # Transforms are not currently workable for CMBNNCS
    pipeline_context.add_pipe(ShowSimsPrepExecutor)

    # pipeline_context.add_pipe(TrainingExecutor)

    pipeline_context.add_pipe(PredictionExecutor)
    pipeline_context.add_pipe(ShowSimsPredExecutor)
    pipeline_context.add_pipe(PostprocessExecutor)
    pipeline_context.add_pipe(CMBNNCSShowSimsPostExecutor)
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
        logger.info("Simulation pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    make_all_simulations()
