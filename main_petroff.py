import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from core import (
                      PipelineContext,
                      LogMaker
                      )
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from petroff import (
                     PreprocessMakeExtremaExecutor,
                     CheckTransformsExecutor,
                     TrainingExecutor,
                     PredictionExecutor,
                     )
from analysis import (
                      PetroffShowSimsPostExecutor,
                      PixelAnalysisExecutor,
                      PixelSummaryExecutor,
                      ConvertTheoryPowerSpectrumExecutor,
                      MakePredPowerSpectrumExecutor,
                      ShowSinglePsFigExecutor
                      )



logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_petroff")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    # Write hydra's full composed config to a log
    logging.getLogger("hydraconf").info(OmegaConf.to_yaml(HydraConfig.get()))

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    # pipeline_context.add_pipe(HydraConfigCMBNNCSCheckerExecutor)

    pipeline_context.add_pipe(PreprocessMakeExtremaExecutor)
    pipeline_context.add_pipe(CheckTransformsExecutor)
    pipeline_context.add_pipe(TrainingExecutor)
    pipeline_context.add_pipe(PredictionExecutor)
    pipeline_context.add_pipe(PetroffShowSimsPostExecutor)



    # pipeline_context.add_pipe(PixelAnalysisExecutor)
    # pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
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
