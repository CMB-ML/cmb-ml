import logging

import hydra

from utils.check_env_var import validate_environment_variable
from core import (
                      PipelineContext,
                      LogMaker
                      )
from src.core.A_check_hydra_configs import HydraConfigCheckerExecutor
# from pyilc_local.B_predict_executor import PredictionExecutor
from analysis import (ShowSimsPrepExecutor, 
                      NILCShowSimsPostExecutor,
                      PixelAnalysisExecutor,
                      PixelSummaryExecutor,
                      ConvertTheoryPowerSpectrumExecutor,
                      MakePredPowerSpectrumExecutor,
                      ShowSinglePsFigExecutor
                      )


logger = logging.getLogger(__name__)

# config_pyilc_t_HILC_backup

# @hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc_t")
@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc_t_HILC_backup")
def make_all_simulations(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # Due to PyILC-matplotlib interaction, these cannot be imported at the same time
    # pipeline_context.add_pipe(PredictionExecutor)

    pipeline_context.add_pipe(NILCShowSimsPostExecutor)
    # pipeline_context.add_pipe(PixelAnalysisExecutor)
    # pipeline_context.add_pipe(PixelSummaryExecutor)

    pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)
    pipeline_context.add_pipe(MakePredPowerSpectrumExecutor)
    pipeline_context.add_pipe(ShowSinglePsFigExecutor)

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
    make_all_simulations()
