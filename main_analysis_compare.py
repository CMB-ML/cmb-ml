from functools import partial
import logging

import hydra

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import PipelineContext, LogMaker
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor

from cmbml.analysis import   (
    PixelCompareTableExecutor,
    PSCompareTableExecutor,
    PostAnalysisPsCompareFigExecutor
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_comp_models")
def run_pyilc_analysis(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    # pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # pipeline_context.add_pipe(PixelCompareTableExecutor)
    # pipeline_context.add_pipe(PSCompareTableExecutor)
    pipeline_context.add_pipe(PostAnalysisPsCompareFigExecutor)

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
    run_pyilc_analysis()
