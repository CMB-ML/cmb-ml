import logging

import hydra

from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (

)
from cmbml.cmbfscnn_local import (

)

from cmbml.analysis import (

)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_cmbfscnn")
def run_cmbfscnn(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    pipeline_context.prerun_pipeline()

    # INSERT STUFF HERE

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
    run_cmbfscnn()