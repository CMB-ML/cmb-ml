import logging
from cmbml.get_data.stage_executors.B_get_cmbml_data import GetNoiseModelExecutor


logger = logging.getLogger(__name__)


class DownloadNoiseModelExecutor(GetNoiseModelExecutor):
    """
    This is simply an alias for GetNoiseModelExecutor from the get_data
    portion of the code. It is renamed for clarity in the pipeline.
    """
