import logging
from .executor_base import BaseStageExecutor

logger = logging.getLogger("stages")

class PipelineContext:
    def __init__(self, cfg, log_maker=None):
        """
        Initialize a PipelineContext object.

        Parameters:
        cfg (object): The configuration object.
        log_maker (object): The log maker object or None.

        Returns:
        None
        """
        self.cfg = cfg
        self.log_maker = log_maker
        self.pipeline = []

    def add_pipe(self, executor: BaseStageExecutor):
        """
        Append an executor to the pipeline.

        Parameters:
        executor (BaseStageExecutor): An executor object.

        Returns:
        None
        """
        self.pipeline.append(executor)

    def prerun_pipeline(self):
        """
        Perform pre-run checks for each stage in the pipeline.

        This method initializes each executor in the pipeline to check for any issues 
        that may arise early, such as pulling resources from configs. However, it will 
        not detect issues with data assets created in the pipeline.

        Returns:
        None
        """
        logger.info("Performing pre-run checks. Trying __init__() method for each stage to check for obvious issues.")
        for stage in self.pipeline:
            logger.info(f"Checking initialization for: {stage.__name__}")
            executor: BaseStageExecutor = stage(self.cfg)
        logger.info("Pre-run checks complete.")

    def run_pipeline(self):
        """
        Run the pipeline by executing each executor in order.

        Returns:
        None
        """
        for executor in self.pipeline:
            self._run_executor(executor)

    def _run_executor(self, stage: BaseStageExecutor):
        """
        Execute a specific stage in the pipeline.

        Parameters:
        stage (BaseStageExecutor): The stage to run.

        Returns:
        None
        """
        logger.info(f"Running stage: {stage.__name__}")
        executor: BaseStageExecutor = stage(self.cfg)
        had_exception = False
        try:
            executor.execute()
        except Exception as e:
            had_exception = True
            logger.exception(f"An exception occurred during stage: {stage.__name__}", exc_info=e)
            raise e
        finally:
            if not had_exception:
                logger.info(f"Done running stage: {stage.__name__}")
            if executor.make_stage_logs:
                stage_str = executor.stage_str
                top_level_working = executor.top_level_working
                stage_dir = self.cfg.pipeline[stage_str].dir_name
                if self.log_maker is not None:
                    self.log_maker.copy_hydra_run_to_stage_log(stage_dir, top_level_working)
            else:
                logger.warning(f"Skipping stage logs for stage {stage.__name__}.")
