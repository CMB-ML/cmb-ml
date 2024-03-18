import logging

from .base_executor import BaseStageExecutor
from .experiment import ExperimentParameters


logger = logging.getLogger("stages")

class PipelineContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.experiment = ExperimentParameters.from_cfg(cfg)
        self.pipeline = []

    def add_pipe(self, executor):
        self.pipeline.append(executor)

    def run_pipeline(self):
        for executor in self.pipeline:
            self.run_executor(executor)

    def run_executor(self, stage):
        """
        Executes a specific stage in the pipeline.

        :param stage: The stage to run; should derive from src.core.stage_executors.base.BaseExecutor
        """
        logger.info(f"Running stage: {stage.__name__}")
        executor: BaseStageExecutor = stage(self.cfg, self.experiment)
        executor.execute()
        logger.info(f"Done running stage: {stage.__name__}")
