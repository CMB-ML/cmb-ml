import logging

from .executor_base import BaseStageExecutor


logger = logging.getLogger("stages")

class PipelineContext:
    def __init__(self, cfg, log_maker):
        self.cfg = cfg
        self.log_maker = log_maker
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
        executor: BaseStageExecutor = stage(self.cfg)
        executor.execute()
        logger.info(f"Done running stage: {stage.__name__}")
        if executor.make_stage_logs:
            stage_str = executor.stage_str
            stage_dir = self.cfg.pipeline[stage_str].dir_name
            self.log_maker.copy_hydra_run_to_stage_log(stage_dir)
