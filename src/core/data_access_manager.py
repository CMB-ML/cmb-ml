# from typing import Dict
# from contextlib import contextmanager

# from .namers import DatasetNamer, Namer
# from .data_assets import StageAccessManager, SplitAccessManager, SimAccessManager


# class DatasetAccessManager:
#     # Needs awareness of the experiment configuration, so it can access the correct fields and detector frequencies
#     def __init__(self, cfg) -> None:
#         self.cfg = cfg
#         self.namer_cfg = DatasetNamer(cfg)
#         self.map_fields = list(cfg.experiment.fields)
#         self.detectors = list(cfg.experiment.detector_freqs)

#         self.all_stages = self.make_all_stage_AMs(cfg)
#         self.all_splits = self.get_all_splits(cfg)

#         self.current_split: SplitAccessManager = None

#     @property
#     def make_namer(self):
#         return Namer(
#             root = self.namer_cfg.path,
#             dataset = self.namer_cfg.name,
#             stage = self.current_stage,
#             split = self.current_split,
#             sim = None,
#             fn = None
#         )

#     def make_all_stage_AMs(self, cfg):
#         stage_names = cfg.pipeline.keys()
#         stages: Dict[str, StageAccessManager] = {}
#         for name in stage_names:
#             stages[name] = StageAccessManager(name, cfg.pipeline[name])
#         # Create assets for each stage AFTER all stages have been created
#         for stage in stages.values(): stage.make_assets_out(dam=self)
#         for stage in stages.values(): stage.make_assets_in(dam=self, stages=stages)
#         return stages

#     def get_stage(self, name) -> StageAccessManager:
#         try:
#             return self.all_stages[name]
#         except KeyError:
#             raise ValueError(f"No stage with name {name} found.")

#     def get_all_splits(self, cfg):
#         split_names = cfg.splits.keys()
#         return [SplitAccessManager(name, cfg.splits[name]) for name in split_names]

#     def set_current_split(self, split: SplitAccessManager):
#         self.current_split = split

#     def unset_current_split(self):
#         self.current_split = None

#     @contextmanager
#     def split_context(self, split):
#         self.set_current_split(split)
#         try:
#             yield
#         except Exception as e:
#             raise e
#         finally:
#             self.unset_current_split()

#     def check_context(self) -> None:
#         # if self.current_stage is None:
#         #     raise ValueError("Current stage is None. Is your Executor execute using @withpipe?")
#         if self.current_split is None:
#             raise ValueError("Current split is None. Is your Executor process_split using @withsplit?")

#     def iter_sims(self):
#         self.check_context()
#         return SimIterator(self)

#     def get_applicable_splits(self, scope):
#         import re
#         kinds_of_splits = [kind.lower() for kind in scope]
#         patterns = [re.compile(f"^{kind}\\d*$", re.IGNORECASE) for kind in kinds_of_splits]

#         filtered_names = []
#         for split in self.all_splits:
#             if any(pattern.match(split.name) for pattern in patterns):
#                 filtered_names.append(split)
#         return filtered_names

#     def get_assets(self, stage_str):
#         stage = self.get_stage(stage_str)
#         return stage.assets_in, stage.assets_out


# class SimIterator:
#     def __init__(self, dam: DatasetAccessManager):
#         self.dam = dam
#         self.index: int = None  # Current simulation index

#     def __iter__(self):
#         self.index = None  # Reset index on new iteration
#         return self

#     def __next__(self):
#         # TODO: Check if this iterator is antipythonic in implementation
#         if self.index is None:
#             self.index = 0
#         elif self.index < self.dam.current_split.n_sims - 1:
#             self.index += 1
#         else:
#             raise StopIteration
#         sim = SimAccessManager(self.index, self.dam)
#         return sim
