from typing import Dict

from pathlib import Path
from contextlib import contextmanager, ExitStack


class Namer:
    def __init__(self, cfg) -> None:
        self.root = Path(cfg.local_system.datasets_root)
        self.dataset_name: str = cfg.dataset_name
        self.sim_folder_prefix: str = cfg.file_system.sim_folder_prefix
        self.sim_str_num_digits: int = cfg.file_system.sim_str_num_digits

        self.context = {}

    @contextmanager
    def set_context(self, level, value):
        original_value = self.context.get(level, Sentinel)
        self.context[level] = value
        try:
            yield
        except Exception as e:
            raise e
        finally:
            if original_value is Sentinel:
                del self.context[level]
            else:
                self.context[level] = original_value

    @contextmanager
    def set_contexts(self, contexts_dict: Dict[str, str]):
        with ExitStack() as stack:
            # Create and enter all context managers
            for level, value in contexts_dict.items():
                stack.enter_context(self.set_context(level, value))
            yield

    def sim_name(self, sim_idx=None):
        if sim_idx is None:
            try:
                sim_idx = self.context["sim_num"]
            except KeyError:
                raise KeyError("No sim_num is currently set.")
        sim_name = self.sim_name_template.format(sim_idx=sim_idx)
        # sim_name = f"{self.sim_folder_prefix}{sim_idx:0{self.sim_str_num_digits}}"
        return sim_name

    @property
    def sim_name_template(self):
        template = f"{self.sim_folder_prefix}{{sim_idx:0{self.sim_str_num_digits}}}"
        return template

    def path(self, path_template: str):
        temp_context = dict(**self.context, 
                            root=str(self.root), 
                            dataset=self.dataset_name)
        if "sim" not in self.context and "sim_num" in self.context:
            temp_context["sim"] = self.sim_name()

        try:
            result_path_str = path_template.format(**temp_context)
        except KeyError as e:
            raise KeyError(f"Key {e.args[0]} likely not found in the context. Ensure that the path_template {path_template} is correct.")
        return Path(result_path_str)


class Sentinel:
    pass