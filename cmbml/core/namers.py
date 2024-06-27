from typing import Dict

from pathlib import Path
from contextlib import contextmanager, ExitStack


class Namer:
    def __init__(self, cfg) -> None:
        self._root = Path(cfg.local_system.datasets_root)
        self._dataset_name: str = cfg.dataset_name
        self._working_dir: str = cfg.get("working_dir", "")
        self.sim_folder_prefix: str = cfg.file_system.sim_folder_prefix
        self.sim_str_num_digits: int = cfg.file_system.sim_str_num_digits
        self.src_root: str = cfg.local_system.assets_dir
        self.context: Dict[str, str] = {}

        # For use outside of pipeline executors
        self.default_path_template = cfg.file_system.default_dataset_template_str

        self._update_context()

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = Path(value)
        self._update_context()

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value
        self._update_context()

    @property
    def working_dir(self):
        return self._working_dir

    @working_dir.setter
    def working_dir(self, value):
        self._working_dir = value
        self._update_context()

    def _update_context(self):
        self.context['dataset'] = self._dataset_name
        self.context['working'] = self._working_dir
        self.context['root'] = str(self._root)
        self.context['src_root'] = self.src_root

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
        temp_context = dict(**self.context)
        if "sim" not in self.context and "sim_num" in self.context:
            temp_context["sim"] = self.sim_name()

        try:
            result_path_str = path_template.format(**temp_context)
        except KeyError as e:
            raise KeyError(f"Key {e.args[0]} not found in the context. Ensure that the path_template {path_template} is correct in the pipeline yaml.")
        return Path(result_path_str)

    @property
    def default_path(self):
        """
        For use outside of pipeline executors
        """
        return self.path(path_template=self.default_path_template)


class Sentinel:
    pass