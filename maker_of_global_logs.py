import shutil
import ast
from pathlib import Path
from os.path import commonpath

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from namer_logs import LogsNamer


class LogMaker:
    def __init__(self, 
                 conf: DictConfig, 
                 dataset_files_path:Path) -> None:
        
        self.namer = LogsNamer(conf, HydraConfig.get(), dataset_files_path)

    def log_scripts_to_hydra(self, source_script) -> None:
        target_root = self.namer.hydra_scripts_path
        target_root.mkdir(parents=True, exist_ok=True)

        imported_local_py_files = self._find_local_imports(source_script)
        base_path = self._find_common_paths(imported_local_py_files)

        for py_path in imported_local_py_files:
            relative_py_path = py_path.relative_to(base_path)
            target_path = target_root / relative_py_path

            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(py_path, target_path)

    def _find_local_imports(self, source_script):
        def find_imports(_filename, _base_path, _visited_files):
            """
            Recursively find and process local imports from a given file.
            """
            if _filename in _visited_files:
                return
            _visited_files.add(_filename)

            with _filename.open("r") as file:
                tree = ast.parse(file.read(), filename=str(_filename))

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in (node.names if isinstance(node, ast.Import) else [node]):
                        module_name = alias.name if isinstance(node, ast.Import) else node.module
                        if module_name is None:
                            continue  # Skip relative import without 'from'
                        module_path = get_full_path(module_name, _base_path)
                        if module_path and module_path.exists():
                            find_imports(module_path, _base_path, _visited_files)

        def get_full_path(_module_name, _base_path):
            """
            Convert a module name to a full file path within a given base path.
            """
            parts = _module_name.split('.')
            path = _base_path.joinpath(*parts)
            # Check for direct .py file or package (__init__.py)
            if path.with_suffix('.py').exists():
                return path.with_suffix('.py')
            elif (path / '__init__.py').exists():
                return path / '__init__.py'
            return None

        base_path = Path(source_script).parent
        visited_files = set()
        find_imports(Path(source_script), base_path, visited_files)
        return visited_files

    @staticmethod
    def _find_common_paths(paths):
        """Finds the most common base path for a list of Path objects."""
        # Ensure all paths are absolute
        absolute_paths = [path.resolve() for path in paths]
        # Use os.path.commonpath to find the common base directory
        common_base = commonpath(absolute_paths)
        return Path(common_base)    

    def copy_hydra_run_to_dataset_log(self):
        self.namer.dataset_logs_path.mkdir(parents=True, exist_ok=True)
        
        for item in self.namer.hydra_path.iterdir():
            destination = self.namer.dataset_logs_path / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)  # For directories
            else:
                shutil.copy2(item, destination)  # For files
