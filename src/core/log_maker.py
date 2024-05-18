import pkg_resources
from importlib.metadata import distributions
import shutil
import ast
import yaml
from pathlib import Path
from os.path import commonpath

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import logging
from .namers import Namer


logger = logging.getLogger(__name__)


class LogMaker:
    def __init__(self, 
                 cfg: DictConfig) -> None:

        self.namer = LogsNamer(cfg, HydraConfig.get())
        self.source_dir = "src"

    def log_procedure_to_hydra(self, source_script) -> None:
        target_root = self.namer.hydra_scripts_path
        target_root.mkdir(parents=True, exist_ok=True)
        self.log_py_to_hydra(source_script, target_root)
        self.log_cfgs_to_hydra(target_root)
        self.log_poetry_lock(source_script, target_root)
        self.log_library_versions(target_root)

    def log_library_versions(self, target_root):
        """
        Logs the versions of all installed packages in the current environment to a requirements.txt file using importlib.metadata.

        Args:
        target_root (str): The root directory where the requirements.txt file will be saved.
        """
        target_path = Path(target_root) / "requirements.txt"
        package_list = []

        for dist in distributions():
            package_list.append(f"{dist.metadata['Name']}=={dist.version}")

        with target_path.open("w") as f:
            f.write("\n".join(package_list))

    def log_poetry_lock(self, source_script, target_root):
        poetry_lock_path = Path(source_script).parent / "poetry.lock"
        if poetry_lock_path.exists():
            target_path = Path(target_root) / "poetry.lock"
            shutil.copy(poetry_lock_path, target_path)

    def log_py_to_hydra(self, source_script, target_root):
        imported_local_py_files = self._find_local_imports(source_script, self.source_dir)
        base_path = self._find_common_paths(imported_local_py_files)

        for py_path in imported_local_py_files:
            # resolve() is needed; absolute path not guaranteed
            relative_py_path = py_path.resolve().relative_to(base_path)
            target_path = target_root / relative_py_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(py_path, target_path)

    def log_cfgs_to_hydra(self, target_root):
        relevant_config_files = self.extract_relevant_config_paths()
        base_path = self._find_common_paths(relevant_config_files)
        # We want to include the parent of all configs in the path for organization. 
        #   Otherwise they're alongside the python files.
        base_path = base_path.parent

        for config_file in relevant_config_files:
            # resolve() is needed; absolute path not guaranteed
            relative_cfg_path = config_file.resolve().relative_to(base_path)
            target_path = target_root / relative_cfg_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(config_file, target_path)

    def extract_relevant_config_paths(self):
        hydra_cfg = HydraConfig.get()

        # Filtering relevant choices
        relevant_choices = {}
        for k, v in hydra_cfg.runtime.choices.items():
            if 'hydra/' not in k:
                if v in ['default', 'null', 'basic']:
                    continue
                relevant_choices[k] = v

        # Extracting paths that are not provided by 'hydra' or 'schema'
        config_paths = [
            Path(source['path']) for source in hydra_cfg.runtime.config_sources
            if source['provider'] not in ['hydra', 'schema'] and source['path']
        ]

        relevant_files = []
        missing_combinations = []

        # Attempting to find each choice in the available config paths
        for choice_key, choice_value in relevant_choices.items():
            found = False
            for config_dir in config_paths:
                config_path = config_dir / f"{choice_key}/{choice_value}.yaml"
                if config_path.exists():
                    relevant_files.append(config_path)
                    found = True
                    break
            if not found:
                missing_combinations.append((choice_key, choice_value))

        # Logging or handling missing configurations
        if missing_combinations:
            logger.warning("Missing configuration files for:", missing_combinations)

        # We sideload yaml files in some cases; hydra does not add these to the "choices" list.
        for config_path in relevant_files:
            config_path = Path(config_path)  # Del this line
            with open(config_path, 'r') as f:
                try:
                    config_data = yaml.safe_load(f)
                    if config_data is None:
                        logger.warning(f"Loaded an empty config file: {config_path}")
                        continue
                    # if defaults is not a key, continue
                    defaults = config_data.get('defaults', [])
                    for item in defaults:
                        # only consider strings, not k:v pairs (or maybe other things)
                        if not isinstance(item, str):
                            continue
                        # _self_ must appear in the defaults list; this is ok and
                        if item == '_self_':
                            continue
                        # if it's a string, hydra interprets it as a path neighboring the current config
                        possible_path = config_path.parent / item
                        # Suffixes in the defaults list are optional ('- test' and '- test.yaml' are equivalent)
                        if not possible_path.suffix:
                            possible_path = possible_path.with_suffix('.yaml')
                        if possible_path.exists():
                            if possible_path not in relevant_files:
                                # Generally dangerous, however in this case we cannot create an infinite loop
                                relevant_files.append(possible_path)
                            else:
                                logger.warning(f"Circular dependency detected for file: {config_path} for line {item}")
                        else:
                            logger.warning(f"File referenced in a defaults was not found: {config_path} for line {item}")
                except yaml.YAMLError as e:
                    logger.error(f"Error parsing YAML file {config_path}: {e}")
        return relevant_files

    @staticmethod
    def _find_local_imports(source_script, source_dir):
        visited_files = set()
        unresolved_imports = set()

        def find_imports(_filename, _current_dir, _visited_files):
            """
            Recursively find and process local imports from a given file.
            """
            if _filename in _visited_files:
                return
            _visited_files.add(_filename)

            with _filename.open("r") as file:
                tree = ast.parse(file.read(), filename=str(_filename))

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        # Handle relative imports without 'from' part
                        level = node.level  # Number of dots in import
                        module_path = _current_dir
                        for _ in range(level - 1):  # Navigate up the directory tree as required
                            module_path = module_path.parent
                        module_file = module_path / '__init__.py'
                        if module_file.exists() and is_within_project(module_file, source_path):
                            find_imports(module_file, module_path, _visited_files)
                        else:
                            unresolved_imports.add()
                    else:
                        # Handle imports from the project root or relative imports
                        parts = node.module.split('.')
                        # Sometimes, "src.whatever" is imported instead of just "whatever". Handle this:
                        if parts[0] == source_dir:
                            parts = parts[1:]
                        # Number of dots before the import; if >0, we backtrack up the directory structure.
                        level = node.level
                        if level == 0:
                            # Import from the project root (e.g., 'src')
                            module_path = source_path
                        else:
                            # Relative import, navigate up the directory tree as needed
                            module_path = _current_dir
                            for _ in range(level - 1):
                                module_path = module_path.parent
                        target_path = module_path.joinpath(*parts)
                        # Check for .py file or package (__init__.py)
                        if target_path.with_suffix('.py').exists() and is_within_project(target_path.with_suffix('.py'), source_path):
                            find_imports(target_path.with_suffix('.py'), target_path.parent, _visited_files)
                        elif (target_path / '__init__.py').exists() and is_within_project(target_path / '__init__.py', source_path):
                            find_imports(target_path / '__init__.py', target_path, _visited_files)
                        else:
                            # print(node.module)
                            unresolved_imports.add(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        module_path = get_full_path(module_name, _current_dir)
                        if module_path and module_path.exists() and is_within_project(module_path, source_path):
                            find_imports(module_path, module_path.parent, _visited_files)
                        else:
                            # print(module_name)
                            unresolved_imports.add(module_name)
                    # module_name = node.names[0].name
                    # module_path = get_full_path(module_name, _current_dir)
                    # if module_path and module_path.exists() and is_within_project(module_path, source_dir):
                    #     find_imports(module_path, module_path.parent, _visited_files)
                    # else:
                    #     temp_name = ".".join([n.name for n in node.names])
                    #     unresolved_imports.add(temp_name)

        def get_full_path(_module_name, _current_dir):
            """
            Convert a module name to a full file path within a given base path.
            """
            parts = _module_name.split('.')
            path = _current_dir.joinpath(*parts)
            if path.with_suffix('.py').exists():
                return path.with_suffix('.py')
            elif (path / '__init__.py').exists():
                return path / '__init__.py'
            return None

        def is_within_project(_path, _base_dir):
            """
            Check if a given path is within the project directory.
            """
            try:
                _path.relative_to(_base_dir)
                return True
            except ValueError:
                return False

        base_path = Path(source_script).parent
        source_path = base_path / source_dir
        visited_files = set()
        find_imports(Path(source_script), source_path, visited_files)

        if unresolved_imports:
            logger = logging.getLogger('unresolved_imports')
            logger.info("\n".join(sorted(unresolved_imports)))

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
        self.copy_hydra_run_to_log(self.namer.dataset_logs_path)

    def copy_hydra_run_to_stage_log(self, stage):
        stage_path = self.namer.stage_logs_path(stage)
        stage_path.mkdir(parents=True, exist_ok=True)
        self.copy_hydra_run_to_log(stage_path)

    def copy_hydra_run_to_log(self, target_root):
        for item in self.namer.hydra_path.iterdir():
            destination = target_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)  # For directories
            else:
                shutil.copy2(item, destination)  # For files


class LogsNamer:
    def __init__(self, 
                 cfg: DictConfig,
                 hydra_config: HydraConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.hydra_run_root = Path(hydra_config.runtime.cwd)
        self.hydra_run_dir = hydra_config.run.dir
        self.scripts_subdir = cfg.file_system.subdir_for_log_scripts
        # self.dataset_logs_dir = cfg.file_system.subdir_for_logs
        self.dataset_template_str = cfg.file_system.log_dataset_template_str
        self.stage_template_str = cfg.file_system.log_stage_template_str
        self.namer = Namer(cfg)

    @property
    def hydra_path(self) -> Path:
        return self.hydra_run_root / self.hydra_run_dir

    @property
    def hydra_scripts_path(self) -> Path:
        return self.hydra_path / self.scripts_subdir

    @property
    def dataset_logs_path(self) -> Path:
        with self.namer.set_context("hydra_run_dir", self.hydra_run_dir):
            path = self.namer.path(self.dataset_template_str)
        return path

    def stage_logs_path(self, stage_dir) -> Path:
        with self.namer.set_contexts({"hydra_run_dir": self.hydra_run_dir,
                                      "stage": stage_dir}):
            path = self.namer.path(self.stage_template_str)
        return path
