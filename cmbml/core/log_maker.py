# import pkg_resources
from importlib.resources import files
from importlib.metadata import distributions
import shutil
import ast
import yaml
import zipfile
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
        self.source_dir = "cmbml"

    def log_procedure_to_hydra(self, source_script) -> None:
        target_root = self.namer.hydra_scripts_path
        target_root.mkdir(parents=True, exist_ok=True)
        self.log_py_to_hydra(source_script, target_root)
        self.log_cfgs_to_hydra(target_root)
        self.log_poetry_lock(source_script, target_root)
        self.log_library_versions(target_root)

    def log_library_versions(self, target_root):
        """
        Logs the versions of all installed packages in the current environment
        to a requirements.txt file using importlib.metadata.
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
        """
        Collects Python source files reachable from source_script and archives
        them as zip files under target_root.

        Two source roots are considered:
          1. Local packages sitting next to source_script (e.g. d2ps_nn/, pyilc_local/).
             These are discovered by scanning the script's directory for importable
             packages and checking which ones are actually imported.
          2. The installed cmbml package, located via cmbml.__file__.

        Each source root is archived as a separate zip:
            <target_root>/<package_name>.zip
        The zip preserves the internal directory structure of the package so that
        it remains human-browsable without extraction.
        """
        source_script = Path(source_script)
        script_dir = source_script.parent

        # --- 1. Collect files from local packages next to the script ---
        local_package_roots = self._find_local_package_roots(script_dir)
        for pkg_root in local_package_roots:
            py_files = self._find_local_imports(source_script, pkg_root)
            if py_files:
                zip_name = pkg_root.name + ".zip"
                self._archive_files(py_files, pkg_root.parent, target_root / zip_name)

        # --- 2. Collect files from the installed cmbml package ---
        cmbml_root = self._find_installed_cmbml_root()
        if cmbml_root is not None:
            cmbml_files = self._find_installed_imports(source_script, cmbml_root)
            if cmbml_files:
                zip_name = cmbml_root.name + ".zip"
                self._archive_files(cmbml_files, cmbml_root.parent, target_root / zip_name)
        else:
            logger.warning("Could not locate installed cmbml package for logging.")

    # ------------------------------------------------------------------
    # Source-root discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_local_package_roots(script_dir: Path) -> list:
        """
        Returns a list of package directories (containing __init__.py) that sit
        directly next to the calling script. These are candidates for local,
        non-installed libraries (e.g. d2ps_nn/, pyilc_local/).
        """
        roots = []
        for item in script_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                roots.append(item)
        return roots

    @staticmethod
    def _find_installed_cmbml_root() -> Path:
        """
        Locates the root directory of the installed cmbml package via its
        __file__ attribute. Returns None if cmbml cannot be imported.
        """
        try:
            import cmbml
            return Path(cmbml.__file__).parent
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Import-tracing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_local_imports(source_script: Path, pkg_root: Path) -> set:
        """
        Traces all Python files reachable from source_script whose resolved
        path falls inside pkg_root.

        This covers the case where the library lives next to the script and is
        not installed (e.g. d2ps_nn/ or pyilc_local/).
        """
        return LogMaker._trace_imports(
            start_file=Path(source_script),
            start_dir=Path(source_script).parent,
            allowed_root=pkg_root,
        )

    @staticmethod
    def _find_installed_imports(source_script: Path, pkg_root: Path) -> set:
        """
        Traces all Python files reachable from source_script whose resolved
        path falls inside pkg_root (the installed package directory).

        Because the installed package is not next to the script, we seed the
        walk with the package's own __init__.py once we detect that source_script
        imports from it, then recursively follow imports within the package.
        """
        # Check whether source_script actually imports from this package at all.
        pkg_name = pkg_root.name  # e.g. "cmbml"
        if not LogMaker._script_imports_package(source_script, pkg_name):
            return set()

        # Seed the walk from the package __init__.py.
        init_file = pkg_root / "__init__.py"
        if not init_file.exists():
            logger.warning(f"No __init__.py found in installed package root: {pkg_root}")
            return set()

        return LogMaker._trace_imports(
            start_file=init_file,
            start_dir=pkg_root,
            allowed_root=pkg_root,
        )

    @staticmethod
    def _script_imports_package(script_path: Path, pkg_name: str) -> bool:
        """
        Returns True if script_path contains any import statement that
        references pkg_name at the top level.
        """
        try:
            with script_path.open("r") as f:
                tree = ast.parse(f.read(), filename=str(script_path))
        except (OSError, SyntaxError) as e:
            logger.warning(f"Could not parse {script_path} for import check: {e}")
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == pkg_name:
                    return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == pkg_name:
                        return True
        return False

    @staticmethod
    def _trace_imports(start_file: Path, start_dir: Path, allowed_root: Path) -> set:
        """
        Recursively walks imports starting from start_file, collecting all .py
        files whose resolved path is inside allowed_root.

        Every visited file (inside or outside allowed_root) is added to `seen`
        to prevent infinite loops. Only files inside allowed_root are added to
        `collected` and returned. This means start_file itself is walked for its
        imports even if it lives outside allowed_root (e.g. it's the calling
        script), but it is not included in the output.

        Args:
            start_file:    The .py file to start tracing from.
            start_dir:     The directory considered "current" for relative imports.
            allowed_root:  Only files inside this directory are collected.
        Returns:
            A set of Path objects for every reachable .py file inside allowed_root.
        """
        collected = set()   # files inside allowed_root (the output)
        seen = set()        # all visited files (loop prevention)
        unresolved = set()

        def _is_inside(path: Path) -> bool:
            try:
                path.resolve().relative_to(allowed_root.resolve())
                return True
            except ValueError:
                return False

        def _get_full_path(module_name: str, current_dir: Path):
            parts = module_name.split(".")
            path = current_dir.joinpath(*parts)
            if path.with_suffix(".py").exists():
                return path.with_suffix(".py")
            if (path / "__init__.py").exists():
                return path / "__init__.py"
            return None

        def _walk(filename: Path, current_dir: Path):
            filename = filename.resolve()
            if filename in seen:
                return
            seen.add(filename)
            if _is_inside(filename):
                collected.add(filename)

            try:
                with filename.open("r") as fh:
                    tree = ast.parse(fh.read(), filename=str(filename))
            except (OSError, SyntaxError) as e:
                logger.warning(f"Could not parse {filename}: {e}")
                return

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        # Bare relative import: "from . import something"
                        level = node.level
                        mod_path = current_dir
                        for _ in range(level - 1):
                            mod_path = mod_path.parent
                        init = mod_path / "__init__.py"
                        if init.exists():
                            _walk(init, mod_path)
                    else:
                        parts = node.module.split(".")
                        # Strip leading package name if it matches allowed_root
                        if parts[0] == allowed_root.name:
                            parts = parts[1:]
                        level = node.level
                        if level == 0:
                            # Absolute import — resolve from allowed_root
                            mod_path = allowed_root
                        else:
                            # Relative import — navigate up from current_dir
                            mod_path = current_dir
                            for _ in range(level - 1):
                                mod_path = mod_path.parent
                        target = mod_path.joinpath(*parts)
                        if target.with_suffix(".py").exists():
                            _walk(target.with_suffix(".py"), target.parent)
                        elif (target / "__init__.py").exists():
                            _walk(target / "__init__.py", target)
                        else:
                            unresolved.add(node.module)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        full_path = _get_full_path(alias.name, current_dir)
                        if full_path and full_path.exists():
                            _walk(full_path, full_path.parent)
                        else:
                            unresolved.add(alias.name)

        _walk(start_file, start_dir)

        if unresolved:
            unresolved_logger = logging.getLogger("unresolved_imports")
            unresolved_logger.info("\n".join(sorted(unresolved)))

        return collected

    # ------------------------------------------------------------------
    # Archiving helper
    # ------------------------------------------------------------------

    @staticmethod
    def _archive_files(py_files: set, base_path: Path, zip_path: Path):
        """
        Archives a collection of .py files into a zip at zip_path.

        Each file's path inside the zip is relative to base_path, so the zip
        remains human-browsable (e.g. cmbml/core/log_maker.py).

        Args:
            py_files:  Set of absolute Path objects to include.
            base_path: The root used to compute relative paths inside the zip.
            zip_path:  Destination .zip file path.
        """
        base_path = base_path.resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for py_file in sorted(py_files):
                try:
                    arcname = py_file.resolve().relative_to(base_path)
                    zf.write(py_file, arcname)
                except ValueError:
                    # File is outside base_path (shouldn't happen, but be safe)
                    logger.warning(f"Skipping file outside base_path during archiving: {py_file}")

    # ------------------------------------------------------------------
    # Config logging (unchanged)
    # ------------------------------------------------------------------

    def log_cfgs_to_hydra(self, target_root):
        relevant_config_files = self.extract_relevant_config_paths()
        
        with open(target_root / "config_sources.txt", "w") as f:
            for provider, config_files in relevant_config_files.items():
                f.write(f"{provider}\n")
                f.write(f"Common path: {self._find_common_paths(config_files)}\n")
                for config_file in config_files:
                    f.write(f"    {config_file}\n")

        for provider, config_files in relevant_config_files.items():
            base_path = self._find_common_paths(config_files)
            base_path = base_path.parent

            for config_file in config_files:
                relative_cfg_path = config_file.resolve().relative_to(base_path)
                target_path = target_root / provider / relative_cfg_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(config_file, target_path)

    def extract_relevant_config_paths(self):
        hydra_cfg = HydraConfig.get()

        relevant_choices = {}
        for k, v in hydra_cfg.runtime.choices.items():
            if 'hydra/' not in k:
                if v in ['default', 'null', 'basic']:
                    continue
                relevant_choices[k] = v

        config_paths = {}
        for source in hydra_cfg.runtime.config_sources:
            if source['provider'] not in ['hydra', 'schema'] and source['path']:
                if source['schema'] == 'file':
                    config_paths[source['provider']] = (Path(source['path']))
                elif source['schema'] == 'pkg':
                    path_parts = source['path'].split('.')
                    path = Path(files(path_parts[0]))
                    for part in path_parts[1:]:
                        path = path / part
                    config_paths[source['provider']] = path

        relevant_files = {}
        top_config_name = hydra_cfg.job.config_name
        for provider, config_path in config_paths.items():
            maybe_path = config_path / f"{top_config_name}.yaml"
            if maybe_path.exists():
                relevant_files[provider] = [maybe_path]
            else:
                relevant_files[provider] = []

        missing_combinations = []

        for choice_key, choice_value in relevant_choices.items():
            found = False
            for provider, config_dir in config_paths.items():
                config_path = config_dir / f"{choice_key}/{choice_value}.yaml"
                if config_path.exists():
                    relevant_files[provider].append(config_path)
                    found = True
                    break
            if not found:
                missing_combinations.append((choice_key, choice_value))

        if missing_combinations:
            logger.warning("Missing configuration files for:", missing_combinations)

        for provider, config_paths in relevant_files.items():
            for config_path in config_paths:
                config_path = Path(config_path)
                with open(config_path, 'r') as f:
                    try:
                        config_data = yaml.safe_load(f)
                        if config_data is None:
                            logger.warning(f"Loaded an empty config file: {config_path}")
                            continue
                        defaults = config_data.get('defaults', [])
                        for item in defaults:
                            if not isinstance(item, str):
                                continue
                            if item == '_self_':
                                continue
                            possible_path = config_path.parent / item
                            if not possible_path.suffix:
                                possible_path = possible_path.with_suffix('.yaml')
                            if possible_path.exists():
                                if possible_path not in relevant_files[provider]:
                                    relevant_files[provider].append(possible_path)
                                else:
                                    logger.warning(f"Circular dependency detected for file: {config_path} for line {item}")
                            else:
                                logger.warning(f"File referenced in a defaults was not found: {config_path} for line {item}")
                    except yaml.YAMLError as e:
                        logger.error(f"Error parsing YAML file {config_path}: {e}")
        return relevant_files

    @staticmethod
    def _find_common_paths(paths):
        """Finds the most common base path for a list of Path objects."""
        absolute_paths = [path.resolve() for path in paths]
        common_base = commonpath(absolute_paths)
        return Path(common_base)

    def copy_hydra_run_to_dataset_log(self):
        self.namer.dataset_logs_path.mkdir(parents=True, exist_ok=True)
        self._copy_hydra_run_to_log(self.namer.dataset_logs_path)

    def copy_hydra_run_to_stage_log(self, stage, top_level_working):
        if stage == "Simulation":
            stage_path = self.namer.stage_logs_path(stage, top_level_working=top_level_working)
        else:
            stage_path = self.namer.stage_logs_path(stage)
        stage_path.mkdir(parents=True, exist_ok=True)
        self._copy_hydra_run_to_log(stage_path)

    def _copy_hydra_run_to_log(self, target_root):
        for item in self.namer.hydra_path.iterdir():
            destination = target_root / item.name
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination)


class LogsNamer:
    def __init__(self, 
                 cfg: DictConfig,
                 hydra_config: HydraConfig) -> None:
        logger.debug(f"Running {__name__} in {__file__}")
        self.hydra_run_root = Path(hydra_config.runtime.cwd)
        self.hydra_run_dir = hydra_config.run.dir
        self.scripts_subdir = cfg.file_system.subdir_for_log_scripts
        self.dataset_template_str = cfg.file_system.log_dataset_template_str
        self.stage_template_str = cfg.file_system.log_stage_template_str
        self.top_level_work_template_str = cfg.file_system.top_level_work_template_str
        self.namer = Namer(cfg)

    @property
    def hydra_path(self) -> Path:
        hydra_cfg = HydraConfig.get()
        from hydra.types import RunMode
        if hydra_cfg.mode == RunMode.MULTIRUN:
            return self.hydra_run_root / hydra_cfg.sweep.dir / str(hydra_cfg.job.num)
        else:
            return self.hydra_run_root / self.hydra_run_dir

    @property
    def hydra_scripts_path(self) -> Path:
        return self.hydra_path / self.scripts_subdir

    @property
    def dataset_logs_path(self) -> Path:
        with self.namer.set_context("hydra_run_dir", self.hydra_run_dir):
            path = self.namer.path(self.dataset_template_str)
        return path

    def stage_logs_path(self, stage_dir, top_level_working: bool = False) -> Path:
        use_template = self.stage_template_str
        if top_level_working:
            use_template = self.top_level_work_template_str
        with self.namer.set_contexts({"hydra_run_dir": self.hydra_run_dir,
                                      "stage": stage_dir}):
            path = self.namer.path(use_template)
        return path