# import pkg_resources
from importlib.resources import files
from importlib.metadata import distributions
import shutil
import ast
import json
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

        # Whitelisted first-party packages: archived wherever they live on disk
        # (e.g. cmbml is installed, so it is found via its __file__, not via the
        # script directory). The script's own directory is handled separately and
        # does NOT need to be listed here.
        #
        # Read from config if present, otherwise fall back to a sane default.
        # This keeps existing config files working untouched while leaving a hook
        # to override later.
        try:
            source_dirs = cfg.file_system.source_dirs
            # OmegaConf ListConfig -> plain list of str
            self.source_dirs = [str(s) for s in source_dirs]
        except Exception:
            self.source_dirs = ["cmbml"]

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

    # ------------------------------------------------------------------
    # Python source logging
    # ------------------------------------------------------------------

    def log_py_to_hydra(self, source_script, target_root):
        """
        Collects every first-party Python file reachable by import from
        source_script and archives them into a single ``code.zip`` under
        target_root.

        "First-party" means either:
          1. The file lives under one of the whitelisted package roots
             (self.source_dirs, e.g. cmbml), located via import, OR
          2. The file lives under the directory of the running script
             (e.g. e-d2ps/ or cmb-ml/), which sweeps in the script itself
             plus any local sub-packages it imports (d2ps_nn, d2ps_fcn, ...).

        External libraries (stdlib, numpy, hydra, etc.) are ignored.

        The zip is laid out so it is human-browsable without extraction, with
        each first-party package and the running script appearing at the top
        level, e.g.:

            code.zip
            ├── main_param_d2ps.py
            ├── d2ps_nn/...
            ├── d2ps_fcn/...
            └── cmbml/...
        """
        source_script = Path(source_script).resolve()
        script_dir = source_script.parent

        # Resolve whitelisted package roots (e.g. cmbml -> .../cmbml).
        whitelist_roots = self._resolve_whitelist_roots()

        # The set of roots we will follow/keep imports from. Order matters for
        # archive-name assignment: whitelist roots are checked before the script
        # dir so that a whitelisted package nested inside the script dir (e.g.
        # cmbml living inside cmb-ml/) is attributed to the package, not the
        # script dir. This prevents duplication.
        keep_roots = list(whitelist_roots) + [script_dir]

        py_files = self._trace_imports(
            start_file=source_script,
            start_dir=script_dir,
            keep_roots=keep_roots,
            whitelist_roots=whitelist_roots,
        )

        if not py_files:
            logger.warning("No first-party Python files found to log.")
            return

        self._archive_files(
            py_files=py_files,
            whitelist_roots=whitelist_roots,
            script_dir=script_dir,
            zip_path=target_root / "code.zip",
        )

    def _resolve_whitelist_roots(self) -> list:
        """
        Resolves each whitelisted package name to its on-disk root directory by
        importing it and inspecting __file__. Packages that cannot be imported
        are skipped with a warning.
        """
        roots = []
        for pkg_name in self.source_dirs:
            try:
                module = __import__(pkg_name)
            except ImportError:
                logger.warning(f"Whitelisted package could not be imported: {pkg_name}")
                continue
            pkg_file = getattr(module, "__file__", None)
            if pkg_file is None:
                logger.warning(f"Whitelisted package has no __file__: {pkg_name}")
                continue
            roots.append(Path(pkg_file).parent.resolve())
        return roots

    @staticmethod
    def _trace_imports(start_file: Path,
                       start_dir: Path,
                       keep_roots: list,
                       whitelist_roots: list) -> set:
        """
        Recursively walks imports starting from start_file, collecting every .py
        file whose resolved path is inside any of keep_roots.

        Every visited file is added to ``seen`` to prevent infinite loops. Only
        files inside a keep_root are added to ``collected``. The start_file is
        always walked for its imports; it will also be collected if it lives
        inside a keep_root (which it does, since its own directory is a keep
        root).

        Resolution rules:
          * Absolute imports of a whitelisted package (e.g. ``from cmbml.core
            import X``) are resolved against that package's actual root, not the
            script directory. This is what makes installed first-party packages
            work.
          * Other absolute imports are resolved against start_dir (the script
            directory), which catches local sibling packages (d2ps_nn, ...).
          * Relative imports (``from . import x``, ``from ..y import z``) are
            resolved by walking up from the importing file's directory.

        Args:
            start_file:       The .py file to start tracing from.
            start_dir:        Directory treated as the base for top-level
                              absolute imports that are not whitelisted.
            keep_roots:       Files under any of these directories are collected.
            whitelist_roots:  Roots of whitelisted packages, keyed by their
                              directory name for absolute-import redirection.
        """
        keep_roots = [Path(r).resolve() for r in keep_roots]
        whitelist_roots = [Path(r).resolve() for r in whitelist_roots]
        # Map top-level package name -> its root, for absolute import redirect.
        whitelist_by_name = {r.name: r for r in whitelist_roots}

        collected = set()
        seen = set()
        unresolved = set()

        def _is_kept(path: Path) -> bool:
            rp = path.resolve()
            for root in keep_roots:
                try:
                    rp.relative_to(root)
                    return True
                except ValueError:
                    continue
            return False

        def _get_full_path(module_name: str, current_dir: Path):
            parts = module_name.split(".")
            path = current_dir.joinpath(*parts)
            if path.with_suffix(".py").exists():
                return path.with_suffix(".py")
            if (path / "__init__.py").exists():
                return path / "__init__.py"
            return None

        def _follow_names_as_submodules(resolved: Path, names):
            """
            For a ``from <pkg> import a, b`` statement where <pkg> resolved to a
            package (its __init__.py), each imported name may itself be a
            submodule (e.g. ``from d2ps_nn import helper`` -> helper.py) rather
            than just an attribute. If a name corresponds to a .py file or
            subpackage next to the __init__.py, follow it. Names that are merely
            attributes (functions, classes) simply won't match and are ignored.
            """
            if resolved is None or resolved.name != "__init__.py":
                return
            pkg_dir = resolved.parent
            for alias in names:
                # Star imports and attribute imports won't resolve to files.
                if alias.name == "*":
                    continue
                sub_py = pkg_dir / f"{alias.name}.py"
                sub_pkg = pkg_dir / alias.name / "__init__.py"
                if sub_py.exists():
                    _walk(sub_py, sub_py.parent)
                elif sub_pkg.exists():
                    _walk(sub_pkg, sub_pkg.parent)

        def _resolve_absolute(parts: list):
            """
            Resolve an absolute (level == 0) dotted import to a file path.
            Whitelisted top-level packages are redirected to their real root;
            everything else is resolved against start_dir (the project base).
            Absolute imports never resolve relative to the importing file.
            """
            if not parts:
                return None
            top = parts[0]
            if top in whitelist_by_name:
                # Resolve remaining parts under the package's actual root.
                base = whitelist_by_name[top]
                rest = parts[1:]
            else:
                base = start_dir
                rest = parts
            target = base.joinpath(*rest)
            if target.with_suffix(".py").exists():
                return target.with_suffix(".py")
            if (target / "__init__.py").exists():
                return target / "__init__.py"
            return None

        def _walk(filename: Path, current_dir: Path):
            filename = filename.resolve()
            if filename in seen:
                return
            seen.add(filename)
            if _is_kept(filename):
                collected.add(filename)

            try:
                with filename.open("r", encoding="utf-8") as fh:
                    tree = ast.parse(fh.read(), filename=str(filename))
            except (OSError, SyntaxError) as e:
                logger.warning(f"Could not parse {filename}: {e}")
                return

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    level = node.level
                    if node.module is None:
                        # Bare relative import: "from . import something"
                        mod_path = current_dir
                        for _ in range(level - 1):
                            mod_path = mod_path.parent
                        init = mod_path / "__init__.py"
                        if init.exists():
                            _walk(init, mod_path)
                            _follow_names_as_submodules(init, node.names)
                        continue

                    if level == 0:
                        # Absolute import — always resolved from the project base
                        # (start_dir) or a whitelisted package root, never from
                        # the importing file's own directory.
                        parts = node.module.split(".")
                        resolved = _resolve_absolute(parts)
                        if resolved is not None:
                            _walk(resolved, resolved.parent)
                            _follow_names_as_submodules(resolved, node.names)
                        else:
                            unresolved.add(node.module)
                    else:
                        # Relative import: walk up from current_dir.
                        mod_path = current_dir
                        for _ in range(level - 1):
                            mod_path = mod_path.parent
                        parts = node.module.split(".")
                        target = mod_path.joinpath(*parts)
                        if target.with_suffix(".py").exists():
                            _walk(target.with_suffix(".py"), target.parent)
                        elif (target / "__init__.py").exists():
                            init_t = target / "__init__.py"
                            _walk(init_t, target)
                            _follow_names_as_submodules(init_t, node.names)
                        else:
                            unresolved.add(node.module)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        # Try whitelist/project-base resolution first, then a
                        # local fallback relative to the importing file.
                        resolved = _resolve_absolute(parts)
                        if resolved is None:
                            resolved = _get_full_path(alias.name, current_dir)
                        if resolved is not None and resolved.exists():
                            _walk(resolved, resolved.parent)
                        else:
                            unresolved.add(alias.name)

        _walk(start_file, start_dir)

        if unresolved:
            unresolved_logger = logging.getLogger("unresolved_imports")
            unresolved_logger.info("\n".join(sorted(unresolved)))

        return collected

    @staticmethod
    def _archive_files(py_files: set,
                       whitelist_roots: list,
                       script_dir: Path,
                       zip_path: Path):
        """
        Archives all collected .py files into a single zip at zip_path.

        Each file's name inside the zip is computed so the archive is browsable
        with packages at the top level:
          * Files under a whitelisted package root are stored relative to that
            root's PARENT (giving e.g. ``cmbml/core/log_maker.py``).
          * All other files are stored relative to script_dir (giving e.g.
            ``main_param_d2ps.py`` and ``d2ps_nn/...`` at the top level).

        Whitelist roots are checked first so a package nested inside script_dir
        (e.g. cmbml inside cmb-ml/) is attributed once, to the package, with no
        duplication.
        """
        whitelist_roots = [Path(r).resolve() for r in whitelist_roots]
        script_dir = Path(script_dir).resolve()
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        def _arcname(py_file: Path):
            rp = py_file.resolve()
            for root in whitelist_roots:
                try:
                    rel = rp.relative_to(root)
                    # Store under <pkgname>/<rel> by going relative to root.parent
                    return rp.relative_to(root.parent)
                except ValueError:
                    continue
            try:
                return rp.relative_to(script_dir)
            except ValueError:
                logger.warning(f"File outside all known roots during archiving: {py_file}")
                return Path(rp.name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for py_file in sorted(py_files):
                zf.write(py_file, _arcname(py_file))

    # ------------------------------------------------------------------
    # Config logging
    # ------------------------------------------------------------------

    def log_cfgs_to_hydra(self, target_root):
        relevant_config_files = self.extract_relevant_config_paths()

        with open(target_root / "config_sources.txt", "w") as f:
            for provider, config_files in relevant_config_files.items():
                f.write(f"{provider}\n")
                common = self._find_common_paths(config_files) if config_files else "N/A"
                f.write(f"Common path: {common}\n")
                for config_file in config_files:
                    f.write(f"    {config_file}\n")

        for provider, config_files in relevant_config_files.items():
            if not config_files:
                continue
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
            logger.warning(f"Missing configuration files for: {missing_combinations}")

        for provider, config_files in relevant_files.items():
            for config_path in config_files:
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

    def log_pipeline(self, pipeline) -> None:
        """
        Record the assembled executor plan for this run as pipeline_progress.json.
        Every stage starts as "pending"; PipelineContext updates status as
        stages complete. Fully-qualified names disambiguate the ex.py catch-all.
        """
        target_root = self.namer.hydra_scripts_path
        target_root.mkdir(parents=True, exist_ok=True)
        self._pipeline_log_path = target_root / "pipeline_progress.json"
        self._pipeline_records = [
            {
                "order": i,
                "name": stage.__name__,
                "qualified": f"{stage.__module__}.{stage.__qualname__}",
                "status": "pending",
            }
            for i, stage in enumerate(pipeline)
        ]
        self._write_pipeline_log()

    def _write_pipeline_log(self) -> None:
        """(Re)write the pipeline log to disk. Cheap; called after each update."""
        if getattr(self, "_pipeline_log_path", None) is None:
            return
        with self._pipeline_log_path.open("w") as f:
            json.dump(self._pipeline_records, f, indent=2)

    def mark_stage(self, order: int, status: str) -> None:
        """Update one stage's status and flush to disk immediately."""
        if getattr(self, "_pipeline_records", None) is None:
            return
        self._pipeline_records[order]["status"] = status
        self._write_pipeline_log()


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
