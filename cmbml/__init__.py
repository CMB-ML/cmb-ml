__version__ = "0.1.0"

from cmbml.core import PipelineContext

# import os
# import importlib.resources as res
# from hydra.core.config_search_path import ConfigSearchPath
# from hydra.plugins.search_path_plugin import SearchPathPlugin

# print("Test case that cmbml is imported")

# class CMBMLSearchPathPlugin(SearchPathPlugin):
#     """Hydra plugin to automatically add `cmbml/cfg/` to the config search path."""

#     def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
#         config_dir = os.getenv("CMBML_CONFIG_PATH")

#         if config_dir is None:
#             config_dir = str(res.files("cmbml").joinpath("cfg"))

#         print(f"🔍 CMBML Plugin: Adding config path → {config_dir}")

#         # Append the path to Hydra's search path
#         search_path.append(provider="cmbml_configs", path=config_dir, anchor="primary")

# # Register the plugin manually
# def register_cmbml_search_path():
#     from hydra.core.global_hydra import GlobalHydra
#     if not GlobalHydra.instance().is_initialized():
#         GlobalHydra.instance().config_search_path = CMBMLSearchPathPlugin()

# register_cmbml_search_path()
