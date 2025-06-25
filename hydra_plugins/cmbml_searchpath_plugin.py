from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

class CMBMLSearchPathPlugin(SearchPathPlugin):
    """Hydra plugin to automatically add `cmbml/cfg/` to the config search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="cmbml_configs", 
                           path="pkg://cmbml/cfg", 
                           )
