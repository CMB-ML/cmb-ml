from typing import Dict, List, Any, Union
import pysm3
import pysm3.units as u
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
import pysm3.data as pysm3_data
import toml
from copy import deepcopy


class FlexSky(pysm3.Sky):
    def __init__(
        self,
        nside: int=None,
        max_nside: int=None,
        preset_strings: List[str]=None,
        component_config: Dict[str, Dict[str, Any]]=None,
        component_objects: List[pysm3.Model]=None,
        component_object_names: List[str]=None,
        output_unit:u.Unit=u.uK_RJ,
        map_dist=None,
    ):
        pysm3.Model.__init__(self, nside=nside, max_nside=max_nside, map_dist=map_dist)

        self.presets = toml.loads(pkg_resources.read_text(pysm3_data, "presets.cfg"))
        self.validate_component_objects(component_objects, component_object_names)
        self.validate_preset_strings(preset_strings)

        # This class will use a comps_dict, so each component can be changed later (without having to reload)
        if component_objects is None:
            self.comps_dict = {}
        else:
            self.comps_dict = {k: v for (k, v) in zip(component_object_names, component_objects)}

        if preset_strings is not None:
            for p_string in preset_strings:
                new_component = self.create_component_from_config(self.presets[p_string])
                self.comps_dict[p_string] = new_component

        if component_config is not None:
            if not isinstance(component_config, dict):
                component_config = toml.load(component_config)
            for k, v in component_config.items():
                new_component = self.create_component_from_config(v)
                self.comps_dict[k] = new_component

        self.output_unit = u.Unit(output_unit)

    @property
    def components(self):
        comps = []
        for v in self.comps_dict.values():
            if isinstance(v, dict):
                comps.extend(v.values())
            else:
                comps.append(v)
        return comps

    def create_component_from_config(self, config) -> Union[pysm3.Model, Dict[str, pysm3.Model]]:
        config = deepcopy(config)
        if "class" in config:
            class_name = config.pop("class")
            model_class = getattr(pysm3.models, class_name)
            return model_class(**config, 
                               nside=self.nside, 
                               map_dist=self.map_dist)
        else:
            sub_components = {}
            for label, subconfig in config.items():
                class_name = subconfig.pop("class")
                model_class = getattr(pysm3.models, class_name)
                sub_components[label] = model_class(**subconfig,
                                                    nside=self.nside,
                                                    map_dist=self.map_dist)
            return sub_components

    def validate_component_objects(self, 
                        # varied_sed_cfg: Dict[str, Any],
                        component_objects: List[pysm3.Model],
                        component_object_names: List[str]
                        ):

        if component_objects is None:
            if component_object_names is not None:
                raise ValueError("component objects are declared but no names are given.")
            # Do no further checks.
            return

        if component_object_names is None:
            raise ValueError("component objects do not have names")

        # Ensure that the component_objects is a list of pysm3 Models
        if not isinstance(component_objects, list):
            raise ValueError(f"component_objects must be a list")
        for component_obj in component_objects:
            if not isinstance(component_obj, pysm3.Model):
                raise ValueError(f"Each object in component_objects must be a PySM3 Model. Got {component_obj}")

        # Ensure that component_object_names is a list of strings
        if not isinstance(component_object_names, list):
            raise ValueError(f"component_object_names must be a list")
        for comp_obj_name in component_object_names:
            if not isinstance(comp_obj_name, str):
                raise ValueError(f"Each component object name must be a string. Got {comp_obj_name}")

        # Ensure there are as many component object names as component objects
        if len(component_object_names) != len(component_objects):
            raise ValueError(f"There should be as many component object names ({len(component_object_names)})" \
                             f" as there are component objects ({len(component_objects)}).")

    def validate_preset_strings(self, preset_strings):
        if preset_strings is None:
            return
        # Ensure preset_strings is a list of strings
        if not isinstance(preset_strings, list):
            raise ValueError("preset_strings must be a list.")
        for preset in preset_strings:
            if not isinstance(preset, str):
                raise ValueError(f"each string in preset strings must be a ... string. Got {preset}")

    def update_component(self, label: str, update: Dict[str, Any]) -> None:
        """
        Update attributes of a top-level component.

        Parameters
        ----------
        label : str
            Key for the component in self.comps_dict, e.g. "d1" or "s2".
            Must not refer to split components like "a1.comp1".
        update : dict
            Dictionary of attribute names and their new values to set on the component.
        """
        if "." in label:
            raise NotImplementedError("Split components (e.g. 'a1.comp1') cannot be modified.")

        try:
            component = self.comps_dict[label]
        except KeyError:
            raise KeyError(f"Component '{label}' not found in comps_dict.")

        for fg_param, settings in update.items():
            if not hasattr(component, fg_param):
                raise AttributeError(
                    f"Component '{label}' ({type(component).__name__}) has no attribute '{fg_param}'"
                )
            value = settings["value"]
            unit = u.Unit(settings.get('unit', ''))
            setattr(component, fg_param, u.Quantity(value, unit))

    def get_component(self, label: str) -> pysm3.Model:
        """
        Retrieve a component or subcomponent by label.

        Parameters
        ----------
        label : str
            Name of the component, e.g., "d1" or "a1.comp1".

        Returns
        -------
        pysm3.Model
            The requested component object.

        Raises
        ------
        KeyError
            If the component or subcomponent does not exist.
        NotImplementedError
            If label contains more than one level of nesting.
        """
        if "." in label:
            parts = label.split(".")
            if len(parts) != 2:
                raise NotImplementedError("Only two-level component access is supported (e.g., 'a1.comp1').")
            outer, inner = parts
            try:
                return self.comps_dict[outer][inner]
            except KeyError:
                raise KeyError(f"No such subcomponent: '{label}'")
        else:
            try:
                return self.comps_dict[label]
            except KeyError:
                raise KeyError(f"No such component: '{label}'")

    def replace_component(self, label: str, new_component) -> None:
        """
        Retrieve a component or subcomponent by label.

        Parameters
        ----------
        label : str
            Name of the component, e.g., "d1" or "a1.comp1".

        Raises
        ------
        KeyError
            If the component or subcomponent does not exist.
        NotImplementedError
            If label contains more than one level of nesting.
        """
        if "." in label:
            parts = label.split(".")
            if len(parts) != 2:
                raise NotImplementedError("Only two-level component access is supported (e.g., 'a1.comp1').")
            outer, inner = parts
            try:
                self.comps_dict[outer][inner] = new_component
            except KeyError:
                raise KeyError(f"No such subcomponent: '{label}'")
        else:
            try:
                self.comps_dict[label] = new_component
            except KeyError:
                raise KeyError(f"No such component: '{label}'")
