import hydra
from omegaconf import DictConfig, OmegaConf


# The @hydra.main decorator is used to indicate that this function is the entry point for the script.
# We specify the config_path, which is the directory containing the config files relative to the current module.
# We also specify the config_name, which is the name of the config file without the extension.
# The version_base argument is set to None; it is for Hydra's backwards compatibility.
@hydra.main(version_base=None, config_path="tutorial_configs", config_name="sample_cfg")
# The main function must take a single argument, which is the config object.
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Accessing the config
    print(f'Attribute style access : {cfg.scenario.nside}')
    print(f'Dictionary style access: {cfg["scenario"]["map_fields"]}')
    print(f'Mixed access           : {cfg["scenario"].units}')
    # The following should be printed to console:
    """
    scenario:
    nside: 512
    map_fields: IQU
    precision: float
    units: K_CMB
    splits:
    name: '1450'
    Train:
        n_sims: 1000
    Valid:
        n_sims: 250
    Test:
        n_sims: 200
    preset_strings:
    - d9
    - s4
    - f1

    Attribute style access : 512
    Dictionary style access: IQU
    Mixed access           : K_CMB
    """

    # The compose overrides shown in the notebook files can be done in the terminal when utilizing a script, for example:
    # python hydra_script_tutorial.py scenario=scenario_128 splits=4-2

if __name__ == "__main__":
    main()