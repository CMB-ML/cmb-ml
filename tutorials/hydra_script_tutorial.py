import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="tut_conf", config_name="sample_cfg")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Accessing the config
    print(f'Attribute style access: {cfg.scenario.nside}')
    print(f'Dictionary style access: {cfg["scenario"]["nside"]}')

    # the compose overrides shown in the notebook files can be done in the terminal when utilizing a script, for example:
    # python hydra_script_tutorial.py scenario=scenario_128 splits=4-2

if __name__ == "__main__":
    main()