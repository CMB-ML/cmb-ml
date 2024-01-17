import logging
from omegaconf import DictConfig, OmegaConf
import hydra

logger = logging.getLogger("__main__")

def bigger_indent(yaml_str, new_indent=3):
    new_yaml = []
    for line in yaml_str.split('\n'):
        ct = (len(line) - len(line.lstrip())) // 2
        new_ct = ct * new_indent
        new_yaml.append(f"{'': >{new_ct}}{line.lstrip()}")
    return new_yaml


def log_cfg(cfg: DictConfig, caller_name=None) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    if caller_name is None:
        to_report = bigger_indent(yaml_str)
    else:
        to_report = [f"Printing a config from {caller_name}", *bigger_indent(yaml_str)]
    logger.debug('\n'.join(to_report))


def log_test():
    logger.info("Info level message.")
    logger.debug("Debug level message.")
