import os, sys
import logging


logger = logging.getLogger("EnvVarCheck")


def validate_environment_variable(env_var_name):
    """
    Check if an environment variable is set and exit early if not set.

    Args:
        env_var_name (str): The name of the environment variable to check.
    """
    # Check if the environment variable is set
    env_var_value = os.getenv(env_var_name)
    if env_var_value is None:
        logger.fatal(f'Error: The environment variable "{env_var_name}" is not set. Please set it to a configuration file name, e.g., "example_pc", located within the cfg/local_system folder.')
        sys.exit(1)

    # If we reach this point, the environment variable is set and valid
    logger.info(f'{env_var_name} is set to {env_var_value}, proceeding with application initialization...')
