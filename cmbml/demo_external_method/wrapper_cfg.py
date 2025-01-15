import subprocess
import os


def run_configed_script(config_fn):
    """Run the simulation script with the specified config file."""
    result = subprocess.run(
        ["python", "do_the_thing.py", config_fn],
        capture_output=True,
        text=True
    )
    print(result.stdout)  # Print the output from the script
    print(result.stderr)  # Print any error messages
    return result.returncode
