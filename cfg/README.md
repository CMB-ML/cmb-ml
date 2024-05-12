# Contents

- [Overview](#overview)
- [Top-Level Configurations](#top-level-configs)
- [Logging Notes](#logging-notes)

# Overview

We attempt to break apart configurations in a meaningful way.

More information on hydra is available at [hydra.cc](https://hydra.cc/docs/intro/).

Hydra configs enable us to better organize our process. We have multiple top-level configs for quickly running debugging tests and for preserving infrequently executed large runs. More detail on the top level configs is below.

We consider the following subdivisions of configuration:
  - *scenario*: Settings that extend across the full process of simulation, inference, and analysis
  - *splits*: Settings of how the dataset is divided up
    - Training, Validation, Test splits are defined here
    - The total number of simulations for each of those subsets
    - Logistical parameters
  - *pipeline*: Settings for each stage of the process pipeline
    - Effort is made to separate this from details of how a model runs
    - Defines the data created by or ingested by a particular stage of the process
    - Subsets of the splits can be defined per stage
  - *model*: Settings for particular models
    - Effort is made to separate this from details of data
    - Includes the simulation model
    - Because analysis may include multiple models, we specify *model/[model name]* in the defaults list
  - *local_system*: File paths particular to your system
  - *file_system*: Any file paths not defined for a particular stage of the pipeline, such as logging file paths

# Top-Level Configurations

Here we define the defaults list, which assembles the portions of the configuration which are needed.

When working on multiple systems and using a remote git repository, it is convenient to define local_system as an environment variable. If you are using a single system, it may be easier to change this to the filename of the yaml defining your local_system.

# Logging Notes

We try to capture hydra logs along with the rest of the source code for reproducability. 



After running a debugging test, ensure that your configs were backed up as expected. We continue to find edge-cases which were not accounted for.

