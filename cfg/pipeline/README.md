# Contents

- [Pipeline Elements](#pipeline-elements)
- [Multi-file Structures](#notes-multi-file-structure)

# Pipeline Elements

- The pipeline is a sequence of stages
- Top-level keys correspond to the name of a stage (string) in the pipeline
  - For each, there is an Executor defined
  - The top-level key should match the Executor's `stage_str`
- Each stage is a dictionary which may have the following common keys:
  - All keys are optional
  - *assets_out*: dictionary of the asset(s) produced by this stage
    - Each asset_out is a key-value pair
      - The key (string) is the name of the asset
        - It will match the python dictionary key for `Executor.assets_out`
      - The value (dictionary) contains the following:
        - *handler*: name (string) of the class that manages this asset
        - *path_template*: file path (string) of file to store the asset; contains elements to be formatted by an executor's name_tracker.
  - *assets_in*: dictionary of assets required by this stage
    - Each asset_in is a key-value pair
      - The key (string) is the name of the asset
        - It will match the python dictionary key for `Executor.assets_in`
      - The value (dictionary) contains the following (for now, one):
        - *stage* (the string "stage"): name (string) of stage that produced the asset
        - The value should probably have already run, but this isn't enforced.
          - e.g. given stages stage_1 through stage_4, 
            - stage_3's assets_in will generally only reference stage_1 or stage_2
            - stage_3 can reference stage_3 (e.g. training a model, which can continue from a checkpoint)
  - *splits*: list of splits to process
    - These should match those in the *splits* yaml
    - Each split is a string
    - These are interpretted as the name portion of the pattern "{name}{number}"
      - Capitalization does not matter
      - listing "test" here will use the splits "Test1" and "Test2"
  - *dir_name* (string): name of the directory to store the assets 
  - *make_stage_log* (bool): If `true`, the logs will be copied to this stage. Should not happen when `false` (note, `False` is different, may be interpretted as a string, and is untested)
- More keys are used later in the pipes:
  - *sim_num*: (list of ints) which simulation nums to process
    - Especially for the purpose of previews
  - *epochs*: (list of ints) which epochs to process
  - *path_template_alt*: (str) Similar to *path_template*; when defined, allows a flag to dictate which template is used.
  - assets_in.*orig_name*: In case a pipeline stage needs to pull output assets with the same name from different stages (useful when comparing assets, such as in when comparing CMBNNCS's preprocessing to the original map in C_show_preprocessed_cmbcnns)

# Notes: Multi-file structure

We break the pipeline into meaningful chunks of things. For instance, we have all elements of the simulation pipeline together in a single simulation_pipe.yaml.

The sample_pipe.yaml puts these chunks together.

Some additional effort was made for logging these "side-loaded" chunks; it may be a brittle implementation in the face of future updates to hydra. Please ensure that logging works appropriately.

It is not necessary to use multi-file structures, it just gets ungainly.

Aliases do not transfer between side-loaded files.