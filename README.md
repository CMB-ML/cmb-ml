# Welcome to the Dump Sink

I'm combining repositories. Here's the combined READMES, which may contain out-of-date (or straight up wrong) information.

Contents:
  - [Simulations](#simulations-readme-ml_cmb_pysm_sims)
  - [CMBNNCS](#cmbnncs)
  - [PyILC](#ml_cmb_model_nilc)
  - [Petroff](#petroff)
  - Todos
    - [Cleaning Up Todos](#cleaning-up-to-dos)

# Simulations Readme ml_cmb_pysm_sims
Rough development repository for PySM simulations for ML project

## Installation

The installation process is generally:
- Set up your local file system locations
- Get the repositories
- Set up your python environment
- Get file assets

### File System Considerations

I suggest using the following directory structure for this local project in your home folder. This keeps data assets separate from the created datasets and from the code used to create it.

```
CMB_Project/
│
├── ml_cmb_pysm_sims/                  ## This repository for ML-driven CMB simulations with PySM
│
├─ SourceDataAssets/                  ## Raw data from various sources
│ ├── Planck/                        ## Data from the Planck mission
│ ├── WMAP_Chains/                   ## Data from the WMAP mission
│ └── ProcessedData/                 ## Intermediate data processed for simulation use
│
└─ DatasetsRoot/                      ## Root directory for datasets generated
  └── [Other Dataset Folders]        ## Other directories for organized datasets
```

Clearly systems vary. Configuration files may be changed to describe your local structure. Create your own configuration file by copying one included (after getting the repo, in conf/local_system).

If you're regularly pulling from the repo, add `export CMB_SIMS_LOCAL_SYSTEM=your_system` to the end of your `.bashrc` file and then either `source ~/.bashrc` or restart your terminal. If using a Mac, use `.zshrc` instead of `.bashrc`. If using WSL TODO: figure this out.

If you won't be actively pulling from the repo, simply change all top-level configurations, e.g. config.yaml, to `defaults: - local_system: your_system` where `your_system` is the filename of your configuration.

### Get Repository

- Clone the repositories into the directories as set up above.
  - Either (git with HTTPS)
  - `git clone https://github.com/JamesAmato/ml_cmb_pysm_sims.git`
  - Or (git with SSH):
  - `git clone git@github.com:JamesAmato/ml_cmb_pysm_sims.git`

### Library Set Up

- Ensure you have python 3.9
  - If you have no Python 3.9 installation in your PYTHON_PATH, a conda environment with 3.9 can be used
  - Create a fresh 3.9 environment: `conda create -n py39 python=3.9`
  - Activate the environment: `conda activate py39`
  - Find the path to python, needed for Poetry: `which python`
  - Importantly, deactivate the conda environment, otherwise Poetry will manage the active environment: `conda deactivate`

- Install Poetry. Instructions are here: [https://python-poetry.org/docs/](Poetry Installation)

- Navigate to the folder containing `pyproject.toml`

- Use Poetry to set up the virtual environment
  - If you have no Python 3.9 installation in your PYTHON_PATH
    - Set the poetry env to point to your python installation (found as per above): `poetry env use /path/to/conda/envs/your-env/bin/python3.9`
  - Initialize your environment: `poetry install`
  - Verify your installation: `poetry env info`

- If working with VS Code
  - You can choose the Poetry environment as the interpretter, as usual
  - Setting up to debug may require making a vscode launch.json

- Get needed files (see below)
  - Needed if running simulations
<!-- - Install healpy using conda (consider skipping this at first; it may only be needed on a Mac; unknown currently.)
    - At least on a mac (maybe? apparently?), healpy doesn't like the pip installer but is ok with the conda installer. Maybe. I'm not sure the correct process; some combination of pip and conda installs and uninstalls of both healpy and pysm3 got it working.
    - Official healpy documentation says the following, but this adds conda-forge to your channels permanently:
    - `conda config --add channels conda-forge` (Don't do this)
    - `conda install healpy` (Don't do this)
    - Instead, I suggest `conda install -c conda-forge healpy` which makes no system-wide changes. -->

<!-- - Try to install all of pysm3 with pip
     - Within the repo, install using `pip install .`
     - That may fail for healpy, but install everything else except pysm3 itself (not the case in Ubuntu docker ?)
     - Then do `pip install --no-deps .` -->

<!-- - Still missing numba and toml
    - Run `conda install numba toml tqdm`
    - Maybe this should go earlier? -->

<!-- - Get the Needed files (see next section) -->

<!-- - Install hydra
    - pip install omegaconf
    - pip install hydra-core --upgrade -->

<!-- - Install CAMB
    - pip install camb -->

### Needed files

Needed files are stored on Markov, in `/bigdata/cmb_project/data/assets/`. This is the fastest way to get them, if you have access.

"SourceDataAssets/WMAP_Chains/" files are used to create the CMB power spectrum. They can be downloaded from [Chain Files Direct Link](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_wmap9_chains_v5.tar.gz), as listed at the [NASA WMAP page](https://lambda.gsfc.nasa.gov/product/wmap/dr5/params/lcdm_wmap9.html).

Different chains are available, adding the parameter `mnu`. They can be downloaded from [Chain Files Direct Link](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_mnu_wmap9_chains_v5.tar.gz), as listed at the [NASA WMAP page](https://lambda.gsfc.nasa.gov/product/wmap/dr5/params/lcdm_mnu_wmap9.html). Note that changes may need to be made in your `local_system` config yaml and your `simulation/cmb` yaml.

"SourceDataAssets/Planck/" files are needed for noise generation. 

There are three ways to get the planck_assets maps. The fastest is from Markov. I found that the ESA Planck page is slower than CalTech, but there could be many reasons for that.

Option 2: Use either get_planck_maps_caltech.sh or get_and_symlink_planck_maps to get the observation maps (the latter is suggested; it was created because I store huge files outside the repository for use with other code). Option 3: From [ESA Planck Page](https://pla.esac.esa.int/##results), choose Maps, then Advanced Search, using the terms "LFI_SkyMap_%-BPassCorrected_1024_R3.00_full.fits" and "HFI_SkyMap_%_2048_R3.01_full.fits" (note that the 353 should be with "-psb", as the version without that has some issue mentioned in the Planck Wiki).

These maps are needed:
- Observation map files:
  - planck_assets/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits
  - planck_assets/LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits
  - planck_assets/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits
  - planck_assets/HFI_SkyMap_100_2048_R3.01_full.fits
  - planck_assets/HFI_SkyMap_143_2048_R3.01_full.fits
  - planck_assets/HFI_SkyMap_217_2048_R3.01_full.fits
  - planck_assets/HFI_SkyMap_353-psb_2048_R3.01_full.fits
  - planck_assets/HFI_SkyMap_545_2048_R3.01_full.fits
  - planck_assets/HFI_SkyMap_857_2048_R3.01_full.fits

## Code Organization

Look at tutorial.ipynb (this shows the creation of many different maps, replacing the dev_## files that we had before).

The file `make_dataset.py` is what would generally be considered the main entrypoint into the software.


I've tried to separate code based on the developer working on it.

These are some general functionalities of the python files:
- Components: Contain classes that manage components of the simulation
- Namers: Determine file system names
- Physics: Contains physics-business logic
- Make: Makes output data assets
- Try: Scripts that try something out (test code, but not used with formal testing framework)

Generally, the kind of class you'd see in those files would be what you expect. There are also:
- Factory: These produce objects and are set up the same way regardless of data split or sim number
- Seed Factory: These produce seeds to be used for components instantiation.
  - Sim Level - most components are initialized within PySM3, which handles T, Q, and U fields at once
  - Field Level - noise components are not handled by PySM3, so different seeds are needed for each field (in hindsight, this is easily fixed...)

I need a better term for components, the lines are blurry there. The CMB component currently contains a lot of physics logic and should get untangled. 

I tried to remove filename tracking from anything that isn't a Namer. However, especially in the case of the namer_dataset_output Namers, there's a lot of management code shoehorned in that needs to be cleaned out. For instance, an outside management class should keep track of current split and simulation. File IO should also be handled elsewhere. A similar system should be used for the WMAP_chains accessor. And the Seed tracking stuff should be incorporated into the management class... and have a different filename.

# To Do

Big:
- CMBNNCS
- Petroff
- PyILC
- Analysis

## Adding CMBNNCS

  - [x] Trying to run at 512 on Markov
    - [x] Need final_infer in pipeline
    - [x] Need to look at result of 120th epoch (cfg file stopped at epoch 40)
    - [x] Resolve issues in appearance (map figure) due to scaling - stray pixels at very high or low values
    - [x] Run for all test set
    - [x] PS is really slow - since it's junk anyways for now; downgrading level of effort there (produce only prediction spectra, not cross-spectra or difference spectra)
  - [ ] Get power spectra; send to Ammar
  - [ ] `final_infer` in pipeline configured?
  - [ ] Look at results at 512 at epoch 120 (after analysis done at next step)

## Adding ILC
  - [x] ILC - Run at 512 on Markov
    - [x] Look at result, describe - just has some high values affecting scales; forcing scale make it look ok
    - [ ] Why high values?
  - [ ] What output precision for ILC Maps? (float or double) How many fields on IQU maps?
  - [ ] Get power spectra for HILC results
  - [ ] Look at results at 512 at epoch 120 (after analysis done at next step)

## Adding Petroff
  - [ ] What output precision? Float
  - [ ] Run at 32 - not good
  - [ ] Collect results, with descriptions (as possible)
  - [ ] Clear out old runs
  - [ ] Run at 128
    - [ ] Find magic model
      - [ ] Why did it work? Would it work on simulation with PS? Without PS? With current code?
  - [ ] 128 Transforms failed
  - Options to consider:
    - L1 Loss
    - No initialization
    - Other regularization
    - ConcreteDropout
    - ConcreteDropout + Uncertainty estimates + Variance regularization

## Analysis
  - [ ] Add better PS One-off generator
  - [ ] Add bulk PS statistics
  - [ ] Add bulk PS figure

## General

- [ ] Simulation Working Dir: Fix it
- [x] Clean up import structure 
    - [x] look for "...core"
    - [ ] put core asset handlers into a single import
    - [ ] get HealpyMap out of core or others into it
- [ ] Where reasonable, load from configs in the `__init__` statement of Executors, for better compatibility with prerun

Polarization:
  - [ ] Decide if CMB simulation maps should be saved with only a single field.
    - Later comparisons would assume the same fields when comparing maps, instead of checking for each

Other:
  - [ ] Logger line with expected output directories at asset creation?
  - [ ] Simulation logs going into working directory
  - [ ] Analysis: Run ANAFAST on maps after conversion to double precision
  - [ ] Separate folders for Simulation / Working / Analysis
    - [x] Partial - Working exists; Analysis stuff for each model is going into model's working.
  - [x] Try Wang resume training
  - [x] Configure PyILC with pipeline
  - [x] Find Petroff actual preprocess method
  - [ ] Inventory other repos for their one-off scripts
  - [ ] Try PyTorch Transform style instead of bulk pre/post-process
    - [x] Works for Petroff
    - [ ] Does not work for Wang (unsure why; can't use num_workers)
  - [ ] Base Executor for Parallel?
  - [ ] Find why Petroff failed
  - [ ] Base Executor for PyTorch, generic for other models; Petroff and CMBNNCS are very similar... (?)
    - [ ] Same for Train
    - [ ] Same for Predict maybe???
  - [ ] Support for multiple HDU's with every map?
  - [ ] Ensure units make sense everywhere
  - [ ] Change src to cmml (package name)
  - [ ] Change loggers from `__name__ ` to shorter names
  - [ ] Enable those names in the console log, check if it looks better
  - [x] Train TQDM should update loss inline
  - [ ] Namer set_contexts should take kwargs instead of a dictionary
  - [ ] Rethink dataset creation (currently within each split using template) [I don't remember what this is]
  - [x] Easier scenario overrides for, e.g., fewer detectors (do other use cases exist? If not, just... do it; no need for a fancy solution.)
  - [x] Easy way to run multiple things on the same simulation dataset into different folders (just document HOW, no need for a fancy interface)
  - [ ] Hydra Checkers have option to collect all issues and then error, or error immediately. (Move to later list)


## Documentation

  - [ ] Citations/references for CMBNNCS, Petroff, PyILC, PySM3, CMB Summer Camp
  - [ ] In what way does CMBNNCS fail when 545, 857 detectors included?

## To do (Make Simulations)

- [ ] Use logger
- [ ] Make instrumentation noise as a PySM3 Model (?)
- [ ] Convert CMB targets to K_CMB (?)

Markov:
- [ ] Mount astropy outside Docker
- [ ] Write shell script for timing tests
  - Or just something to scrape the timings from the logs

## Analysis

- [ ] Get rid of Mover AssetHandlers (?)
- [ ] Implement better power spectra object
- [ ] Clean up PS Plotting code to match some of 5-11 discussion

## UG Ready

- [ ] Where not enough variation exists (read: same thing), use PySM component_objects interface instead of preset_strings 
- [ ] Make tests
- [ ] Review: use or remove configuration items
- [ ] Make a script to get Planck Assets from ESA
- [ ] Time how long each preset string adds
- [ ] For each preset string, at each freq, in each field, what is the max and minimum value? What is the median, Q1, Q3, and sd?
- [ ] Are the CalTech maps the same as the ESA maps? Just need to load the maps and calculate the difference.
- [ ] Ensure variation between: sims, components, fields, noise
- [ ] Fiducial & Derived PS on a single plot in inspect_simulations.py
- [ ] Check if hydra sweeps work with the current logging setup (probably not, in hindsight)

## Cleaning Up To-Do's

- Analysis Compares
  - [ ] Detectors in config file is potentially confusing, but hydraconfigs checker looks for it; make solution? Or replace configchecker

- Configs
  - [x] In pipeline / local system: establish better {src_root} system
  - [x] Clean up local_system: use interpolation for directories?
  - [ ] Side load defaults for commonly changed parameters (instead of scenario even?)
  - [ ] Look for make_dirs or make_cache references; remove them [What did this mean?]

- Core
  - [ ] BaseExecutor/Asset: Remove ban on "fn" in pipeline configs.

- Rename analysis stages and directories
- Rename "ps_fidu_fixed" to "fixed_theory_ps" or "ps_theory_fixed" or "ps_thry_fixed"
- Decide on using "theory" or "thry"

## More todos from a text file... need to organize and clean these out

Later:
    - [ ] Confirm simulations are repeatable
    - Use CSV handler for px analysis
    - [ ] Logging: "Output to ____ for X asset"
    - [x] Cross-check if Wang prediction dropoff matches simulation nside_max (Result: It's from applying the beam; we set an lmax there.)
    - [x] Script backup needs to include 
        - [x] library versions
        - [x] config files (and hydra condensed config)
        - [x] copied into per directory, not just top level
    - [x] Logs into stages
    - Root out all hardcoding of polarization fields
    - Validation:
        - Flag for bailing out?
        - Decreasing loss?
    - [x] Remove 'experiment' from executors. Just pass the config around.
    - [x] Better use of 'ExperimentParameters' (Removed instead)

Maybe:
    - [ ]TESTs (note: check PySM3)
    - [x] Early check of configs for conflicts/issues (don't fail late because of mismatched config settings)
    - Power spectrum object outlined below
    - [x] Instrument object outlined below
    - Optional instantiation of portions of Hydra Configs?
    - [x] Preprocess(A, making params): Can use same structure as px_analysis to generate a report/summary (Done: "Parallelized")
    - [x] Parallelize some stages
    - Decide and clean up: No cfg outside of class initializations (for Executors, in some I use self.cfg.whatever... or I can define that in __init__ for self.whatever) ?
    - Encourage one-offs w/ the same code calls and less boilerplate (API related)
    - [x] Temporary pipeline addition for "view"ing trials
    - How to API this thing??
        - Easy to use DataLoader
        - Easy to use Analysis
    - TQDM
        - [ ] TQDM flag
        - [x] add TQDM iterator sometimes; 
        - [ ] add levels (?)
    - [x] Generic ManyMaps (Removed ManyX handlers entirely. Much cleaner)
    - [ ] HD5 output instead of FITS
    - [ ] Check map file units; ensure compliance in AssetHandler reader/writer
    - [ ] File cleanup flag for user to reclaim disk space (per asset? how?)
    - [ ] Each contaminant we want in its own stage
    - [ ] Each precursor values with prng values in its own stage
    - [x] Simulation splits separate from training splits +1: filter to ensure training is subset of simulation (and same for analysis) (I think this is referring to a potential conflict in the config files. Something along the lines of sim_num 5 required for stage C, but not stage B. Similar is which epochs are saved during training, used for prediction, used for post, use for analysis)
        - Done: Overrides are added for configs
    - [ ] CMB-like spectrum generator based on... ??? (Hu?)
    - [x] AssetHandlers broken into multiple files?
    - [ ] Enable Multirun

- Physics questions:
    - When running lower resolution, how do we pick parameters?
    - Analysis Lmax?
    - NILC specifics: 
        - GN_FWHM_arcmin
        - Ell_peaks (CN)
        - Taper width
        - EllMax
        - Perform_ILC_at_Beam
    - Sims processing is correct?
        - Ringing in 70 GHz detector
        - CMB "realization" has a lot of texture (does it need preprocessing to be comparable?)
    - How to mask?
    - Other questions from backlog in Google Doc

- Petroff
    - Run on bad units only
    - Add validation to ensure progress
    - Do units matter for Petroff? Is K_CMB <--> K_RJ linear? No.
    Models trained on (working): 128, non-converted units
    Models trained on (failed) : 32, 128, 512 converted units

- Add to codebase
    - [ ] Object to track PS
        - Contains powerspectrum; is_Cl (bool); ellmin, ellmax (ints)
    - [x] Object to track instrumentation
        - Handle janky if detector is 545/847 for polarization
        - Note what this impacts; handle all instances of it
            - Asset handlers (Many maps)
        - Related to Root out hardcoding of polarization fields

## Stretch

  - [ ] When starting a stage, snapshot non-log files in that directory. If everything is over-written, remove old logs as well.

# Credit / References

- [CAMB for Python](https://camb.readthedocs.io/en/latest/CAMBdemo.html)
- [CMBNNCS Library](https://github.com/Guo-Jian-Wang/cmbnncs/tree/master)
- [DeepSphere for Pytorch](https://github.com/deepsphere/deepsphere-pytorch/tree/master)

# Notes for Documentation

## Common Errors

- When an asset Exception has occurred: TypeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
write() takes 1 positional argument but 2 were given
  - This usually means that I've forgotten `whatever.\[read/write](data=)` (or `model=`)

## Structure
    
Conventions I'd love to have stuck with:
- When referring to detectors
  - adding an "s" is the plural of the following
  - "freq" is an integer. "frq" and "frequency" are not used.
  - "detector" is the object. "det" is used in lower level functions for brevity.
- When dealing with files
  - Use full words "source" and "destination"
  - Use abbreviation "dir" for "directory"

## Views

- It's interesting to look at radio galaxies alone (no cmb, no other contaminants) at resolution 128 and 512; ringing
  - plot_rot=(280, -60) for low frequencies (30 GHz)
  - plot_rot=(250, -50) for high frequencies (857 GHz) to see the impact of ringing (due to detector fwhm?)
  - better coordinates needed; levels seem to vary greatly by rotation

# CMBNNCS

This is very much a work in progress.

python main hydra.verbose=false
pyreverse -o png -p ml_cmb_model_wang src 

## Design decisions

- Not sure how Wang was using validation data; there's no records in the code that illustrate how they do so.
- Considering adding a Validation set in adding to Train, TextX sets. Currently, I've got 1250 in the Train set, of which 250 are set aside.
- Updating learning rate per batch, not per epoch (matching Wang)
- Using PyTorch's LambdaLR instead of Wang's LearningRateDecay class
- Not using PyTorch's ExponentialLR (which uses a fixed gamma, and thus an unknown final LR value)
- Using epoch system with full training set exposure
- Not sure about Wang's triple exposure method.

- Analysis
    - Per epoch ok
    - A single monolithic report is produced across all epochs
    - Output images are into directories per epoch

## Notes about Wang's code during review

- Requires Python 3.9 (currently)
    - In sequence.py, an import of "from collections import Iterable" causes failure
    - Instead, "from collections.abc import Iterable" should be used.
    - TO DO: pull request when closer to publication:
        - something like `try: from collections import Iterable, except: from collections.abc import Iterable`

- Apparent workflow:
    - Simulations:
        - Each sim_X.py (for each component)
            - Note that pixel order is mangled at this point
        - add_tot_full_beamNoise
        - add_X_beam

        - preprocessing (normalization) happens at training time
            - random_arrange_fullMap & random_arrange_fullMap_CMBS4 calls transform (normalize?) IF THE FLAG `normed` is True
            - test_cmb_full, which calls random_arrange_fullMap, has normed=False
            - No normalization was used in the example files. The paper is unclear, but it seems they use the blanket value 5 as a normalization factor.
                - This is good for running time... I don't have to scan for min / max values.
    - Modelling (examples folder):
        - train_cmb_unet_X
        - test_cmb_X

- It seems that they normalize by fixed max values depending on the contaminants used (examples/loader.py)
    - Instead of more typical, min-max or standardization

- In train_cmb_unet_full.py
    - They've created their own dataloader instead of using PyTorch's
        - "loader.random_arrange_fullMap" returns a batch of maps
    - xx are input features, yy are target labels
    - They repeat_n, loading a batch of training and validation data, and repeating it 3 times (???)
    - To pick which maps are loaded, they seem to use np.random.choice, pulling from the same bank of options each iteration
        - Not the usual epoch training

- in loader.py
    - Similar code is used for component maps and total maps
    - load_oneFullMap and load_oneFullMap_2.... are similar in that they get maps from files
        - The first handles components and the spectral indices
        - The second handles full sky simulations

- cmbnncs
    - data_process.py: simple utils for np <--> torch
    - element.py: simple torch nn elements (homebrew functools "partial")
    - loadFile.py: simple file namer 
    - optimize.py: sets learning rate for iterations
    - sequence.py: defines layers of the network
    - simulator.py: 
        - makes power spectrum using draws of cosmo params based on mean and variance
        - Spectra class: Name, Cls (maybe is 2D array of ells/Cl^TT/Cl^? ?), isCl/isDl
        - Components class: nside, parameters for spectral variation
        - X Components: (one subclass for each type of component, making map for each)
        - readCl from X: generating power spectrum using CAMB or PyCAMB
        - sim X: make a map for each component
        - Unsure if output of this is Mangled
    - (!) spherical.py
        - Cut: class containing methods for chopping / unchopping map into 12 K pixels
            - parameter `subblocks_nums` is K; can subdivide into more than 12 pieces. Doesn't seem to be used?
        - piecePlanes2spheres: SkyMap -> MangledMap
        - Others seem to be used for figures only
    - (!) unet.py
        - Defines the network based on sequence.py
        - UNet5
        - UNet8
    - utils.py
        - mkdir, rmdir: obv
        - saveX: X \in {dat, npy, txt}, saves files
        - Logger: simple custom Logger implementation
- Examples
    - add_gaussian_beam.py & add_planck_beam.py
        - For CMB_S4 detectors (X_gaussian_X) and planck (X_planck_X)
        - Basically the same file
        - Loads a map, demangles it, applies a gaussian beam, remangles it, saves it
    - add_tot_full_beamNoise_CMB-s4.py & add_tot_full_beamNoise.py
        - For CMB_S4 detectors (X-s4) and planck (X)
        - Basically the same file
        - Algo:
            - output = 0
            - For all X in {*components, instrumentation noise}
                - Loads X, add to output
            - Save file
        - Planck gets noise draws from Planck's noise maker
        - Each calls its own load_oneFullMap
    - get_block_map_CMB-s4.py
        - Gets a single block and saves it to new file
    - loader.py
        - Methods for loading maps/beams/batches
        - "transform" = quasi-normalization
            - I can't tell if this is reasonable... it seems like they may be normalizing each component separately?
    - plot_cmb_block_CMB_S4_EE_BB.py
        - Loads model, features (`tot`), labels (`cmb`)
        - Gets predicted label (`cmb_ml`)
        - Demangles `tot`, `cmb`, `cmb_ml`
        - Plots them
    - plotter.py
        - computing power spectra
        - make matplotlib plots
        - masking
    - sim_X.py
        - X \in {ame, cmb, dust, free, sync}
        - uses cmbnncs/simulator.py to make component
        - seeds are 50,000 (why this ##?) draws from np.random.choice
        - gets I, Q, U maps for each component
        - mangles component map
        - saves it
    - sim_noise_CMB-S4.py
        - same as sim_X.py, but only noise for CMB-S4
        - Note that noise for Planck is from Planck supplied maps
    - test_cmb_block_CMB-S4.py, test_cmb_full.py
        - basically, same as plot_cmb_block_CMB_s4_EE_BB.py ?
    - train_cmb_unet_block_CMB-S4.py, train_cmb_unet_full.py
        - See above

<!-- ## Set up Environment

Need:
    - PyTorch
    - hydra
    - pytest
    - numpy

To install on a new system... (dev... main branch should be a different procedure)
    - Get poetry
    - created conda environment with python 3.9 (Why 3.9? I don't know. Defunct memories of non-existent dependencies.)
      - `conda create -n ml_cmb_model_wang python=3.9`
    - activate it
    - `which python` -->
<>
    <!-- - install PyTorch according to PyTorch's instructions (https://pytorch.org/get-started/locally/)
    - e.g.: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` -->
    <!-- - Have Poetry set up everything else, based on the `pyproject.toml`
    - `poetry install` -->

## Set up on Markov...

tmux:

connect to markov:

connect to container:
- `ssh -p 31324 jim@localhost`
run python script:
- `cd /shared/code/ml_cmb_model_wang`
- `python main.py`

Set-up
- Build docker container using Dockerfile
- `docker exec -it cmb_wang /bin/bash`
- `conda create -n cmb_wang python=3.9`
- `conda init`
    - If (base) does not appear in the command line
- `source /home/jim/.bashrc`
- `conda activate cmb_wang`
<!-- - `conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia` -->
<!-- - `cd /shared/code/ml_cmb_model_wang` -->
<!-- - `poetry install` -->
    <!-- - delete any poetry lock files first! -->
- Ensure Environment variable is set
    - check with `echo $CMB_SIMS_LOCAL_SYSTEM`
    - command to set it is... ?


## Goal (Relics from wang's README's below)

Train and run Wang's model. Produce output that can be analyzed the exact same way as other models.

The structure is such that the configurations for simulation and other models do not need to repeat information. Ideally, anyway. We'll see if we get that.

The steps are:

- Determine preprocessing parameters
    - Input: Raw maps for Train split
    - Output: min and max values per channel/field tuple
- Preprocess maps 
    - Input: Raw maps for all splits
    - Output: Min-Max scaled maps such that values are in [0,1]
- Train wang's model (output )
    - Input: Preprocessed maps for Train split
    - Output: Models - at checkpoints and final
- Inference using trained model
    - Input: Final model, Preprocessed maps for all Test splits (optionally including the Train split)
    - Output: Prediction maps for all Test splits (still scaled)
- Postprocess
    - Input: Prediction maps for all Test splits (scaled)
    - Output: 
        - Prediction maps for all Test splits, unscaled
        - Power spectra (with? without? beam convolution)

## Consider

- When defining splits, 
  - We expect the pipeline yaml to have
    - Names that match the names in stage executor classes (eventually, replace this with hydra object instantiation)
    - For each pipeline process, a splits: [list] where the list has "kinds" of splits
  - We expect the splits yaml to have
    - Splits which are named to match the splits used in the pipeline yaml
    - the matching ignores capitalization 
    - the name should be [whatever kind] followed by digits

- Assets:
    - Input assets have a stage_context which can vary depending on what stage produced the asset. For instance, a normalization file for preprocessing is part of the preprocessing pipe. A set of preprocessed maps were made in the preprocessing pipe. This informs the namers.
    - Output assets have a stage_context which is always the current stage (current Executor).

## To Do Eventually
- Revisit
- Tests
  - For each asset_in and asset_out in wang.yaml, ensure the Asset can be produced
  - Consider different combinations of ## fields for simulation and modelling
    - If simulation has 3 fields, modelling has 3 fields, may have 857,545; these do not have polarization information
    - If simulation has 3 fields, modelling has 1 field; do not load extra fields just because they exist
    - If simulation has 1 field, modelling has 3 fields; should fail to run
    - If simulation has 1 field, modelling has 1 field; should run trivially


# ml_cmb_model_nilc

This is very much a work in progress.

## Set up Environment

Installation method to be revised. The `pyilc` repository is not set up as a library for use with other code. For now, we play a weird game, minimizing changes to it.


    - Get pyilc; put it in a separate directory
    - `git clone https://github.com/jcolinhill/pyilc.git`
    - Get this repository
    - `git clone https://github.com/JamesAmato/ml_cmb_model_nilc.git`
    - Switch to vary_detector_check branch
    - Ensure that cfg information matches your local system
      - Check the /cfg/local_system directory for normal changes
      - Ensure that you've set an environment variable to match the name of the local_system yaml
        - `export CMB_SIMS_LOCAL_SYSTEM=markov`
      - In /src/pyilc/__init__.py, change the directory here to your pyilc installation folder
      - (Later) In .vscode, match your launch.json to launch.json
    - Get poetry
    - created conda environment with python 3.9
      - `conda create -n ml_cmb_model_nilc python=3.9`
    - activate it
    - Use `which python` to find the python location; update your `launch.json`.
    - Have Poetry set up everything, based on the `pyproject.toml`
    - `poetry install`


## Test pyilc

- Create conda environment (`conda create --name pyilc_test python=3.8`)
- Activate it (`conda activate pyilc_test`)
- clone repo into folder for pyilc (`git clone https://github.com/jcolinhill/pyilc.git`)
- Install unknown libraries (oops, no record of which... probably just simple ones)
- Install jupyter notebook into conda environment for test (`conda install -c conda-forge notebook`)
- Install h5py into conda environment for test (`conda install -c conda-forge h5py`)
- Install matplotlib and healpy into conda environment for test (`conda install -c conda-forge matplotlib healpy`)
- Run Planck_CMB_HILC.ipynb first.
- Add the following to `sample_run_Planck_CMB_HILC.yml`:
  -`ILC_bias_tol: 0.01`
  -`save_as: 'hdf5'`
- The cell containing three calls to `run_full_ILC()` takes ~6 minutes to run.
- Two cells lower, the cell starting with `def download_preprocessing_mask()`, uncomment the call to that function.
- Issues were found at this point; the wavelets.py file sets some features for matplotlib which cause them to fail (lines 67-71).
- Other work... went sideways.

## Commands to remember

pyreverse -o png -p ml_cmb_model_nilc src 
python main hydra.verbose=false

# Petroff

 - Original used these components: synch, thermal dust, free-free, ame, co, sz for clusters(?), CIB
   - Note: No point sources
 - Original used all HFI detectors and 70 GHz LFI
 - Petroff's original absmax normalization (from Petroff Source/simulations/outputnops512_dm8/normalization.npz, based on 512nside)
    - CMB: 631.7
    - 70:  10.3749
    - 100: 0.05426
    - 143: 0.06272
    - 217: 0.2349
    - 353: 1.713
    - 545: 31.30
    - 857: 2605