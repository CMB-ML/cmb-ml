# ml_cmb_pysm_sims
Rough development repository for PySM simulations for ML project

# Installation

The installation process is generally:
- Set up your local file system locations
- Get the repositories
- Set up your python environment
- Get file assets

Note that the repository depends heavily on PySM3, which currently must be used directly from the current repository.

## File System Considerations

I suggest using the following directory structure for this local project in your home folder. This keeps data assets separate from the created datasets and from the code used to create it.

```
CMB_Project/
│
├── ml_cmb_pysm_sims/                  # This repository for ML-driven CMB simulations with PySM
├── pysm3/                             # PySM3
│
├── SourceDataAssets/                  # Raw data from various sources
│   ├── Planck/                        # Data from the Planck mission
│   ├── WMAP_Chains/                   # Data from the WMAP mission
│   └── ProcessedData/                 # Intermediate data processed for simulation use
│
└── DatasetsRoot/                      # Root directory for datasets generated
    └── [Other Dataset Folders]        # Other directories for organized datasets
```

Clearly systems vary. Configuration files may be changed to describe your local structure. I suggest creating your own configuration file by copying one included (after getting the repo, in conf/local_system).

If you're regularly pulling from the repo, add `export CMB_SIMS_LOCAL_SYSTEM=your_system` to the end of your `.bashrc` file and then either `source ~/.bashrc` or restart your terminal. If using a Mac, use `.zshrc` instead of `.bashrc` (or whatever... Jim is moving on but one day these will be good general instructions).

If you won't be actively pulling from the repo (how I long for that day), simply change all top-level configurations, e.g. config.yaml, to `defaults: - local_system: your_system` where `your_system` is the filename of your configuration.

## Get Repositories

- Clone the repositories into the directories as set up above.
    - Either (git with HTTPS)
    - `git clone https://github.com/JamesAmato/ml_cmb_pysm_sims.git`
    - `git clone https://github.com/galsci/pysm.git`
    - Or (git with SSH):
    - `git clone git@github.com:JamesAmato/ml_cmb_pysm_sims.git`
    - `git clone git@github.com:galsci/pysm.git`

## Library Set Up

- Create and activate a conda environment
    - `conda create --name ml_cmb_pysm_sims python=3.8`
    - `conda activate ml_cmb_pysm_sims`

- Install healpy using conda (consider skipping this at first; it may only be needed on a Mac; unknown currently.)
    - At least on a mac (maybe? apparently?), healpy doesn't like the pip installer but is ok with the conda installer. Maybe. I'm not sure the correct process; some combination of pip and conda installs and uninstalls of both healpy and pysm3 got it working.
    - Official healpy documentation says the following, but this adds conda-forge to your channels permanently:
    - `conda config --add channels conda-forge` (Don't do this)
    - `conda install healpy` (Don't do this)
    - Instead, I suggest `conda install -c conda-forge healpy` which makes no system-wide changes.

- Try to install all of pysm3 with pip
     - Within the repo, install using `pip install .`
     - That may fail for healpy, but install everything else except pysm3 itself (not the case in Ubuntu docker ?)
     - Then do `pip install --no-deps .`

- Still missing numba and toml
    - Run `conda install numba toml tqdm`
    - Maybe this should go earlier?

- Get the Needed files (see next section)

- Install hydra
    - pip install omegaconf
    - pip install hydra-core --upgrade

- Install CAMB
    - pip install camb

## Needed files

Needed files are stored on Markov, in `/bigdata/cmb_project/data/assets/`. This is the fastest way to get them, if you have access.

"SourceDataAssets/WMAP_Chains/" files are used to create the CMB power spectrum. They can be downloaded from [Chain Files Direct Link](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_wmap9_chains_v5.tar.gz), as listed at the [NASA WMAP page](https://lambda.gsfc.nasa.gov/product/wmap/dr5/params/lcdm_wmap9.html).

Different chains are available, adding the parameter `mnu`. They can be downloaded from [Chain Files Direct Link](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_mnu_wmap9_chains_v5.tar.gz), as listed at the [NASA WMAP page](https://lambda.gsfc.nasa.gov/product/wmap/dr5/params/lcdm_mnu_wmap9.html). Note that changes may need to be made in your `local_system` config yaml and your `simulation/cmb` yaml.

"SourceDataAssets/Planck/" files are needed for noise generation. 

There are three ways to get the planck_assets maps. The fastest is from Markov. I found that the ESA Planck page is slower than CalTech, but there could be many reasons for that.

Option 2: Use either get_planck_maps_caltech.sh or get_and_symlink_planck_maps to get the observation maps (the latter is suggested; it was created because I store huge files outside the repository for use with other code). Option 3: From [ESA Planck Page](https://pla.esac.esa.int/#results), choose Maps, then Advanced Search, using the terms "LFI_SkyMap_%-BPassCorrected_1024_R3.00_full.fits" and "HFI_SkyMap_%_2048_R3.01_full.fits" (note that the 353 should be with "-psb", as the version without that has some issue mentioned in the Planck Wiki).

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

# Code Organization

Look at tutorial.ipynb (this shows the creation of many different maps, replacing the dev_# files that we had before).

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

# To do

- [x] Figure out pip and conda installation steps
- [x] Noise, CMB, and all components in a single map (see fixed_map_synth4.py)
- [x] CMB component determined by cosmological parameter draws. (partial, see simple_camb_ps.py)
- [x] CMB component determined by cosmological parameter draws from WMAP 9 chains. 
- [x] Output, per component, default variation (requires 2 runs); compare them (see check_variation_in_base.py)
- [x] Switch to uK_CMB instead of uK_RJ 
  - [x] simple fix: when initializing Sky(), include "output_unit='uK_CMB'"
  - [ ] uglier fix: noise is broken (see fixed_map_synth3.py [not 4] results)
- [x] Organize development scripts for others to follow the trail of work 
- [ ] Make presentation of the above
- [x] Replace the cmb_component.py code that relies on fixed cosmo_params
- [x] Traceability/reproducability (this is a lot of stuff, todo: break down further)
- [x] Move to Markov
- [ ] Run simulations v1
  - [x] Timing & size tests
  - [x] 128T, Full Suite (1250 Training, 200 Test varied ps, 1000 Test [10 ps sets of 100 each])
  - [ ] 512T, Full Suite (1250 Training, 200 Test varied ps, 1000 Test [10 ps sets of 100 each])
- [x] Make dev_pathX...py files into python notebook
- [x] Clean up
  - [x] Better names
  - [x] Get rid of testing/learning one-offs
  - [ ] Make instrumentation noise as a PySM3 Model
- [x] Change the CalTech shell script to get the LFI maps (Planck Assets) as well

Markov:
- [ ] Mount astropy outside Docker
- [x] Timing tests (shell script still needed)
- [ ] Autorun Python code in Docker

# UG Ready

- [ ] Where not enough variation exists (read: same thing), use PySM component_objects interface instead of preset_strings 
- [ ] Make tests
- [ ] Review: use or remove configuration items
- [ ] Make a script to get Planck Assets from ESA
- [ ] Time how long each preset string adds
- [ ] How much time does each preset string add?
- [ ] For each preset string, at each freq, in each field, what is the max and minimum value? What is the median, Q1, Q3, and sd?
- [ ] Are the CalTech maps the same as the ESA maps? Just need to load the maps and calculate the difference.
- [ ] Ensure variation between: sims, components, fields, noise
- [x] Local_system config tracking (I think I meant backing up the config files)
- [ ] Demonstration of smiley-face thing
- [ ] Convert dev path files (dev#, hydra#) to tutorial notebooks & tests
- [ ] Fiducial & Derived PS on a single plot in inspect_simulations.py
- [ ] Backup of script with created dataset (possibly to be used even after DVC is in place)
- [ ] Make noise as a pysm3.Model instead of ... what it is now
- [ ] Check if hydra sweeps work with the current logging setup (probably not, in hindsight)
- [ ] Ensure use of create_dirs flags in config file
- [ ] Ensure use of create_cache flag in config file
- [ ] Shell script to automatically get sizes and run times:
  - Clear noise cache (and others, if they exist)
  - Run 128_1_1 (create new noise cache)
  - Run 128_2_2 (get timing)
  - Run 512_1_1
  - Run 512_2_2
  - Run 2048_1_1
  - Run 2048_2_2
  - Get run time for each resolution @ 2_2
  - Get file size for each resolution, per sim
  - Without changing NUMBA_NUM_THREADS and OMP_NUM_THREADS ??? Or with? Or check =10, =100 for each?
  - Run 128_2_2 with non-recommended nside_sky of 128 or 256?

# Credit / References

https://camb.readthedocs.io/en/latest/CAMBdemo.html


# Structure
    
Conventions I'd love to have stuck with:
- When referring to detectors
  - adding an "s" is the plural of the following
  - "freq" is an integer. "frq" and "frequency" are not used.
  - "detector" is the object. "det" is used in lower level functions for brevity.
- When dealing with files
  - Use full words "source" and "destination"
  - Use abbreviation "dir" for "directory"


# Full output of URLErrors (temporary section):

```
(ml_cmb_pysm_sims) jimcamato@Jims-Laptop ml_cmb_pysm_sims % /Users/jimcamato/miniconda3/envs/ml_cmb_pysm_sims/bin/python /Users/jim
camato/Developer/ml_cmb_pysm_sims/fixed_map_synth4.py
545
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
Downloading https://portal.nersc.gov/project/cmb/pysm-data/websky/0.4/radio/radio_0584.0.fits
|==============================================================>------------------------------| 1.0G/1.6G (67.43%) ETA 24m 7s
File not found, please make sure you are using the latest version of PySM 3
While working on lowplus, error in detector 545.
<urlopen error Unable to open any source! Exceptions were {'https://portal.nersc.gov/project/cmb/pysm-data/websky/0.4/radio/radio_0584.0.fits': ContentTooShortError('File was supposed to be 1610619840 bytes but we only got 1086108642 bytes. Download failed.'), 'http://www.astropy.org/astropy-data/websky/0.4/radio/radio_0584.0.fits': <HTTPError 404: 'Not Found'>}>
857
Downloading https://portal.nersc.gov/project/cmb/pysm-data/websky/0.4/cib/cib_0857.0.fits
|==============================================================>------------------------------| 1.0G/1.6G (67.68%) ETA 23m25s
File not found, please make sure you are using the latest version of PySM 3
While working on lowplus, error in detector 857.
<urlopen error Unable to open any source! Exceptions were {'https://portal.nersc.gov/project/cmb/pysm-data/websky/0.4/cib/cib_0857.0.fits': ContentTooShortError('File was supposed to be 1610619840 bytes but we only got 1090100572 bytes. Download failed.'), 'http://www.astropy.org/astropy-data/websky/0.4/cib/cib_0857.0.fits': <HTTPError 404: 'Not Found'>}>
```
