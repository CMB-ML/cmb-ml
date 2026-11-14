# CMB-ML: A Cosmic Microwave Background Radiation Dataset for Machine Learning

Contents:
- [Introduction](#introduction)
  - [Simulation](#simulation)
  - [Baseline Models](#baseline-models)
  - [Metrics](#metrics)
- [Installation](#installation)
- [Comparing Results](#comparing-results)
  - [Outside Works](#outside-works)
  - [Errata](#errata)
- [Data File Links](#data-file-links)
- [Simulation Dataset: I_512_1850_UnifWMAP_Wide](#prd-simulation-dataset-i_512_1850_unifwmap_wide)

# Introduction

![CMB Radiation Example](assets/readme_imgs/cmb.png)

The Cosmic Microwave Background radiation (CMB) signal is one of the cornerstones upon which modern cosmologists understand the universe. The signal must be separated out from these other natural phenomena which either emit microwave signals or change the CMB signal itself. Modern machine learning and computer vision algorithms are seemingly perfect for the task, but generation of the data is cumbersome. Simulated observations provide a means of developing and evaluating methods for this task.

CMB-ML is a Python framework for generating, managing, and analyzing simulated CMB observations. It produces multifrequency sky maps by combining realizations of the CMB with foreground emission and applying instrumental effects and noise. The resulting datasets can be used to develop and evaluate machine-learning methods and other approaches to CMB analysis.

We hope that the structure of CMB-ML gives you an opportunity to focus on a small portion of the pipeline. Details are provided below and in ancillary material for how to acquire the datasets, apply a cleaning method, and use the analysis code included.

A goal of this project has been to encapsulate the various stages of the pipeline separately from the operational parameters. It is our hope that this enables you to easily compare your results with other methods.

Several tools enable this work. [Hydra](https://hydra.cc/) is used to manage a pipeline so that coherent configurations are applied consistently. CMB-ML uses the [PySM3](https://pysm3.readthedocs.io/en/latest/) simulation library in conjunction with [CAMB](https://camb.info/), [astropy](https://www.astropy.org/), and [Healpy](https://healpy.readthedocs.io/en/latest/) to handle much of the astrophysics.

Several methods are implemented. One comes from astrophysics: [PyILC](https://github.com/jcolinhill/pyilc)'s implementation of the CNILC method. Another uses machine learning: [cmbNNCS](https://github.com/Guo-Jian-Wang/cmbnncs)'s UNet8. A third is a simple [PyTorch](https://pytorch.org/) UNet implementation intended to serve as a template. The D2PS method predicts power spectra instead of cleaned CMB maps. The analysis portion of the pipeline uses a few simple metrics from [scikit-learn](https://scikit-learn.org/stable/) along with the astrophysics tools.

CMB-ML was originally developed for research presented at ICCV 2025 and has subsequently been extended to support the Direct-to-Power-Spectrum (D2PS) project. This repository provides the shared simulation and pipeline infrastructure, while implementations of individual methods are maintained in separate repositories. The earlier ICCV 2025 version is still available in the [tagged release](https://github.com/CMB-ML/cmb-ml/releases/tag/v0.1.1-iccv2025).

## Simulation

![CMB Radiation Example](assets/readme_imgs/observations_and_cmb_small.png)

The real CMB signal is observed at several microwave wavelengths. To mimic this, we make a ground truth CMB map and several contaminant foregrounds. We "observe" these at the different wavelengths, where each foreground has different levels. Then we apply instrumentation effects to get a set of observed maps. The standard dataset is produced at a low resolution, so that many simulations can be used in a reasonable amount of space.

# Installation

Installation of CMB-ML requires setting up the repository and Python environment, then acquiring the data assets needed for the portions of the pipeline you intend to run.

- Assuming you install all your repos in the same directory, e.g. `~/repos/`
    - Go to that folder `cd ~/repos`
- Download the CMB-ML repository
    - `cd ~/repos` (or whatever top-level repo folder you use)
    - `git clone https://github.com/CMB-ML/cmb-ml.git`
    - `cd cmb-ml`
    - `git switch --detach v0.3.0-d2ps`
- Create the conda environment 
    - `conda env create -f env.yaml`
    - To use a different environment name, add `-n <name>` to the command.
- Activate the conda environment
    - `conda activate cmb-ml`
- Install CMB-ML
    - `pip install .`
 - Install PySM3
     - `python -m pip install pysm3==3.4.1`
 - Check the installation
     - `python -m pip check`
 - Configure the data directory
     - Choose a directory for downloaded assets and simulation datasets.
     - For example:
         - `export CMB_ML_DATA="$HOME/CMB_Data"`
         - `mkdir -p "$CMB_ML_DATA"`
     - CMB-ML stores science assets under `$CMB_ML_DATA/Assets/` and simulation datasets under `$CMB_ML_DATA/Datasets/`.
     - The environment variable must be set in each new terminal session, or added to your shell configuration.
 - Install [rclone](https://rclone.org/install/)
     - The public Box download utilities require `rclone` on your PATH.
     - No Box account or configured rclone remote is required.
  - Obtain assets as needed (see below)

# Science Assets
CMB-ML uses science assets from the Planck and WMAP missions. Copies of the required assets are available through the following public Box folders:
- [Planck assets](https://utdallas.box.com/s/ufr8k9dvtdfl60tzifmerieo532rib91)
- [WMAP9 chains](https://utdallas.box.com/s/t6plr547kihgh3shocde4ohcj4mz7hlh)
To download the assets, run the following from the CMB-ML repository root after completing the installation instructions:
```bash
python get_data/get_assets.py
```
This script will download and verify the required Planck and WMAP assets, extract the WMAP9 chains, and copy the supporting CMB-ML files included in this repository. Files will be stored under `$CMB_ML_DATA/Assets/`.

## Original Sources
The following links identify the original sources of the science assets used by CMB-ML. The download manifests in `$CMB_ML_DATA/Assets/CMB-ML/` identify the specific files distributed through Box.

- Science assets
  - From the source
    - Planck Maps
      - Planck Collaboration observation maps include variance maps needed for noise generation:
        - [Planck Collaboration Observation at 30 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 44 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 70 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 100 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 143 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 217 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 353 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 545 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 857 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_full.fits)
      - For the NILC mask, see the `TMASK` field (3 or 4 depending on 0- or 1-indexing) of:
        - [Planck Collaboration NILC-cleaned Map](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-nilc_2048_R3.00_full.fits)
      - WMAP9 chains for CMB simulation:
        - [WMAP9 Chains, direct download](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_mnu_wmap9_chains_v5.tar.gz)
      - Planck delta bandpass table:
        - [Planck delta bandpass table, from Simons Observatory](https://github.com/galsci/mapsims/raw/main/mapsims/data/planck_deltabandpass/planck_deltabandpass.tbl)
      - CMB-ML delta bandpass table:
        - [CMB-ML delta bandpass table](assets/CMB-ML/cmb-ml_deltabandpass.tbl)
          - CMB-ML modifies these instrumentation properties

## Notes on Running Simulations

- Generating the set of simulations takes considerable time, due to the large number.
- Downloading them is likely to be faster.
- When generating simulations for the first time, [PySM3](https://pysm3.readthedocs.io/en/latest/) relies on [astropy](https://www.astropy.org/) to download and cache template maps.
  - These will be stored in an `.astropy` directory.
  - Downloading templates is sometimes interrupted resulting in an error and the code crashing. It is annoying and beyond our control. However, because the templates are cached, the pipeline can be resumed and will proceed smoothly.

To generate simulations, first complete the installation and science-asset acquisition procedures described above. Run the following from the repository root:
```bash
python main_sims.py splits=1-1
```
This generates a single Test simulation using the current simulation configuration. Output is stored under `$CMB_ML_DATA/Datasets/`. Simulation parameters and pipeline stages are configured through the YAML files in `cmbml/cfg/`.


# Datasets

## I_512_1850_UnifWMAP_Wide

**Note: This is the dataset referenced in the D2PS paper.**

This dataset is distributed as 222 uncompressed `.tar` shards on [UT Dallas Box](https://utdallas.box.com/s/tvq6c72fj2t4lgkqdbh13dvn310da047). It includes the existing simulation and working data. The shards do not need to be repackaged or regenerated to use the released dataset.

The command-line downloader requires Python 3 and [rclone](https://rclone.org/install/) on your `PATH`. All commands below are run from the root of this CMB-ML repository. The dataset is downloaded outside the repository, to your `$CMB_ML_DATA` location.

To download (approximately 877 GiB of archives plus small metadata files):

```bash
python3 get_data/download_unifwmap_wide.py \
    --manifest assets/CMB-ML/UnifWMAP_Wide_download_manifest.json \
    --dest "$CMB_ML_DATA/UnifWMAP_Wide_shards" \
    --progress
```

The downloader validates the size of each downloaded file and verifies the shards against their published SHA-256 hashes. Previously downloaded files are checked and skipped when valid.

Extract to the dataset directory (requiring approximately 877 GiB more disk space):

```bash
python3 get_data/extract_tar_shards.py \
    "$CMB_ML_DATA/UnifWMAP_Wide_shards" \
    "$CMB_ML_DATA/Datasets/I_512_1850_UnifWMAP_Wide" \
    --progress-interval 5
```

The extractor checks the shard hashes and validates extracted members against the manifest. Its destination argument is the exact restored dataset directory; it recreates the contents of the original dataset inside that directory. Extraction can be restarted using the same command.

To download just one file, use `--only` with its exact filename; for example, replace the download command above with the following small metadata test:

```bash
python3 get_data/download_unifwmap_wide.py \
    --manifest assets/CMB-ML/UnifWMAP_Wide_download_manifest.json \
    --dest "$CMB_ML_DATA/UnifWMAP_Wide_test" \
    --only SHA256SUMS
```

**If retaining both the downloaded shards and extracted dataset, plan for approximately 1.8 TiB of disk space, plus filesystem overhead.**

The public Box folder, the download manifest, and these two utilities are for `I_512_1850_UnifWMAP_Wide` only. Earlier datasets, including those used for the ICCV work, retain their existing download instructions and the original [`get_data/get_dataset.py`](./get_data/get_dataset.py) implementation.

## ICCV 2025 Datasets

The ICCV2025 version of this repository is available at https://github.com/CMB-ML/cmb-ml/releases/tag/v0.1.1-iccv2025

It references two different datasets, one at Nside=512 and the other, for smoke testing, at Nside=128.

The datasets are available on UT Dallas Box:
- [CMB_ML_512_1450](https://utdallas.box.com/v/cmb-ml-512-1450)
- [CMB_ML_128_1450](https://utdallas.box.com/v/cmb-ml-128-1450)

The datasets can be downloaded using `get_data/get_dataset.py`.
For Nside=512:
```bash
python get_data/get_dataset.py
```
For Nside=128:
```bash
python get_data/get_dataset.py nside=128
```

## Simulation file structure

Files are expected to follow the directory structure below. Other structures require changes to the pipeline YAML configurations:
```
$CMB_ML_DATA/
└─ Datasets
   └─ Some Particular Dataset (e.g., I_512_1850_UnifWMAP_Wide)
      ├─ Simulation
      |   ├─ Train
      |   |     ├─ sim0000
      |   |     ├─ sim0001
      |   |     └─ etc...
      |   ├─ Valid
      |   |     ├─ sim0000
      |   |     ├─ sim0001
      |   |     └─ etc...
      |   └─ Test
      |         ├─ sim0000
      |         ├─ sim0001
      |         └─ etc...
      └─ Simulation_Working
          ├─ Simulation_B_Noise_Cache
          ├─ Simulation_C_Configs            (containing cosmological parameters)
          └─ Simulation_CMB_Power_Spectra
```

# Baseline Methods

Several methods have been implemented using the CMB-ML framework. Their implementations, installation instructions, and associated resources are maintained in separate repositories.

- **PyILC (CNILC):** [CMB-ML/cmb-ml-pyilc](https://github.com/CMB-ML/cmb-ml-pyilc) — A needlet internal linear combination method for CMB component separation.
- **cmbNNCS (UNet8):** [CMB-ML/cmb-ml-cmbnncs](https://github.com/CMB-ML/cmb-ml-cmbnncs) — A machine-learning approach to CMB component separation using a UNet8 architecture.
- **PatchNN:** [CMB-ML/cmb-ml-patch-nn](https://github.com/CMB-ML/cmb-ml-patch-nn) — A simple PyTorch UNet implementation intended to serve as a template.
- **Direct-to-Power-Spectrum (D2PS):** [CMB-ML/e-d2ps](https://github.com/CMB-ML/e-d2ps) - A method which estimates the CMB angular power spectrum directly rather than producing a cleaned CMB map.
