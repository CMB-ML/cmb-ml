# CMB-ML: A Cosmic Microwave Background Radiation Dataset for Machine Learning

[![DOI](https://zenodo.org/badge/733740847.svg)](https://zenodo.org/badge/latestdoi/733740847)

<p align="center">
  (This page is under continuous construction.)
</p>

Contents:
- [Quick Start](#quick-start)
- [Introduction](#introduction)
  - [Simulation](#simulation)
  - [Baseline Models](#baseline-models)
  - [Metrics](#metrics)
- [Installation](#installation)
- [Demonstrations](#Demonstrations)
- [Comparing Results](#comparing-results)
  - [Outside Works](#outside-works)
  - [Errata](#errata)
- [Data File Links](#data-file-links)

# Quick Start

To get started:
- Get this repository
- Set up your Python environment, using the pyproject.toml
- Create or download datasets, see [Data File Links](#data-file-links)
- Train models 
- Run inference
- Compare results

See [Installation](#installation) and [Demonstrations](#Demonstrations) for more detail.


# Introduction

![CMB Radiation Example](assets/cmb.png)

The Cosmic Microwave Background radiation (CMB) signal is one of the cornerstones upon which modern cosmologists understand the universe. The signal is measured at different microwave frequencies - similar to how humans detect three different colors of light. Other natural phenomena either emit microwave signals or change the CMB signal itself. Thus, the signal must be separated out from these other contaminants. Modern machine learning and computer vision algorithms are seemingly perfect for the task, but generation of the data is cumbersome and no standard public datasets are available. Models and algorithms created for the task are seldom compared outside the largest collaborations. 

The CMB-ML dataset bridges these gaps. It handles simulation, modeling, and analysis.

Hydra is used to manage manage a pipeline so that coherent configurations are applied consistently. It uses the PySM3 simulation library in conjunction with CAMB, astropy, and Healpy to handle much of the astrophysics. Two baselines are implemented, with more to follow. One baseline comes from astrophysics: PyILC's implementation of the CNILC method. The other baseline uses machine learning: CMBNNCS's UNet8. The analysis portion of the pipeline first puts predictions into a consistent form, then generates summary statistics, and finally compares between models' performances.


## Simulation

![CMB Radiation Example](assets/observations_and_cmb_small.png)

The real CMB signal is observed at several microwave wavelengths. To mimic this, we make a ground truth CMB map and several contaminant foregrounds. We "observe" these at the different wavelengths, where each foreground has different levels. Then we apply instrumentation effects to get a set of observed maps.

## Baseline Models

Several methods have been used to clean the signal, but comparison is difficult. On the astrophysics side, the methods are generally considered "parametric" or "non-parametric." Algorithms of the first class model each foreground component, while the latter simply considers them all a common nuissance without differentiating them. Those typically operate on just a single map. There are also machine learning methods, which generally use UNets and consider the task a regression task, relying on many instances to train a model to make the differentiation.

<!-- 
### PyILC

PyILC implements three different Internal Linear Combination methods. We use one based on Cosine Needlets. A good explanation of needlets would need to start with Fourier and spherical harmonic transforms, then explain how wavelets are a hybrid that provide localization in the harmonic space. The CNILC method applies a needlet transform on each of the observation maps, then finds the linear combination of those transforms which minimizes covariance across the different observation frequencies.

### CMBNNCS

CMBNNCS implements  -->


## Metrics

We can compare the CMB predictions to the ground truths in order to determine how well the model works. However, because the models operate in fundamentally different ways, care is needed to ensure that they are compared in a consistent way. We first mask each prediction where the signal is often to bright to get meaningful predictions. We then remove effects of instrumentation from the predictions. The pipeline set up to run each method is then used in a slightly different way, to pull results from each method and produce output which directly compares them.


# Installation

Installation of CMB-ML requires setting up the repository, then getting the data assets for the portion you want to run. Demonstrations are available with many practical examples. The early ones cover how to set up CMB-ML to run on your system. The latter give examples of how to use the software, for those curious or hoping to extend it.

Setting up the repository:
- Download the repository
  - We suggest using `git clone https://github.com/JamesAmato/cmb-ml.git`, as the library is under development and may be updated.
- Get Python 3.9
  - General installation instructions are at: [Downloading Python](https://wiki.python.org/moin/BeginnersGuide/Download)
  - Be sure to get Python 3.9
    - One of the models, CMBNNCS, does not work with Python 3.10 at this time
  - Creating a conda environment with 3.9 will suffice
    - If you choose to do this:
    - Create the environment: `conda create -n <env_name> python=3.9` 
    - Activate the environment: `conda activate <env_name>`
    - Get the location of the python executable: `which python`
    - Exit the environment: `conda deactivate`
    - Do not use the environment for other things
- Get [Poetry](https://python-poetry.org/)
- Create a virtual environment:
  - Exit any virtual environments, otherwise Poetry will manage the currently active one instead
  - `poetry env use <your python location here>` will direct Poetry to use the appropriate Python installation, if it is not your default.
  - `poetry install` will get the libraries required, as listed in pyproject.toml
  - `poetry shell` will make it easier to run scripts from the command line
  - Alternatively, `poetry env info` will give you this virtual environment's python executable; this can be used with VS Code
- Configure your local system
  - In the configuration files, enter the directories where you will keep datasets and science assets
  - See [Setting up your environment](.demonstrations/C_setting_up_local.ipynb) for more information
- Download the science assets
  - These are available from the original sources and a mirror set up for this purpose
  - If you are not creating simulations, you only need one science asset: "COM_CMB_IQU-nilc_2048_R3.00_full.fits" (for the mask)
  - Files can be downloaded manually from [Science Assets on Box](https://utdallas.box.com/v/cmb-ml-science-assets)
  - Scripts are available in the `get_data` folder, which will download all files.
    - [Download from original sources](./get_data/get_orig_science_assets.py)
    - [Download from Box](./get_data/get_box_science_assets.py)
- Download the simulations
  - Two sets are available
    - The full set is IQU-512-1450
    - A smaller set for demonstration and debugging is I-128-1450
  - These can also be generated, as they are completely deterministic, given the same configurations
  - These can be downloaded manually at these box links:
    - [Box link for IQU-512-1450](https://utdallas.box.com/v/cmb-ml-IQU-512-1450)
    - [Box link for I-128-1450](https://utdallas.box.com/v/cmb-ml-I-128-1450)
  - Scripts for download are available as well
    - [Script for downloading IQU-512-1450](./get_data/get_box_IQU_512_1450.py)
    - [Script for downloading I-128-1450](./get_data/get_box_I_128_1450.py)
- Run code
  - Set up configurations
  - To generate simulations, use `main_sims.py`
  - To train, predict, and run analysis using CMBNNCS, use `main_cmbnncs.py`$\dagger$
    - Simulations must be available
  - To predict using PyILC, use `main_pyilc_predict.py`$\dagger$
    - Simulations must be available
  - To run analysis for PyILC, use `main_pyilc_analysis.py`
    - PyILC predictions must be available
  - To compare results between CMBNNCS and PyILC, use `main_analysis_compare.py`
    - PyILC and CMBNNCS analysis must already be done


# Demonstrations

Demonstrations exist for:
- [Hydra and its use in CMB-ML](./demonstrations/A_hydra_tutorial.ipynb) <!-- [Notebook Name](https://nbviewer.jupyter.org/github/username/repository/blob/branch/path/to/notebook.ipynb) -->
- [Hydra in scripts](./demonstrations/B_hydra_script_tutorial.ipynb) (*.py files)<!-- [Notebook Name](https://nbviewer.jupyter.org/github/username/repository/blob/branch/path/to/notebook.ipynb) -->
- [Setting up your environment](./demonstrations/C_setting_up_local.ipynb)
- [Getting and looking at simulation instances](./demonstrations/D_getting_dataset_instances.ipynb)
<!-- [Downloading full datasets](.) -->

# Comparing Results

The below is list of best results on the dataset. Please contact us through this repository to have your results listed. We do ask for the ability to verify those results.

We list below the datasets and model's aggregated (across the Test split) performance. We first calculate each measure for each simulation. The tables below contain average values of those for each metric. The metrics currently implemented are Mean Absolute Error (MAE), Mean Square Error (MSE), Normalized Root Mean Square Error (NRMSE), and Peak Signal-to-Noise Ratio (PSNR). The first three give a general sense of precision. PSNR gives a worst instance measure.

## On TQU-512-1450

### Pixel Space Performance

| Model   | MAE                    | MSE                  | NRMSE                    | PSNR                  |
|---------|------------------------|----------------------|--------------------------|-----------------------|
| CMBNNCS | $\bf{8.105 \pm 0.066}$ | $\bf{107.0 \pm 1.7}$ | $\bf{0.1020 \pm 0.0014}$ | $\bf{39.96 \pm 0.37}$ |
| CNILC   | $22.34 \pm 5.48$       | $813.6 \pm 386.2$    | $0.2744 \pm 0.0609$      | $32.36 \pm 1.79$      |


### Power Spectrum Performance

| Model   | MAE                  | MSE                                 | NRMSE                      | PSNR                  |
|---------|----------------------|-------------------------------------|----------------------------|-----------------------|
| CMBNNCS | $119.0 \pm 4.0$      | $\bf{(2.882 \pm 0.209)\times 10^4}$ | $\bf{0.07764 \pm 0.00190}$ | $\bf{31.00 \pm 0.27}$ |
| CNILC   | $\bf{101.6 \pm 3.7}$ | $(3.044 \pm 1.458)\times 10^4$      | $0.07855 \pm 0.01411$      | $30.94 \pm 0.86$      |


# Outside Works

CMB-ML was built in the hopes that researchers can compare on this as a standard. In the future, we hope to add more datasets. If you would like your model or dataset listed, please contact us.


## Works using datasets from this repository

None so far!


## Errata

Any issues in the original dataset will be listed here. If there are critical issues, we will do our best to keep the current dataset and release an updated one as well.

- The CMB ground truth may have been downgraded in a way that affects the power spectrum. We are investigating and looking at a different method.

# Data File Links

We provide links to the various data used. Alternatives to get this data are in `get_data` and the `Demonstrations`. "Science assets" refers to data created externally.

- Science assets
  - From the source
    - Planck Maps
      - For noise:
        - [Planck Collaboration Observation at 30 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 44 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 70 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits)
        - [Planck Collaboration Observation at 100 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 143 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 217 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 353 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_353-psb_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 545 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_full.fits)
        - [Planck Collaboration Observation at 847 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_full.fits)
      - For the mask:
        - [Planck Collaboration NILC-cleaned Map](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-nilc_2048_R3.00_full.fits)
      - WMAP9 chains:
        - [WMAP9 Chains, direct download](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_mnu_wmap9_chains_v5.tar.gz)
      - Planck delta bandpass table:
        - [Planck delta bandpass table, from Simons Observatory](https://github.com/galsci/mapsims/raw/main/mapsims/data/planck_deltabandpass/planck_deltabandpass.tbl)
      - [Downloading script](./get_data/get_orig_science_assets.py)
  - On Box: 
    - [All Science Assets](https://utdallas.box.com/v/cmb-ml-science-assets)
    - [Downloading script](./get_data/get_box_science_assets.py)

- Datasets
  - IQU-512-1450
    - Bulk data: [Box link, IQU-512-1450, monolithic](https://utdallas.box.com/v/cmb-ml-IQU-512-1450-lump), Note that this is ~1 TB
      - Download files individually. Downloading the directory will result in a single zip folder, which must then be extracted.
      - After downloading files individally, use something like the following to reassemble them:
          ```
          part_files=$(find "$dataset_dir" -name '*.part_*')
          total_size=$(du -cb $part_files | grep total$ | awk '{print $1}')
          cat $part_files | pv -s $total_size > "${data_dir}/${reconstructed_tar}"
          ``` 
    - Individual files: [Box Link, IQU-512-1450, individual](https://utdallas.box.com/v/cmb-ml-IQU-512-1450)
      - Each simulation instance is in its own tar file and will need to be extracted before use
      - The power spectra and cosmological parameters are in Simulation_Working.tar.gz
      - Log files, including the exact code used to generate simulations, are in Logs.tar.gz. No changes of substance have been made to the code in this archive.
      - A script for these download is available [here](./get_data/get_box_IQU_512_1450.py)
  - I-128-1450
    - Lower resolution simulations ($\text{N}_\text{side}=128$), for use when testing code and models
    - Instructions and examples on the way (Estimated June 24)
    - Bulk files: [Box link, I-128-1450, monolithic](https://utdallas.box.com/v/cmb-ml-I-128-1450-lump)
      - Files must be assembled with `cat`, as described above, then extracted
    - Individual instance files: [Box Link, I-128-1450, individual](https://utdallas.box.com/v/cmb-ml-I-128-1450)
    - A script for these download is available [here](./get_data/get_box_IQU_512_1450.py)
  - Files are expected to be in the following folder structure, any other structure requires changes to the pipeline yaml's:
```
└─ Datasets
   ├─ Simulations
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
- Trained models
  - CMBNNCS
    - [UNet8 trained on IQU_512_1450, at various epochs](https://utdallas.box.com/v/ml-cmb-UNet8-IQU-512-1450-bl)
