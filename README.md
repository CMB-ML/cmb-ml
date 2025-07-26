# CMB-ML: A Cosmic Microwave Background Radiation Dataset for Machine Learning

ZENODO (DOI) BADGE HERE

This is an old version of the CMB-ML repository. It reflects the state of the code at the time of the writing of the work to be presented at ICCV 2025 and is preserved for visibility. The blind version of the repository provided to reviewers is available at [https://github.com/CMB-ML/cmb-ml-ICCV2025](https://github.com/CMB-ML/cmb-ml-ICCV2025). The blind version matches the commit in this branch ending in `fffae92`. The blind version of the repository containing the notebooks used to produce figures in the paper is available at [https://github.com/CMB-ML/paper_figures-ICCV2025](https://github.com/CMB-ML/paper_figures-ICCV2025).

We suggest using the code as it appears on the `main` branch.

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
- Set up your Python environment
- Create datasets (Downloading is usually an option; contact the authors of the repository if needed)
- Train models
- Run inference
- Compare results

See [Installation](#installation) and [Demonstrations](#Demonstrations) for more detail.


# Introduction

![CMB Radiation Example](assets/readme_imgs/cmb.png)

The Cosmic Microwave Background radiation (CMB) signal is one of the cornerstones upon which modern cosmologists understand the universe. The signal must be separated out from other natural phenomena which either emit microwave signals or change the CMB signal itself. Modern machine learning and computer vision algorithms are seemingly perfect for the task, but generation of the data is cumbersome and no standard public datasets are available. Models and algorithms created for the task are seldom compared outside the largest collaborations. 

The CMB-ML dataset bridges the gap between astrophysics and machine learning. It handles simulation, modeling, and analysis.

This is somewhat complicated. We hope that the structure of CMB-ML gives you an opportunity to focus on a small portion of the pipeline. For many users, we expect this to be the modeling portion. Several examples are presented, showing how different methods can be used to clean the CMB signal. Details are provided below and in ancilliary material for how to acquire the dataset, apply a cleaning method, and use the analysis code included.

Other portions of the pipeline may also be changed. Simulated foregrounds can be changed simply with different parameters for the core engine. With more work, alternative or additional components can be used, or the engine itself can be changed out. A couple noise models particular to the Planck mission have been developed. At the other end of the pipeline, the analysis methods can be altered to match different methods. We are currently improving this portion of the pipeline.

A goal of this project has been to encapsulate the various stages of the pipeline separately from the operational parameters. It is our hope that this enables you to easily compare your results with other methods.

Several tools enable this work. [Hydra](https://hydra.cc/) is used to manage manage a pipeline so that coherent configurations are applied consistently. It uses the [PySM3](https://pysm3.readthedocs.io/en/latest/) simulation library in conjunction with [CAMB](https://camb.info/), [astropy](https://www.astropy.org/), and [Healpy](https://healpy.readthedocs.io/en/latest/) to handle much of the astrophysics. Three baselines are implemented, with more to follow. One baseline comes from astrophysics: [PyILC](https://github.com/jcolinhill/pyilc)'s implementation of the CNILC method. Another baseline uses machine learning: [cmbNNCS](https://github.com/Guo-Jian-Wang/cmbnncs)'s UNet8. A third is a simple [PyTorch](https://pytorch.org/) UNet implementation (intended to serve as a template for others). The analysis portion of the pipeline uses a few simple metrics from [scikit-learn](https://scikit-learn.org/stable/) along with the astrophysics tools.

## Simulation

![CMB Radiation Example](assets/readme_imgs/observations_and_cmb_small.png)

The real CMB signal is observed at several microwave wavelengths. To mimic this, we make a ground truth CMB map and several contaminant foregrounds. We "observe" these at the different wavelengths, where each foreground has different levels. Then we apply instrumentation effects to get a set of observed maps. The standard dataset is produced at a low resolution, so that many simulations can be used in a reasonable amount of space.

## Cleaning

Three models are included as baselines in this repository. One is a classic astrophysics algorithm, a flavor of **i**nternal **l**inear **c**ombination methods, which employs **c**osine **n**eedlets (CNILC). Another is a machine learning method (a UNet) implemented and published in the astrophysics domain, CMBNNCS. The third is a simple PyTorch implementation of a UNet, written to adhere more closely to typical design patterns.

The CNILC method was implemented by [PyILC](https://github.com/jcolinhill/pyilc), and is described in [this paper](https://arxiv.org/abs/2307.01043).

The cmbNNCS method was implemented by [cmbNNCS](https://github.com/Guo-Jian-Wang/cmbnncs), and is described in [this paper](https://iopscience.iop.org/article/10.3847/1538-4365/ac5f4a).

A third method, a [PyTorch](https://pytorch.org/) implementation of a UNet, is very similar to cmbNNCS and many other published models. Unlike cmbNNCS, it operates on small patches of maps instead of the full sky. This model is not discussed in the dataset paper.

## Analysis

We can compare the CMB predictions to the ground truths in order to determine how well the model works. However, because the models operate in fundamentally different ways, care is needed to ensure that they are compared in a consistent way. We first mask each prediction where the signal is often too bright to get meaningful predictions. We then remove effects of instrumentation from the predictions. The pipeline set up to run each method is then used in a slightly different way, to pull results from each method and produce output which directly compares them. The following figures were produced automatically by the pipeline, for quick review.

![Map Cleaning Example](assets/readme_imgs/CNILC_px_comp_sim_0005_I.png)
![Power Spectrum Example](assets/readme_imgs/CNILC_ps_comp_sim_0005_I.png)

Other figures are produced of summary statistics, but these are far more boring (for now!).

# New Methods

We encourage you to first familiarize yourself with the content of the tutorial notebooks and Hydra. Afterwards, you may want to follow either the patterns set in either the [classic method](cmbml/demo_external_method) or [ML method](cmbml/demo_patch_nn/) demonstrations. The main difference between these is the amount of stuff you want to do within CMB-ML's pipeline; if you already have code that can take input parameters, the patterns for classic methods may be more appropriate.

At this time, the classic method patterns are non-functional suggestions. To see operational code, the PyILC method should suffice. Please excuse any confusion caused by the hoops which enable us to run it on many simulations at once. Start with the [first top-level script](main_pyilc_predict.py), which gets the pipeline through the cleaning process. Then the [second top-level script](main_pyilc_analysis.py) must be run to finish the process. Both of these scripts use the same configuration file, there is simply a conflict in execution due to settings of `matplotlib`.

All of the ML patterns are functional. We suggest using the demonstration network as a prototype. The pipeline overview is in the [top-level script](main_patch_nn.py). This network operates on patches of sky maps, cut directly from the HEALPix arrangement. Some preprocessing stages are needed to enable fast training. The training and prediction executors follow common PyTorch design patterns ([train](cmbml/demo_patch_nn/stage_executors/E_train.py) and [predict](cmbml/demo_patch_nn/stage_executors/F_predict.py)). Both training and prediction use subclasses of a PyTorch [Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

As an alternative, see the cmbNNCS [top-level script](main_cmbnncs.py). The executors for this method are very similar to the demonstration network, though some changes are needed in order to adhere to the method described in the paper. It does differ more significantly in the [predict](cmbml/cmbnncs/stage_executors/E_predict.py) stage, as this model predicts entire skymaps in a single operation.

# Installation

See the next section if you don't want to install CMB-ML and instead just want the dataset.

Installation of CMB-ML requires setting up the repository, then getting the data assets for the portion you want to run. Demonstrations are available with practical examples. The early ones cover how to set up CMB-ML to run on your system.

Setting up the repository:
- Clone this repository
- Set up the Python environment, using `conda`
  - From within the repository, create a "cmb-ml" environment using the included `env.yaml`
    - `conda env create -f env.yaml`
  - Activate the environment
    - `conda activate cmb-ml`
- Get [PyILC](https://github.com/jcolinhill/pyilc)
  - Simply clone the repository
  - No installation is needed, CMB-ML runs the code as its own
  - This was run and tested with [the version from April 30, 2024](https://github.com/jcolinhill/pyilc/tree/7ced3ec392a520977b3c672a2a7af62064dcc296)
- Configure your local system
  - In the [configuration files](./cfg), enter the directories where you will keep datasets and science assets
  - In pyilc_redir, edit the `__init__.py` file to point to the directory containing your local installation of pyilc (containing the pyilc `inputs.py` and `wavelets.py`)
  - See [Setting up your environment](./demonstrations/C_setting_up_local.ipynb) for more information
- Download some external science assets and the CMB-ML assets
  - External science assets include Planck's observations maps (from which we get information for producing noise) and Planck's NILC prediction map (for the mask; NILC is a parameter)
  - These are available from the original sources and a mirror set up for this purpose
  - CMB-ML assets include the substitute detector information and information required for downloading datasets
  - If you are not creating simulations, you only need one external science asset: "COM_CMB_IQU-nilc_2048_R3.00_full.fits" (for the mask)
  - Scripts are available in the `get_data` folder, which will download all files.
    - [Downloads from original sources](./get_data/get_assets.py) gets files from the official sources (and the CMB-ML files from this repo)
    - If you prefer to download fewer files, adjust [this executor](get_data/stage_executors/A_get_assets.py) (not recommended)
- Next, set up to run.
  - You will need to either generate simulations or download them.

## Notes on Running Simulations

- Generating the set of simulations takes considerable time, due to the large number.
- Downloading them is likely to be faster.
- When generating simulations for the first time, [PySM3](https://pysm3.readthedocs.io/en/latest/) relies on [astropy](https://www.astropy.org/) to download and cache template maps.
  - These will be stored in an `.astropy` directory.
  - Downloading templates is sometimes interrupted resulting in an error and the code crashing. It is annoying and beyond our control. However, because the templates are cached, the pipeline can be resumed and will proceed smoothly.

## For CMB_ML_512_1450

- Download CMB_ML_512_1450
  - [Use the downloading script](./get_data/get_dataset.py)
  - `python ./get_data/get_dataset.py`
  - Files are visible at this [Box link for CMB_ML_512_1450](https://utdallas.box.com/v/cmb-ml-512-1450)
  - Alternatively, to generate simulations, use `python main_sims.py`
- To train, predict, and run analysis with the demonstration UNet model
  - `python main_patch_nn.py`
- To train, predict, and run analysis using CMBNNCS
  - `python main_cmbnncs.py`
- To predict using PyILC (this must be performed separately from analysis due to import issues)
  - `python main_pyilc_predict.py`
- To run analysis for PyILC
  - `python main_pyilc_analysis.py`
- To compare results between CMBNNCS and PyILC
  - `python main_analysis_compare.py`

## For CMB_ML_128_1450

This will run more quickly than the higher resolution.

- Download CMB_ML_128_1450:
  - [Use the downloading script](./get_data/get_box_CMB_ML_128_1450.py)
    - Change [cfg/pipeline/pipe_sim.yaml](../cfg/pipeline/pipe_sim.yaml) to use the correct set of shared links. In this yaml, look for `download_sims_reference` and change the `path_template` (replace '512' with '128').
  - Files are visible at this [Box link for CMB_ML_128_1450](https://utdallas.box.com/v/cmb-ml-128-1450)
  - Alternatively, to generate simulations, use `python main_sims.py dataset_name=CMB_ML_128_1450 nside=128`
- Run CMBNNCS on CMB_ML_128_1450 (the smaller UNet5 must be used):
    - `python main_cmbnncs.py dataset_name=CMB_ML_128_1450 working_dir=CMBNNCS_UNet5/ nside=128 num_epochs=2 use_epochs=[2] model/cmbnncs/network=unet5`
- Run PyILC on CMB_ML_128_1450:
    - `python main_pyilc_predict.py dataset_name=CMB_ML_128_1450 nside=128 ELLMAX=382 model.pyilc.distinct.N_scales=5 model.pyilc.distinct.ellpeaks=[100,200,300,383]`
    - `python main_pyilc_analysis.py dataset_name=CMB_ML_128_1450 nside=128 ELLMAX=382 model.pyilc.distinct.N_scales=5 model.pyilc.distinct.ellpeaks=[100,200,300,383]`
    - An even faster method is available, using PyILC's HILC method.
- Run Comparison:
    - `python main_analysis_compare.py --config-name config_comp_models_t_128`

# Dataset Only

If you only want to get the dataset, you can use [this notebook](./demonstrations/_0_get_dataset_only.ipynb) to download them. It includes a (short) list of required libraries.

# Demonstrations

CMB-ML manages a complex pipeline that processes data across multiple stages. Each stage produces outputs that need to be tracked, reused, and processed in later stages. Without a clear framework, this can lead to disorganized code, redundant logic, and errors.

The CMB-ML library provides a set of tools to manage the pipeline in a modular and scalable way. 

We include a set of demonstrations to help with both installation and introduction to core concepts. The first introduces our approach configuration management. That background paves the way to set up a local configuration and get the required files. Following this are a series of tutorials for the Python objects.

Most of these are in jupyter notebooks:
- [Hydra and its use in CMB-ML](./demonstrations/A_hydra_tutorial.ipynb)
- [Hydra in scripts](./demonstrations/B_hydra_script_tutorial.ipynb) (*.py files)
- [Setting up your environment](./demonstrations/C_setting_up_local.ipynb)
- [Getting and looking at simulation instances](./demonstrations/D_getting_dataset_instances.ipynb)
- [CMB_ML framework: stage code](./demonstrations/E_CMB_ML_framework.ipynb)
- [CMB_ML framework: pipeline code](./demonstrations/F_CMB_ML_pipeline.ipynb)
- [CMB_ML framework: Executors](./demonstrations/G_CMB_ML_executors.ipynb)

Only the Setting up your environment is really critical, though the others should help.

I'm interested in hearing what other demonstrations would be helpful. Please let me know what would be helpful. I've considered these notebooks:
- Executors, continued: showing how executors are set up for PyTorch training/inference and matplotlib figure production
- Looking at actual pipeline stages and explaining them
- Paper figure production (available, in another repository, need cleaning)



<!-- TODO: Move these to another repository; these are unneccesarily large files. -->
<!-- More demonstrations are available that use the data generated while running the CMB-ML pipeline. Note that (1) they require the pipeline has been run and (2) they were not developed as tutorials, unlike previous notebooks.
- [paper_figure_planck_obs_and_target.ipynb](../paper_figures/paper_figure_planck_obs_and_target.ipynb): Creates figures of Planck's observation maps and predicted CMB
- [dataset_results.ipynb](../paper_figures/dataset_results.ipynb): Plots maps after cleaning, to be assembled externally (e.g., in LaTeX)
- [make_component_maps.ipynb](../paper_figures/make_component_maps.ipynb): Creates single-component maps, for use in other analysis (next line)
- [paper_components.ipynb](../paper_figures/paper_components.ipynb): Creates figures showing single components (requires previous line having been run)
- [paper_figure_planck_variance.ipynb](../paper_figures/paper_figure_planck_variance.ipynb): Creates the figure of Planck's variance map at 100 GHz
- [planck_fwhm_detail.ipynb](../paper_figures/planck_fwhm_detail.ipynb): Creates figures with the detail view of Plancks's maps, such that the effect of different FWHMs is visible -->

# Comparing Results

The below is list of best results on the dataset. Please contact us through this repository to have your results listed. We do ask for the ability to verify those results.

We list below the datasets and model's aggregated (across the Test split) performance. We first calculate each measure for each simulation. The tables below contain average values of those for each metric. The metrics currently implemented are Mean Absolute Error (MAE), Mean Square Error (MSE), Normalized Root Mean Square Error (NRMSE), and Peak Signal-to-Noise Ratio (PSNR). The first three give a general sense of precision. PSNR gives a worst instance measure.

## On TQU-512-1450

### Pixel Space Performance

Operating on Deconvolved maps:

| Model   | MAE                   | RMSE                  | NRMSE                    | PSNR                  |
|---------|-----------------------|-----------------------|--------------------------|-----------------------|
| CMBNNCS | $\bf{18.50 \pm 0.19}$ | $\bf{23.26 \pm 0.23}$ | $\bf{0.2280 \pm 0.0030}$ | $\bf{32.72 \pm 0.36}$ |
| CNILC   | $59.83 \pm 0.12$      | $76.45 \pm 0.15$      | $0.7492 \pm 0.0138$      | $33.29 \pm 0.26$      |

Operating on maps convolved to 20.6 arcmin beam:

| Model   | MAE                    | RMSE                   | NRMSE                     | PSNR                  |
|---------|------------------------|------------------------|---------------------------|-----------------------|
| CMBNNCS | $\bf{3.314 \pm 0.017}$ | $\bf{4.235 \pm 0.023}$ | $\bf{0.04920 \pm 0.0009}$ | $\bf{45.71 \pm 0.41}$ |
| CNILC   | $6.391 \pm 0.420$      | $8.686 \pm 0.555$      | $0.1009  \pm 0.0062$      | $41.42 \pm 0.77$      |

Operating on maps convolved to 1 degree beam:

| Model   | MAE                    | RMSE                   | NRMSE                      | PSNR                  |
|---------|------------------------|------------------------|----------------------------|-----------------------|
| CMBNNCS | $\bf{0.594 \pm 0.005}$ | $\bf{0.788 \pm 0.008}$ | $\bf{0.01265 \pm 0.00037}$ | $\bf{56.81 \pm 0.48}$ |
| CNILC   | $3.870 \pm 0.63$       | $5.887 \pm 0.777$      | $0.09439 \pm 0.01168$      | $39.43 \pm 1.12$      |

# Outside Works

CMB-ML was built in the hopes that researchers can compare on this as a standard. In the future, we hope to add more datasets. If you would like your model or dataset listed, please contact us.

## Works using datasets from this repository

None so far!

# Errata

Unchanged, but of note as of July 2025:
- This code produces maps like those in the dataset downloaded.
- The set of cosmological parameters used is a modified standard model. This has been fixed in the main repository. This branch uses the sum of neutrino masses, does not include an amplitude parameter, and uses a scalar pivot which does not match those used by the WMAP group for generating the WMAP9 chains (this code used the CAMB default of 0.05, when the matching value should have been 0.002).

July 2025:
- This version reflects what was provided to reviewers; it is versioned so that a DOI can be issued. It reflects the state of the code at the time of the writing of the work to be presented at ICCV 2025 and is preserved for visibility. Some corrections have been made, either re-enabling functionality (blinding necessitated removal of links to university assets) or fixing bugs caught after-the-fact.
- Minor changes were made to get the seed values used to match those in the dataset. This was due to the generating code using longer strings (including the full dataset name) for the hash. The main branch uses shorter strings (for reproducability when making multiple datasets).
- Results in paper (and above) used a method for common beam convolution. This method is included in [this branch](https://github.com/CMB-ML/cmb-ml/tree/archive-iccv2025-convolution-fix) (or [this tag](https://github.com/CMB-ML/cmb-ml/releases/tag/archive-iccv-2025-analysis-beam-fix)).
  - To duplicate results, change the `target_beam` in the [analysis model](cfg/model/analysis/basic_analysis.yaml) configuration. Use either `_target_` "cmbml.utils.physics_beam.GaussianBeam" with a `beam_fwhm` or, for deconvolved results, use ""cmbml.utils.physics_beam.NoBeam"
  - There is no need to retrain models; simply remove those from the pipeline in the top-level script.

February 2025: 
- The repository history was edited to reduce the `.git` size.
  - The `.git` information was **300 MB**, due to several maps and large python notebooks.
  - It has been reduced to **21 MB**.  The bulk of this is images for this README and the demonstration notebooks.

November 2024: New dataset released:
- The noise generation procedure has been revised to be non-white noise
- The detector FWHM's were changed
  - Previously they were sub-pixel
  - They are now larger and still vary
  - More details [here](assets/CMB-ML/README.txt)
- The CMB signal was changed away from and returned to using CMBLensed
- Because the work is still unpublished and we do not know of anyone else using it, references to previous datasets have been updated. The original dataset will be removed June 30, 2025, unless we're made aware of anyone using it.

# Data File Links

We provide links to the various data used. Alternatives to get this data are in `get_data` and the `Demonstrations`. "Science assets" refers to data created by long-standing cosmological surveys.

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
        - [Planck Collaboration Observation at 847 GHz](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_full.fits)
      - For the mask:
        - [Planck Collaboration NILC-cleaned Map](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-nilc_2048_R3.00_full.fits)
      - WMAP9 chains for CMB simulation:
        - [WMAP9 Chains, direct download](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_mnu_wmap9_chains_v5.tar.gz)
      - Planck delta bandpass table:
        - [Planck delta bandpass table, from Simons Observatory](https://github.com/galsci/mapsims/raw/main/mapsims/data/planck_deltabandpass/planck_deltabandpass.tbl)
      - CMB-ML delta bandpass table:
        - [Original delta bandpass table, from Simons Observatory](assets/delta_bandpasses/CMB-ML/cmb-ml_deltabandpass.tbl)
          - CMB-ML modifies these instrumentation properties
        - Simply move the CMB-ML directory contained in assets/delta_bandpasses to your assets folder (as defined in e.g., [your local_system config](cfg/local_system/generic_lab.yaml))
      - [Downloading script](./get_data/get_assets.py)
  - On Box: 
    - [All Science Assets](https://utdallas.box.com/v/cmb-ml-science-assets)

- Datasets
  - CMB_ML_512_1450
    - Individual files: [Box Link, CMB_ML_512_1450](https://utdallas.box.com/v/cmb-ml-512-1450)
      - Each simulation instance is in its own tar file and will need to be extracted before use
      - The power spectra and cosmological parameters are in Simulation_Working.tar.gz
      - Log files, including the exact code used to generate simulations, are in Logs.tar.gz. No changes of substance have been made to the code in this archive.
      - A script for these download is available [here](./get_data/get_dataset.py)
  - CMB_ML_128_1450
    - Lower resolution simulations ($\text{N}_\text{side}=128$), for use when testing code and models
    - Individual instance files: [Box Link, CMB_ML_128_1450](https://utdallas.box.com/v/cmb-ml-128-1450)
    - A script for these download is available [here](./get_data/get_box_CMB_ML_128_1450.py)
      - Change [cfg/pipeline/pipe_sim.yaml](./cfg/pipeline/pipe_sim.yaml) to use the correct set of shared links. In this yaml, look for download_sims_reference and change the path_template (replace '512' with '128').

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
    - [UNet8 trained on CMB_ML_512_1450, at various epochs](https://utdallas.box.com/v/ml-cmb-UNet8-IQU-512-1450-bl)
