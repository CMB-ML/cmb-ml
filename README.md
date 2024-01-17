# ml_cmb_pysm_sims
Rough development repository for PySM simulations for ML project

# Set Up

(This Section is a DRAFT)

- Clone the repositories (both ml_cmb_pysm_sims and the dependency, pysm3).
    - Either (git with HTTPS)
    - `git clone https://github.com/JamesAmato/ml_cmb_pysm_sims.git`
    - `git clone https://github.com/galsci/pysm.git`
    - Or (git with SSH):
    - `git clone git@github.com:JamesAmato/ml_cmb_pysm_sims.git`
    - `git clone git@github.com:galsci/pysm.git`

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
     - That may fail for healpy, but install everything else except pysm3 itself
     - Then do `pip install --no-deps .`

- Still missing numba and toml
    - Run `conda install numba toml tqdm`
    - Maybe this should go earlier?

- Get the Needed files (see next section)

# Needed files

"planck_assets/" files are needed for noise generation. There's a lot of data there (~2GB?). "fidu_noise/" files are only for comparisons of noise. WMAP 9 chains are not yet used. 

There are two ways to get the planck_assets maps. I found that the ESA Planck page is slower than CalTech, but there could be many reasons for that. I'm not sure if the CalTech maps are Bandpass Corrected (TODO below). I'm pretty sure the Planck Maps are all Bandpass corrected.

Option 1: Use either get_planck_maps_caltech.sh or get_and_symlink_planck_maps to get the observation maps (the latter is suggested; it was created because I store huge files outside the repository for use with other code). Option 2: From [ESA Planck Page](https://pla.esac.esa.int/#results), choose Maps, then Advanced Search, using the terms "LFI_SkyMap_%-BPassCorrected_1024_R3.00_full.fits" and "HFI_SkyMap_%_2048_R3.01_full.fits" (note that the 353 should be with "-psb", as the version without that has some issue mentioned in the Planck Wiki).

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
- Noise files (found with "ffp10_noise_%%%_full_map_mc_00000.fits" on the ESA archive; unknown if CalTech has these.):
  - fidu_noise/ffp10_noise_030_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_044_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_070_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_100_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_143_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_217_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_353_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_545_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_857_full_map_mc_00000.fits
  - fidu_noise/ffp10_noise_353_psb_full_map_mc_00000.fits

From [NASA WMAP page](https://lambda.gsfc.nasa.gov/product/wmap/dr5/params/lcdm_wmap9.html), [Chain Files Direct Link](https://lambda.gsfc.nasa.gov/data/map/dr5/dcp/chains/wmap_lcdm_wmap9_chains_v5.tar.gz)

# Notes

first_sim.py is just PySM3 without custom stuff.

fixed_map_synth1.py adds the CMB component ("Syn"). Currently I use CAMB to get a power spectrum. Currently, cosmological parameters are hard-coded. Later they will be drawn from WMAP9 Chains. Beam convolution is done poorly with a hard-coded beam fwhm roughly corresponding to the 100 GHz detector.

fixed_map_synth2.py adds beam convolutions which match values pulled from a PySM3 data table for the Planck Mission. Ideally, these would be pulled from Planck assets instead. That's a future update.

fixed_map_synth3.py adds noise. Noise is generated using official Planck observation maps. Those maps have the following fields
* TTYPE1  = 'I_STOKES'
* TTYPE2  = 'Q_STOKES'
* TTYPE3  = 'U_STOKES'
* TTYPE4  = 'HITS    '
* TTYPE5  = 'II_COV  '
* TTYPE6  = 'IQ_COV  '
* TTYPE7  = 'IU_COV  '
* TTYPE8  = 'QQ_COV  '
* TTYPE9  = 'QU_COV  '
* TTYPE10 = 'UU_COV  '

I use the II_COV, QQ_COV, and UU_COV fields, following the precedent set by Petroff. Those values are (if I understand correctly) the variance values per pixel for that field; it's covariance only in terms of the different I/Q/U, not how the pixels covary. I can draw random values using a Numpy RNG, which uses a scale parameter that's the standard deviation. Thus, I take the square root of the map values. I'm kinda out of my depth on that.

The noise in the polarization maps is extreme compared to the CMB signal. I checked that this is reasonable in two ways. The magnitude of CMB anisotropy in T is ~300 uK; for Q and U it's ~2.5 uK. Compare that to the variance maps: for instance, at 100 GHz the median T noise sd value is 44 uK (makes sense); Q is 68 uK; U is 67 uK. I also looked at the 2018 release noise simulations. In this case I think a good marker is the IQR. I see similar results: T: 59 uK, Q:89 uK, U:88 uK.

The comparison to variance results are generated by "inspect_planck_fits.py" and saved in "inspect_planck_results.txt". The comparison to noise simulations results are generated by "inspect_noise_fits.py" and saved in "inspect_noise_results.txt". 

fixed_map_synth4.py requires downloading all files (~63.2 GB). Having trouble with getting some files (timing out? it downloads for a while, then bails every time. Need to revisit testing at UTD/Markov)
 * http://www.astropy.org/astropy-data/websky/0.4/radio/radio_0584.0.fits
 * https://portal.nersc.gov/project/cmb/pysm-data/websky/0.4/radio/radio_0584.0.fits

check_variation_in_base.py loads each working component (by string) and creates two sky simulations without resetting numpy's random seed (deliberately, TODO verify). Findings: no variation exists by default.

try_cmb_component.py produces a pysm3 CMB component using WMAP chains information. It demonstrates the option to save the power spectrum file.

# To do

- [x] Figure out pip and conda installation steps
- [x] Noise, CMB, and all components in a single map (see fixed_map_synth4.py)
- [x] CMB component determined by cosmological parameter draws. (partial, see simple_camb_ps.py)
- [x] CMB component determined by cosmological parameter draws from WMAP 9 chains. 
- [x] Output, per component, default variation (requires 2 runs); compare them (see check_variation_in_base.py)
- [x] Switch to uK_CMB instead of uK_RJ 
    - [x] simple fix: when initializing Sky(), include "output_unit='uK_CMB'"
    - [ ] uglier fix: noise is broken (see fixed_map_synth3.py [not 4] results)
- [ ] Where not enough variation exists (read: same thing), use PySM component_objects interface instead of preset_strings 
- [ ] Make presentation of the above
- [ ] Move to Markov
- [ ] Traceability/reproducability (this is a lot of stuff, todo: break down further)
- [ ] Run simulations v1
- [ ] Clean up (better names for files, get rid of testing/learning one-offs)
- [ ] Are the CalTech maps the same as the ESA maps? Just need to load the maps and calculate the difference.
- [x] Change the CalTech shell script to get the LFI maps as well. Or just... make a different script to get them from ESA.

Physics questions:
* Does this method work?
* Why do maps look weird (especially polarization) re: smoothing?
* Why do maps look weird (especially polarization) re: instrument noise?
* Are unit conversions correct?
* Is variance correct?


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
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/a6997a166010916ce04ada58b4872b86/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/1a77be884471968effb02a9e53c2a236/contents
545
857
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/2233e1624d4c24ef8100824b8c088c68/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/ca73bb01bc98b80a5f50b6066aa9c8fe/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/2233e1624d4c24ef8100824b8c088c68/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/ca73bb01bc98b80a5f50b6066aa9c8fe/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/a6997a166010916ce04ada58b4872b86/contents
No physical unit associated with file /Users/jimcamato/.astropy/cache/download/url/1a77be884471968effb02a9e53c2a236/contents
545
857
(ml_cmb_pysm_sims) jimcamato@Jims-Laptop ml_cmb_pysm_sims % 
```


# Credit / References

https://camb.readthedocs.io/en/latest/CAMBdemo.html


# Structure

Making simulations principles:
    - Define the steps
        Each step is defined by the single data object output.
        Making a simulation set is a process composed of many steps. 
    - Each step should deterministically follow the previous steps.
        Use seeds stored in configuration files.
    - Leave lots of breadcrumbs, leave few loaves of bread.
        Save the small ones. Save the large ones only if they're to be used in the end analysis.
        Save configuration files, even if they're only used to produce other configuration files.
        - Sepcifically, keep as individual files:
            - cosmo params
            - full dataset setup configuration
            - each split setup configuration
            - each simulation setup configuration
            - power spectra
            - contaminated maps at each frequency
        - Do not keep:
            - Individual contaminants at each frequency
    - Have two sets of configurations to look up for any step
        - The first set is common across all simulations, the other is everything else.
        - There are no intermediate levels.
        - The common set is a set.
        - The varied set may be a set (power spectra cosmo params 😓)
        - For specfic steps:
            - For making Power Spectra, this is common set and whatever contains WMAP parameters
    - Process is data
        Ensure the scripts used to produce some data are associated with and recoverable for the data.
    
Creating a dataset
    - Parameters
        - local configuration
            - datasets root directory
            - source data locations
                - wmap_chains_files
                - map files (noise data)
                - planck detector information
        - file system
            - DatasetName/Raw/SplitName/SimulationNum
                * At Raw/ level: cfg for splits
                * At SplitName level: cfg for this split, ps if needed
                * At SimulationNum level: cfg for this simulation
            - common filenames
                - fiducial cmb map, clean
                - derived cmb map, lensing only (TBD)
                - derived cmb power spectrum (function of the fiducial map)
                - derived cmb power spectrum (function of the derived w/ lensing map) (TBD)
                - obs maps fn format (include {} for detector frequency)
        - splits configuration
            - names of splits
            - number of simulations for each split
            - if split has single cosmo_param_set
        - simulation configuration
            - contaminant strings
            - Contaminant classes and parameters (possibly in subconfigs) (TBD, none currently)
            - wmap parameters to draw (list of strings)
            - detector frequencies

        Autogenerated configurations:
        - split configuration (autogenerated)
            - name of split
            - number of simulations
            - if split has single cosmo_param_set
            - split PySM3 base seed (x10^6 for each split)
            - reference to autogeneration script
            - WMAP param indices (end of file)
        - simulation configuration (autogenerated)
            - WMAP param index
            - WMAP-sourced cosmo params
            - PySM3 seeds (x10^2 for each simulation)
            - power spectrum filename (relative to DatasetName/, meaning it always includes Raw/SplitName for clarity)

    - Procedure
        - Validate local and splits configurations
        - Generate WMAP indices list
        - Generate split configurations
        - Generate simulation configurations
        - Generate power spectra based on WMAP cosmo parameters
        - Generate maps
        
