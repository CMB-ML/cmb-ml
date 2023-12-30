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

Output is currently in K_RJ; I think it should be K_CMB. That's just the result of a bad guess I made early on and I'll correct it. The result is low signal at higher detector bands.

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

# To do

* Figure out pip and conda installation steps
* Noise, CMB, and all components in a single map
* CMB component determined by cosmological parameter draws from WMAP 9 chains.
* Output, per component, default variation (requires 2 runs)
* Make presentation of the above
* Where not enough variation exists (read: same thing), use PySM component_objects interface instead of preset_strings 
* Move to Markov
* Traceability/reproducability (this is a lot of stuff, todo: break down further)
* Run simulations v1
* Clean up (better names for files, get rid of testing/learning one-offs)
* Are the CalTech maps the same as the ESA maps? Just need to load the maps and calculate the difference.
* Change the CalTech shell script to get the LFI maps as well. Or just... make a different script to get them from ESA.

Physics questions:
* Does this method work?
* Why do maps look weird (especially polarization) re: smoothing?
* Why do maps look weird (especially polarization) re: instrument noise?
* Are unit conversions correct?
* Is variance correct?
