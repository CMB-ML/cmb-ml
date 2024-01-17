import tempfile
import inspect
from pysm3 import CMBLensed
import camb


# Based on https://camb.readthedocs.io/en/latest/CAMBdemo.html


# TODO: remove get_cls which relies on a fixed cosmology
def get_cls(ellmax):
    # Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    # This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)

    # TODO: (Physics) Ensure correct CMB_unit
    # We want raw_cl: this is unnormalized C_ell; the alternative is normalized D_ell
    # TODO: (Physics) Using 'total' spectra; is this correct?
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', spectra=('total',), raw_cl=True)

    totCL=powers['total']

    return totCL


# TODO: remove write_cls which relies on a fixed cosmology
def write_cls(ellmax):
    filename = "cmb_spectrum.txt"
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    results.save_cmb_power_spectra(filename=filename)

    return filename


def create_cmb_lensed_from_params(cosmo_params, 
                                  lmax, 
                                  nside,
                                  cmb_ps_file_out=None,
                                  max_nside=None,
                                  cmb_seed=None, 
                                  apply_delens=False,
                                  delensing_ells=None,
                                  map_dist=None
                                  ) -> CMBLensed:
    # TODO Clean up documentation
    """Returns a pysm3 CMBLensed object

    Documentation from there

    Takes an input unlensed CMB and lensing spectrum from CAMB and uses
    Taylens to apply lensing, it optionally simulates delensing by
    suppressing the lensing power at specific scales with the user
    provided `delensing_ells`.

    Parameters
    ----------

    cosmo_params : dict(str: float)
        Cosmological parameters to use
    :param lmax: :math:`\ell_{\rm max}` you want
    nside: int
        Resolution parameter at which this model is to be calculated.
    cmb_ps_file_out: 
        Filename to save
    max_nside: int
        Keeps track of the the maximum Nside this model is available at
        by default 512 like PySM 2 models
    cmb_seed : int
        Numpy random seed for synfast, set to None for a random seed
    apply_delens : bool
        If true, simulate delensing with taylens
    delensing_ells : path
        Space delimited file with ells in the first columns and suppression
        factor (1 for no suppression) in the second column
    map_dist : bool
        Something to do with MPI. No PySM3 Documentation in version used.
    """

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()

    # I don't know what CAMB is doing, I just know I've got some lump of parameters to 
    #   plug in to two different functions, so I read signatures to split the params up
    set_cosmology_params = get_param_names(pars.set_cosmology)
    init_power_params = get_param_names(pars.InitPower.set_params)

    # Split the cosmo_params dictionary
    set_cosmology_args = {k: v for k, v in cosmo_params.items() if k in set_cosmology_params}
    init_power_args = {k: v for k, v in cosmo_params.items() if k in init_power_params}

    # Set the parameters
    pars.set_cosmology(**set_cosmology_args)
    pars.InitPower.set_params(r=0, **init_power_args)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    results = camb.get_results(pars)

    def get_cmb_lensed(fn):
        return CMBLensed(nside=nside,
                         cmb_spectra=fn,
                         max_nside=max_nside,
                         cmb_seed=cmb_seed,
                         apply_delens=apply_delens,
                         delensing_ells=delensing_ells,
                         map_dist=map_dist)

    if cmb_ps_file_out is None:
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tmp_file:
            results.save_cmb_power_spectra(filename=tmp_file)
            tmp_file.flush()
            cmb_lensed = get_cmb_lensed(tmp_file.name)
    else:
        results.save_cmb_power_spectra(filename=cmb_ps_file_out)
        cmb_lensed = get_cmb_lensed(cmb_ps_file_out)

    return cmb_lensed


# Janky way of handling not knowing what's what in CAMB
# TODO: Remove the need for this
def get_param_names(method):
    sig = inspect.signature(method)
    return [param.name for param in sig.parameters.values() if param.name != 'self']
