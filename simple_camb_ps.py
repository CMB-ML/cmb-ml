import os
import numpy as np
import pysm3
import healpy as hp
import camb
import matplotlib.pyplot as plt
import pysm3.units as u


def setup_camb(cosmo_params):
    pars = camb.CAMBparams()

    # Original from https://camb.readthedocs.io/en/latest/CAMBdemo.html
    # pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
    # pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    # pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

    pars.set_cosmology(H0=cosmo_params["H0"], 
                       ombh2=cosmo_params["omegabh2"], 
                       omch2=cosmo_params["omegach2"], 
                       tau=cosmo_params["tau"]
                       )
    pars.InitPower.set_params(
                              As=cosmo_params["As"], 
                              ns=cosmo_params["ns002"]
                              )
    return pars


def run_camb(pars, ellmax):
    pars.set_for_lmax(ellmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    return results


def write_cls(filename, camb_results):
    camb_results.save_cmb_power_spectra(filename=filename)


def main():
    ellmax=2000
    cosmo_params = dict(
        H0=67.4,
        omegabh2=0.0224,
        omegach2=0.120,
        As=2e-9,
        ns002=0.965,
        tau=0.054
    )
    camb_params = setup_camb(cosmo_params)
    camb_results = run_camb(camb_params, ellmax=ellmax)
    ps = camb_results.get_cmb_power_spectra(camb_params, CMB_unit='muK')

    totCL = ps['total']
    unlensedCL = ps['unlensed_scalar']

    ls = np.arange(totCL.shape[0])
    fig, ax = plt.subplots(2,2, figsize = (12,12))
    ax[0,0].plot(ls,totCL[:,0], color='k')
    ax[0,0].plot(ls,unlensedCL[:,0], color='C2')
    ax[0,0].set_title(r'$TT\, [\mu K^2]$')
    ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0])
    ax[0,1].set_title(r'Fractional TT lensing')
    ax[1,0].plot(ls,totCL[:,1], color='k')
    ax[1,0].plot(ls,unlensedCL[:,1], color='C2')
    ax[1,0].set_title(r'$EE\, [\mu K^2]$')
    ax[1,1].plot(ls,totCL[:,3], color='k')
    ax[1,1].plot(ls,unlensedCL[:,3], color='C2')
    ax[1,1].set_title(r'$TE\, [\mu K^2]$')
    plt.show()


if __name__ == "__main__":
    main()
