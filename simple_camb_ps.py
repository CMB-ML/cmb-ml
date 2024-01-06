import numpy as np
import camb
import matplotlib.pyplot as plt


class SimpleCAMBfromWMAP:
    def __init__(self, ellmax, cosmo_params):
        self.pars = camb.CAMBparams()

        # Original from https://camb.readthedocs.io/en/latest/CAMBdemo.html
        # pars.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120, mnu=0.06, omk=0.001, tau=0.054)
        # pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        # pars.set_for_lmax(ellmax, lens_potential_accuracy=0)

        self.pars.set_cosmology(H0=cosmo_params["H0"], 
                        ombh2=cosmo_params["omegabh2"], 
                        omch2=cosmo_params["omegach2"], 
                        tau=cosmo_params["tau"]
                        )
        self.pars.InitPower.set_params(
                                As=cosmo_params["As"], 
                                ns=cosmo_params["ns002"]
                                )

        self.pars.set_for_lmax(ellmax, lens_potential_accuracy=0)
        self.results = camb.get_results(self.pars)
        self.all_ps = self.results.get_cmb_power_spectra(self.pars, CMB_unit='muK')
        self.ps = self.all_ps['total']

    def write_cls(self, filename):
        self.results.save_cmb_power_spectra(filename=filename)


def show_camb_results(simple_camb):
    ps = simple_camb.all_ps

    totCL = simple_camb.ps
    unlensedCL = ps['unlensed_scalar']

    ls = np.arange(totCL.shape[0])
    fig, ax = plt.subplots(2,2, figsize = (6,6))
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
    simple_camb = SimpleCAMBfromWMAP(ellmax=ellmax,
                                     cosmo_params=cosmo_params)
    simple_camb2 = SimpleCAMBfromWMAP(ellmax=ellmax,
                                     cosmo_params=cosmo_params)
    a = simple_camb.ps - simple_camb2.ps
    print(np.max(a))


if __name__ == "__main__":
    main()
