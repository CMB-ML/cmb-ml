from cmbml.pyilc_redir import ILCInfo, Wavelets, harmonic_ILC, wavelet_ILC


def run_partial_api(cfg_path):
    # PyILC is written such that the input must be written to a file and then read
    #    I prefer to run it within my code so that I can debug within my environment
    ilc_info = ILCInfo(input_file=cfg_path)
    use_ilc_info(ilc_info)


def use_ilc_info(info):
    # Copied from main.py in the pyilc library
    ##########################
    # read in bandpasses
    info.read_bandpasses()
    # read in beams
    info.read_beams()
    ##########################

    ##########################
    # construct wavelets
    wv = Wavelets(N_scales=info.N_scales, ELLMAX=info.ELLMAX, tol=1.e-6, taper_width=info.taper_width)
    if info.wavelet_type == 'GaussianNeedlets':
        ell, filts = wv.GaussianNeedlets(FWHM_arcmin=info.GN_FWHM_arcmin)
    elif info.wavelet_type == 'CosineNeedlets': # Fiona added CosineNeedlets
        ell,filts = wv.CosineNeedlets(ellmin = info.ellmin,ellpeaks = info.ellpeaks)
    elif info.wavelet_type == 'TopHatHarmonic':
        ell,filts = wv.TopHatHarmonic(info.ellbins)
    elif info.wavelet_type == 'TaperedTopHats':
        ell,filts = wv.TaperedTopHats(ellboundaries = info.ellboundaries,taperwidths=info.taperwidths)
    else:
        raise TypeError('unsupported wavelet type')
    # example plot -- output in example_wavelet_plot
    #wv.plot_wavelets(log_or_lin='lin')
    ##########################

    ##########################
    # wavelet ILC
    if info.wavelet_type == 'TopHatHarmonic':
        info.read_maps() 
        info.maps2alms()
        info.alms2cls()
        harmonic_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
    else:
        wavelet_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
    ##########################
