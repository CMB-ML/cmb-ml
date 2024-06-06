"""
This module has dual purposes:
1. Generate maps of the components of the CMB and plot them
2. Demonstrate how CMBML classes can be used to interact with the configuration yamls
"""

from functools import partial

import hydra
import numpy as np

import pysm3
import matplotlib.pyplot as plt
import healpy as hp

import seaborn as sns

from src.core.config_helper import ConfigHelper
from src.core.namers import Namer
from src.core.asset_handlers.healpy_map_handler import HealpyMap
from src.core.asset_handlers.qtable_handler import QTableHandler  # import to register handler
from src.utils.planck_instrument import make_instrument
from src.sims.random_seed_manager import FieldLevelSeedFactory
from src.sims.physics_instrument_noise import planck_result_to_sd_map, make_random_noise_map


@hydra.main(version_base=None, config_path="cfg", config_name="config_sim_t")
def main(cfg):
    namer = Namer(cfg)
    cfg_helper = ConfigHelper(cfg)
    planck_info_asset = cfg_helper.get_some_asset_in(name_tracker=namer,
                                                     asset="planck_deltabandpass",
                                                     stage_str="raw_reference")
    noise_map_asset = cfg_helper.get_some_asset_in(name_tracker=namer,
                                                   asset="noise_src_maps",
                                                   stage_str="make_noise_cache")
    planck_info = planck_info_asset.read()
    instrument = make_instrument(cfg, planck_info)
    map_handler = HealpyMap()
    namer.working_dir = "Component_Figures"
    map_path_template = "{root}/{dataset}/{working}/at_{freq}_component_{component}.fits"
    fig_path_template = "{root}/{dataset}/{working}/at_{freq}_component_{component}.png"

    # Get the list of all components to be plotted
    components = list(cfg.model.sim.preset_strings)

    # Add the CMB component to the list of components to be plotted
    components.append("c1")

    # Get simulation parameters
    nside_sky = cfg.model.sim.nside_sky
    lmax = int(cfg.model.sim.pysm_beam_lmax_ratio * cfg.scenario.nside)
    output_unit = cfg.scenario.units
    nside_out = cfg.scenario.nside

    for nom_freq in [30, 44, 70, 100, 143, 217, 353, 545, 857]:
        with namer.set_context("freq", nom_freq):
            print(f"Working on {nom_freq} GHz")

            use_frequency = instrument.dets[nom_freq].cen_freq

            # partial is used just so it's easier to comment and uncomment the function calls
            _make_maps = partial(make_maps, components=components,
                                            detector=nom_freq,
                                            nside_sky=nside_sky,
                                            nside_out=nside_out,
                                            lmax=lmax,
                                            output_unit=output_unit,
                                            use_frequency=use_frequency,
                                            instrument=instrument,
                                            map_handler=map_handler,
                                            namer=namer,
                                            map_path_template=map_path_template)
            _make_noise = partial(make_noise_map, cfg=cfg,
                                                  instrument=instrument,
                                                  detector=nom_freq,
                                                  namer=namer,
                                                  in_noise_src=noise_map_asset,
                                                  field_idx=4,
                                                  nside=nside_out,
                                                  map_path_template=map_path_template,
                                                  map_handler=map_handler)
            _make_bounds = partial(make_bounds, components=[*components, "noise"],
                                                map_path_template=map_path_template,
                                                map_handler=map_handler,
                                                namer=namer)
            _make_figs = partial(make_figs, components=[*components, "noise"],
                                            map_path_template=map_path_template,
                                            map_handler=map_handler,
                                            namer=namer,
                                            fig_path_template=fig_path_template)

            _make_maps()
            _make_noise()
            # _make_bounds()

            log_plot = {'min': None, 'max': None, 'norm': 'log'}
            auto_plot = {'min': None, 'max': None, 'norm': None}

            plotting_params = dict(
                d9=auto_plot,
                s4=auto_plot,
                f1=auto_plot,
                a1=auto_plot,
                co1=auto_plot,
                cib1=auto_plot,
                ksz1=auto_plot,
                tsz1=auto_plot,
                rg1=auto_plot,
                c1=auto_plot,
                noise=auto_plot,
            )

            _make_figs(plotting_params=plotting_params)


def make_maps(components,
              detector,
              nside_sky,
              nside_out,
              lmax,
              output_unit,
              use_frequency,
              instrument,
              map_handler,
              namer,
              map_path_template):
    # Produce maps and plots for each component
    for preset in components:
        print(f"Generating map for {preset} component")
        preset_strings = [preset]
        sky = pysm3.Sky(nside=nside_sky, 
                        preset_strings=preset_strings,
                        output_unit=output_unit)
        skymap = sky.get_emission(use_frequency)[0]
        smoothed = pysm3.apply_smoothing_and_coord_transform(skymap, 
                                                              instrument.dets[detector].fwhm, 
                                                              lmax=lmax, 
                                                              output_nside=nside_out)
        with namer.set_contexts({"component": preset}):
            map_path = namer.path(map_path_template)
            map_handler.write(path=map_path, data=smoothed)


def make_noise_map(cfg, 
                   instrument, 
                   detector, 
                   namer, 
                   in_noise_src, 
                   field_idx, 
                   nside, 
                   map_path_template, 
                   map_handler):
    # Produce noise map
    print("Generating noise map")
    seed_factory = FieldLevelSeedFactory(cfg, cfg.model.sim.noise.seed_string)
    noise_seed = seed_factory.get_seed(split="", sim=0, freq=detector, field_str="T")
    center_frequency = instrument.dets[detector].cen_freq

    fn       = cfg.model.sim.noise.src_files[detector]
    src_root = cfg.local_system.assets_dir
    contexts_dict = dict(src_root=src_root, filename=fn)

    with namer.set_contexts(contexts_dict):
        src_path = in_noise_src.path

    # Get a standard deviation map from a Planck result
    st_dev_skymap = planck_result_to_sd_map(fits_fn=src_path, 
                                            hdu=1, 
                                            field_idx=field_idx, 
                                            nside_out=nside, 
                                            cen_freq=instrument.dets[detector].cen_freq)

    noise_map = make_random_noise_map(st_dev_skymap, noise_seed, center_frequency)
    with namer.set_contexts({"component": "noise"}):
        map_path = namer.path(map_path_template)
        map_handler.write(path=map_path, data=noise_map)


def make_bounds(all_preset_strings, 
              map_path_template,
              map_handler,
              namer):
    # Produce plots for each component

    for preset in all_preset_strings:
        with namer.set_contexts({"component": preset}):
            map_path = namer.path(map_path_template)
            smoothed = map_handler.read(map_path)[0]
            sns.boxplot(x=smoothed)
            plt.xscale('log')
            plt.show()


def make_figs(components, 
              map_path_template,
              map_handler,
              namer, 
              fig_path_template, 
              plotting_params):
    # Produce plots for each component

    for preset in components:
        print(f"Creating image for {preset} component")
        with namer.set_contexts({"component": preset}):
            map_path = namer.path(map_path_template)
            smoothed = map_handler.read(map_path)[0]
            plt.figure(dpi=300)
            hp.mollview(smoothed, 
                        xsize=2400,
                        title=f"{preset} component", 
                        unit="$\mu K_\\text{CMB}$", 
                        cmap="gist_grey",
                        hold=True, 
                        **plotting_params[preset])
            fig_path = namer.path(fig_path_template)
            plt.savefig(fig_path)
            plt.close()


if __name__ == "__main__":
    main()
