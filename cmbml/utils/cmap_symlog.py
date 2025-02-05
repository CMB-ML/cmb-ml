from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import healpy as hp
from pysm3.units import Unit

from .cmap_interp import get_symlog_cmap
from .planck_cmap import colombi1_cmap
from .number_plot_ticks import make_tick_labels


@dataclass
class SymLogPlotSettings:
    vmin: float
    vmax: float
    linthresh: float
    linscale: float
    unit: Optional[Union[Unit, str]]
    figsize: Optional[Tuple[float, float]] = None
    ticks: Optional[List[float]] = None



def single_symlog_map_fig(map_data,
                          symlog_settings: SymLogPlotSettings,
                          title=None,
                          ):
    if symlog_settings.figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=symlog_settings.figsize)

    norm = cm.colors.SymLogNorm(vmin=symlog_settings.vmin, 
                                vmax=symlog_settings.vmax, 
                                linthresh=symlog_settings.linthresh, 
                                linscale=symlog_settings.linscale)
    symlog_cmap = get_symlog_cmap(colombi1_cmap, norm)
    plot_params = dict(min=0, 
                       max=1, 
                       unit=symlog_settings.unit, 
                       cbar=False, 
                       cmap=symlog_cmap,
                       hold=True)

    try:
        mask = map_data.mask
        normed_map = norm(map_data)
        normed_map = hp.ma(normed_map)
        normed_map.mask = mask
    except AttributeError:
        normed_map = norm(map_data)

    hp.mollview(normed_map, 
                title=title, 
                **plot_params)

    cax = fig.add_axes([0.2, 0.2, 0.6, 0.02])
    # cax = fig.add_subplot(gs[n_rows, :])
    mappable = cm.ScalarMappable(norm=norm, cmap=symlog_cmap)
    cb = plt.colorbar(mappable, cax=cax, orientation='horizontal', extend='both')

    if symlog_settings.ticks is not None:
        cb.set_ticks(symlog_settings.ticks)
        cb.set_ticklabels(make_tick_labels(symlog_settings.ticks))

    if isinstance(symlog_settings.unit, Unit):  # If it's just an Astropy Unit
        cb_label = "$\\Delta \\text{T}$ " + symlog_settings.unit.to_string('latex')
    elif symlog_settings.unit:  # If it's some other string (or anything else)
        cb_label = symlog_settings.unit
    else:  # Assume Delta T in uK_CMB
        cb_label= '$\\Delta \\text{T} \\; [\\mu \\text{K}_\\text{CMB}]$'
    cb.set_label(cb_label)

    linthresh_pos = norm(symlog_settings.linthresh)
    linthresh_neg = norm(-symlog_settings.linthresh)

    # cax.text(linthresh_neg, 0.5, '$\\![$', fontsize=14, ha='center', va='center', transform=cax.transAxes)
    cax.plot([linthresh_neg], [0.5], '4', color='black', markersize=8, transform=cax.transAxes, clip_on=False)
    # cax.text(linthresh_pos, 0.5, '$\\!]$', fontsize=14, ha='center', va='center', transform=cax.transAxes)
    cax.plot([linthresh_pos], [0.5], '3', color='black', markersize=8, transform=cax.transAxes, clip_on=False)

    return plt


def many_symlog_map_fig(map_data,
                        symlog_settings: SymLogPlotSettings,
                        dest,
                        title=None,
                        ):
    norm = cm.colors.SymLogNorm(vmin=symlog_settings.vmin, 
                                vmax=symlog_settings.vmax, 
                                linthresh=symlog_settings.linthresh, 
                                linscale=symlog_settings.linscale)
    symlog_cmap = get_symlog_cmap(colombi1_cmap, norm)
    plot_params = dict(min=0, 
                       max=1, 
                       unit=symlog_settings.unit, 
                       cbar=False, 
                       cmap=symlog_cmap,
                       hold=True)
    plt.axes(dest)

    try:
        mask = map_data.mask
        normed_map = norm(map_data)
        normed_map = hp.ma(normed_map)
        normed_map.mask = mask
    except AttributeError:
        normed_map = norm(map_data)

    hp.mollview(normed_map, 
                title=title, 
                **plot_params)


def many_symlog_map_add_cbar(symlog_settings: SymLogPlotSettings, fig):
    norm = cm.colors.SymLogNorm(vmin=symlog_settings.vmin, 
                                vmax=symlog_settings.vmax, 
                                linthresh=symlog_settings.linthresh, 
                                linscale=symlog_settings.linscale)
    symlog_cmap = get_symlog_cmap(colombi1_cmap, norm)

    cax = fig.add_axes([0.2, 0.25, 0.6, 0.02])
    # cax = fig.add_subplot(gs[n_rows, :])
    mappable = cm.ScalarMappable(norm=norm, cmap=symlog_cmap)
    cb = plt.colorbar(mappable, cax=cax, orientation='horizontal', 
                      extend='both', extendfrac=0.02)

    if symlog_settings.ticks is not None:
        cb.set_ticks(symlog_settings.ticks)
        cb.set_ticklabels(make_tick_labels(symlog_settings.ticks))

    if isinstance(symlog_settings.unit, Unit):  # If it's just an Astropy Unit
        cb_label = "$\\Delta \\text{T}$ " + symlog_settings.unit.to_string('latex')
    elif symlog_settings.unit:  # If it's some other string (or anything else)
        cb_label = symlog_settings.unit
    else:  # Assume Delta T in uK_CMB
        cb_label= '$\\Delta \\text{T} \\; [\\mu \\text{K}_\\text{CMB}]$'
    cb.set_label(cb_label)

    linthresh_pos = norm(symlog_settings.linthresh)
    linthresh_neg = norm(-symlog_settings.linthresh)

    # cax.text(linthresh_neg, 0.5, '$\\![$', fontsize=14, ha='center', va='center', transform=cax.transAxes)
    cax.plot([linthresh_neg], [0.5], '4', color='black', markersize=20, transform=cax.transAxes, clip_on=False)
    # cax.text(linthresh_pos, 0.5, '$\\!]$', fontsize=14, ha='center', va='center', transform=cax.transAxes)
    cax.plot([linthresh_pos], [0.5], '3', color='black', markersize=20, transform=cax.transAxes, clip_on=False)
