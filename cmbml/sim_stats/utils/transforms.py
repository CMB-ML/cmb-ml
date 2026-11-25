from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from cmbml.sim_stats.utils.stat_convention import (
    STAT_MIN, 
    STAT_MAX, 
    STAT_MEAN, 
    STAT_STDV
    )


class _BaseTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _check_tensor(x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError(
                f"Expected torch.Tensor, got {type(x).__name__}."
            )

        if x.ndim < 2:
            raise ValueError(
                f"Expected tensor with shape [C, ...], got {tuple(x.shape)}."
            )


class GlobalMeanStd(_BaseTransform):
    """
    Per-channel global mean/std standardization.

    Expected input shape:
        [C, ...]
    """

    def __init__(self, means, stds):
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)

        if means.ndim != 1:
            raise ValueError(
                f"means must be 1-D, got shape {means.shape}."
            )

        if stds.ndim != 1:
            raise ValueError(
                f"stds must be 1-D, got shape {stds.shape}."
            )

        if means.shape != stds.shape:
            raise ValueError(
                f"means and stds must have matching shapes, "
                f"got {means.shape} and {stds.shape}."
            )

        if np.any(~np.isfinite(means)):
            raise ValueError("means contains non-finite values.")

        if np.any(~np.isfinite(stds)):
            raise ValueError("stds contains non-finite values.")

        if np.any(stds <= 0):
            raise ValueError(
                "All standard deviations must be positive."
            )

        self.means = torch.from_numpy(means).reshape(-1, 1)
        self.stds = torch.from_numpy(stds).reshape(-1, 1)

    @property
    def n_channels(self):
        return self.means.shape[0]

    def _stats_for(self, x):
        self._check_tensor(x)

        if x.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got tensor shape {tuple(x.shape)}."
            )

        means = self.means.to(
            device=x.device,
            dtype=x.dtype,
        )
        stds = self.stds.to(
            device=x.device,
            dtype=x.dtype,
        )

        return means, stds

    def __call__(self, x):
        means, stds = self._stats_for(x)
        return (x - means) / stds

    def inverse(self, x):
        means, stds = self._stats_for(x)
        return x * stds + means


class GlobalMinMax(_BaseTransform):
    """
    Per-channel global min/max scaling to [0, 1].

    Expected input shape:
        [C, ...]
    """

    def __init__(self, mins, maxs):
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)

        if mins.ndim != 1:
            raise ValueError(
                f"mins must be 1-D, got shape {mins.shape}."
            )

        if maxs.ndim != 1:
            raise ValueError(
                f"maxs must be 1-D, got shape {maxs.shape}."
            )

        if mins.shape != maxs.shape:
            raise ValueError(
                f"mins and maxs must have matching shapes, "
                f"got {mins.shape} and {maxs.shape}."
            )

        if np.any(~np.isfinite(mins)):
            raise ValueError("mins contains non-finite values.")

        if np.any(~np.isfinite(maxs)):
            raise ValueError("maxs contains non-finite values.")

        if np.any(maxs <= mins):
            raise ValueError(
                "Every maximum must be greater than its minimum."
            )

        self.mins = torch.from_numpy(mins).reshape(-1, 1)
        self.maxs = torch.from_numpy(maxs).reshape(-1, 1)

    @property
    def n_channels(self):
        return self.mins.shape[0]

    def _stats_for(self, x):
        self._check_tensor(x)

        if x.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got tensor shape {tuple(x.shape)}."
            )

        mins = self.mins.to(
            device=x.device,
            dtype=x.dtype,
        )
        maxs = self.maxs.to(
            device=x.device,
            dtype=x.dtype,
        )

        return mins, maxs

    def __call__(self, x):
        mins, maxs = self._stats_for(x)
        return (x - mins) / (maxs - mins)

    def inverse(self, x):
        mins, maxs = self._stats_for(x)
        return x * (maxs - mins) + mins


class LocalMeanStd(_BaseTransform):
    """
    Per-channel, per-pixel mean/std standardization.

    Statistics must have shape:
        [C, Npix]

    Expected input shape:
        [C, Npix]
    """

    def __init__(self, means, stds):
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)

        if means.ndim != 2:
            raise ValueError(
                f"means must have shape [C, Npix], got {means.shape}."
            )

        if stds.ndim != 2:
            raise ValueError(
                f"stds must have shape [C, Npix], got {stds.shape}."
            )

        if means.shape != stds.shape:
            raise ValueError(
                f"means and stds must have matching shapes, "
                f"got {means.shape} and {stds.shape}."
            )

        if np.any(~np.isfinite(means)):
            raise ValueError("means contains non-finite values.")

        if np.any(~np.isfinite(stds)):
            raise ValueError("stds contains non-finite values.")

        if np.any(stds <= 0):
            raise ValueError(
                "All local standard deviations must be positive."
            )

        self.means = torch.from_numpy(means)
        self.stds = torch.from_numpy(stds)

    @property
    def shape(self):
        return tuple(self.means.shape)

    def _stats_for(self, x):
        self._check_tensor(x)

        if x.ndim != 2:
            raise ValueError(
                f"Expected [C, Npix], got {tuple(x.shape)}."
            )

        if tuple(x.shape) != self.shape:
            raise ValueError(
                f"Expected shape {self.shape}, got {tuple(x.shape)}."
            )

        means = self.means.to(
            device=x.device,
            dtype=x.dtype,
        )
        stds = self.stds.to(
            device=x.device,
            dtype=x.dtype,
        )

        return means, stds

    def __call__(self, x):
        means, stds = self._stats_for(x)
        return (x - means) / stds

    def inverse(self, x):
        means, stds = self._stats_for(x)
        return x * stds + means


class LocalMinMax(_BaseTransform):
    """
    Per-channel, per-pixel min/max scaling to [0, 1].

    Statistics must have shape:
        [C, Npix]

    Expected input shape:
        [C, Npix]
    """

    def __init__(self, mins, maxs):
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)

        if mins.ndim != 2:
            raise ValueError(
                f"mins must have shape [C, Npix], got {mins.shape}."
            )

        if maxs.ndim != 2:
            raise ValueError(
                f"maxs must have shape [C, Npix], got {maxs.shape}."
            )

        if mins.shape != maxs.shape:
            raise ValueError(
                f"mins and maxs must have matching shapes, "
                f"got {mins.shape} and {maxs.shape}."
            )

        if np.any(~np.isfinite(mins)):
            raise ValueError("mins contains non-finite values.")

        if np.any(~np.isfinite(maxs)):
            raise ValueError("maxs contains non-finite values.")

        if np.any(maxs <= mins):
            raise ValueError(
                "Every local maximum must be greater than its minimum."
            )

        self.mins = torch.from_numpy(mins)
        self.maxs = torch.from_numpy(maxs)

    @property
    def shape(self):
        return tuple(self.mins.shape)

    def _stats_for(self, x):
        self._check_tensor(x)

        if x.ndim != 2:
            raise ValueError(
                f"Expected [C, Npix], got {tuple(x.shape)}."
            )

        if tuple(x.shape) != self.shape:
            raise ValueError(
                f"Expected shape {self.shape}, got {tuple(x.shape)}."
            )

        mins = self.mins.to(
            device=x.device,
            dtype=x.dtype,
        )
        maxs = self.maxs.to(
            device=x.device,
            dtype=x.dtype,
        )

        return mins, maxs

    def __call__(self, x):
        mins, maxs = self._stats_for(x)
        return (x - mins) / (maxs - mins)

    def inverse(self, x):
        mins, maxs = self._stats_for(x)
        return x * (maxs - mins) + mins


def make_map_transforms(
    scaling: str,
    freqs: Sequence,
    *,
    scalar_stats: Mapping | None = None,
    local_feature_stats: Mapping | None = None,
    local_target_stats=None,
):
    """
    Construct feature and target transforms for the requested scaling mode.

    Parameters
    ----------
    scaling
        One of:
            "meanstdv_global"
            "meanstdv_local"
            "minmax_global"
            "minmax_local"

    freqs
        Frequency/channel identifiers in model-input order.

    scalar_stats
        Parsed scalar_stats.yaml mapping. Required for global modes.
        Frequency keys may be strings or integers.

    local_feature_stats
        Mapping from frequency key to local-statistics array.
        Frequency keys may be strings or integers.
        Each array must have shape [N_STATS, Npix].
        Required for local modes.
        Pixel ordering must match that of the maps being transformed.

    local_target_stats
        Local CMB statistics array with shape [N_STATS, Npix].
        Required for local modes.

    Returns
    -------
    feature_transform, target_transform
    """

    if scaling == "meanstdv_global":
        _require_scalar_stats(
            scalar_stats,
            required_keys=[*freqs, "cmb"],
        )

        feature_transform = GlobalMeanStd(
            means=[
                _get_stat_entry(scalar_stats, freq)["mean"]
                for freq in freqs
            ],
            stds=[
                _get_stat_entry(scalar_stats, freq)["stdv"]
                for freq in freqs
            ],
        )

        cmb_stats = _get_stat_entry(scalar_stats, "cmb")

        target_transform = GlobalMeanStd(
            means=[cmb_stats["mean"]],
            stds=[cmb_stats["stdv"]],
        )

    elif scaling == "minmax_global":
        _require_scalar_stats(
            scalar_stats,
            required_keys=[*freqs, "cmb"],
        )

        feature_transform = GlobalMinMax(
            mins=[
                _get_stat_entry(scalar_stats, freq)["vmin"]
                for freq in freqs
            ],
            maxs=[
                _get_stat_entry(scalar_stats, freq)["vmax"]
                for freq in freqs
            ],
        )

        cmb_stats = _get_stat_entry(scalar_stats, "cmb")

        target_transform = GlobalMinMax(
            mins=[cmb_stats["vmin"]],
            maxs=[cmb_stats["vmax"]],
        )

    elif scaling == "meanstdv_local":
        _require_local_stats(
            local_feature_stats=local_feature_stats,
            local_target_stats=local_target_stats,
            freq_keys=freqs,
        )

        feature_means = np.stack(
            [
                np.asarray(
                    _get_stat_entry(local_feature_stats, freq)
                )[STAT_MEAN]
                for freq in freqs
            ],
            axis=0,
        )

        feature_stds = np.stack(
            [
                np.asarray(
                    _get_stat_entry(local_feature_stats, freq)
                )[STAT_STDV]
                for freq in freqs
            ],
            axis=0,
        )

        target_stats = np.asarray(local_target_stats)

        feature_transform = LocalMeanStd(
            means=feature_means,
            stds=feature_stds,
        )

        target_transform = LocalMeanStd(
            means=target_stats[STAT_MEAN][None, :],
            stds=target_stats[STAT_STDV][None, :],
        )

    elif scaling == "minmax_local":
        _require_local_stats(
            local_feature_stats=local_feature_stats,
            local_target_stats=local_target_stats,
            freq_keys=freqs,
        )

        feature_mins = np.stack(
            [
                np.asarray(
                    _get_stat_entry(local_feature_stats, freq)
                )[STAT_MIN]
                for freq in freqs
            ],
            axis=0,
        )

        feature_maxs = np.stack(
            [
                np.asarray(
                    _get_stat_entry(local_feature_stats, freq)
                )[STAT_MAX]
                for freq in freqs
            ],
            axis=0,
        )

        target_stats = np.asarray(local_target_stats)

        feature_transform = LocalMinMax(
            mins=feature_mins,
            maxs=feature_maxs,
        )

        target_transform = LocalMinMax(
            mins=target_stats[STAT_MIN][None, :],
            maxs=target_stats[STAT_MAX][None, :],
        )

    else:
        raise ValueError(
            "Unknown scaling mode "
            f"{scaling!r}. Expected one of: "
            "'meanstdv_global', "
            "'meanstdv_local', "
            "'minmax_global', "
            "'minmax_local'."
        )

    return feature_transform, target_transform


def _require_scalar_stats(
    scalar_stats,
    required_keys,
):
    if scalar_stats is None:
        raise ValueError(
            "scalar_stats is required for global scaling."
        )

    for key in required_keys:
        _get_stat_entry(scalar_stats, key)


def _get_stat_entry(stats, key):
    """
    Look up a statistics entry allowing equivalent string/integer keys.

    Examples:
        "100" <-> 100
        "143" <-> 143

    Non-numeric keys such as "cmb" are used as-is.
    """
    if key in stats:
        return stats[key]

    if isinstance(key, str):
        try:
            int_key = int(key)
        except ValueError:
            pass
        else:
            if int_key in stats:
                return stats[int_key]

    elif isinstance(key, (int, np.integer)):
        str_key = str(key)
        if str_key in stats:
            return stats[str_key]

    raise KeyError(f"Statistics entry {key!r} not found.")


def _require_local_stats(
    local_feature_stats,
    local_target_stats,
    freq_keys,
):
    if local_feature_stats is None:
        raise ValueError(
            "local_feature_stats is required for local scaling."
        )

    if local_target_stats is None:
        raise ValueError(
            "local_target_stats is required for local scaling."
        )

    for key in freq_keys:
        _validate_local_stat_array(
            _get_stat_entry(local_feature_stats, key),
            name=f"local_feature_stats[{key!r}]",
        )

    _validate_local_stat_array(
        local_target_stats,
        name="local_target_stats",
    )


def _validate_local_stat_array(arr, name):
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must have shape [N_STATS, Npix], "
            f"got {arr.shape}."
        )

    if arr.shape[0] <= STAT_STDV:
        raise ValueError(
            f"{name} must contain at least "
            f"{STAT_STDV + 1} statistic rows, got {arr.shape}."
        )
