from pathlib import Path
import numpy as np
import pandas as pd

import logging


logger = logging.getLogger(__name__)


def add_missing_multipoles(df, path_label=None, max_ell=None):
    # If max_ell not provided, set it from the data
    # Truncate if needed
    # Figure out needed ells
    # If there's missing stuff:
        # Warn about 
        #   Missing monopole/dipole
        #   Missing interior ells
        #   Missing extension up to max_ell
        # Insert all missing rows
    # Reset the index to clean up

    df['L'] = df['L'].astype(int)
    original_max = df['L'].max()

    # If max_ell not provided, set it from the data
    if max_ell is None:
        max_ell = original_max

    # Truncate if needed
    if original_max > max_ell:
        df = df[df['L'] <= max_ell]

    # Figure out needed ells
    all_L = np.arange(0, max_ell + 1)
    missing = sorted(set(all_L) - set(df['L']))

    if missing:
        path_str = f" in {path_label}" if path_label is not None else ""
        # Missing monopole/dipole
        low = [L for L in missing if L in {0, 1}]
        if low:
            logger.warning(f"Missing monopole/dipole {low}{path_str}, inserting zeros.")

        # Missing interior ells
        interior = [L for L in missing if 2 <= L <= original_max]
        if interior:
            logger.warning(f"Missing interior multipoles {interior}{path_str}, inserting zeros.")

        # Missing extension up to max_ell
        extension = [L for L in missing if L > original_max]
        if extension:
            logger.warning(f"Extending{path_str} up to ell={max_ell}, inserting zeros for {extension[0]} to {extension[-1]}.")

        # Insert all missing rows
        zero_rows = pd.DataFrame(
            [[L] + [0.0] * (len(df.columns) - 1) for L in missing],
            columns=df.columns,
        )
        df = pd.concat([df, zero_rows], ignore_index=True)

    # Finalize
    df = df.sort_values('L').reset_index(drop=True)
    return df


def add_PT_PE_and_reorder(ps_df, path_label):
    if 'PT' not in ps_df.columns:
        ps_df['PT'] = 0.0
        if path_label:
            logger.info(f"Added PT column of zeros for {path_label}")

    if 'PE' not in ps_df.columns:
        ps_df['PE'] = 0.0
        if path_label:
            logger.info(f"Added PE column of zeros for {path_label}")

    # CAMB expected order
    camb_order = ["L", "TT", "EE", "BB", "TE", "PP", "PT", "PE"]

    # Keep only what exists in df (handles Planck vs CAMB gracefully)
    cols_in_order = [col for col in camb_order if col in ps_df.columns]
    
    # Reorder + append any extra columns at the end
    other_cols = [col for col in ps_df.columns if col not in cols_in_order]
    ps_df = ps_df[cols_in_order + other_cols]

    return ps_df

