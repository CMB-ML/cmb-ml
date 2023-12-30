#!/bin/bash

# Flag to determine if a separate data directory is used
USE_SEPARATE_DATA_DIR=true

# Define local variables for directories
DATA_DIR="/Users/jimcamato/Data/SPACE"
REPO_DIR="/Users/jimcamato/Developer/ml_cmb_pysm_sims/planck_assets"

# Check if the necessary directories exist, exit if they don't
if $USE_SEPARATE_DATA_DIR && [ ! -d "$DATA_DIR" ]; then
    echo "Data directory $DATA_DIR does not exist. Exiting."
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    echo "Repository directory $REPO_DIR does not exist. Exiting."
    exit 1
fi

# Common URL part
BASE_URL="https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps"

# File names
FILES=(
    # "LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits"
    "LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits"
    # "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits"
    # "HFI_SkyMap_100_2048_R3.01_full.fits"
    # "HFI_SkyMap_143_2048_R3.01_full.fits"
    # "HFI_SkyMap_217_2048_R3.01_full.fits"
    # "HFI_SkyMap_353-psb_2048_R3.01_full.fits"
    # "HFI_SkyMap_545_2048_R3.01_full.fits"
    # "HFI_SkyMap_857_2048_R3.01_full.fits"
)

# Loop over the files, download them and create symlinks if necessary
for file in "${FILES[@]}"; do
    if $USE_SEPARATE_DATA_DIR; then
        # Download the file to the data directory
        wget "$BASE_URL/$file" -P "$DATA_DIR"

        # Check if download was successful
        if [ -f "$DATA_DIR/$file" ]; then
            # Create a symlink in the repository directory
            ln -sf "$DATA_DIR/$file" "$REPO_DIR/$file"
        else
            echo "Failed to download $file"
        fi
    else
        # Download the file directly to the repository directory
        wget "$BASE_URL/$file" -P "$REPO_DIR"
        if [ ! -f "$REPO_DIR/$file" ]; then
            echo "Failed to download $file"
        fi
    fi
done
