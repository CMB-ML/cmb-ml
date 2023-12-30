#!/bin/sh
#
# NASA/IPAC Infrared Science Archive
# Planck Data Release 3: All-Sky Maps
#
# Original script from: https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/index.html
#
# Edit this script to select which files to download and where they go.
#
#!/bin/bash

# Define local variables for directories
DATA_DIR="/Users/jim/Data/SPACE"
# If not symlinking files (downloading directly to planck_assets, remove the line following ### $$$ below (currently line 45)
SYMLINK_DIR="/Users/jim/Developer/ml_cmb/planck_assets"

# Common URL part
BASE_URL="https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps"

# File names
FILES=(
    "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits"
    "LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits"
    "LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits"
    "HFI_SkyMap_100_2048_R3.01_full.fits"
    "HFI_SkyMap_143_2048_R3.01_full.fits"
    "HFI_SkyMap_217_2048_R3.01_full.fits"
    "HFI_SkyMap_353-psb_2048_R3.01_full.fits"
    "HFI_SkyMap_545_2048_R3.01_full.fits"
    "HFI_SkyMap_857_2048_R3.01_full.fits"
)

# Ensure the data directory exists
mkdir -p "$DATA_DIR"

# Loop over the files, download them and create symlinks
for file in "${FILES[@]}"; do
    # Download the file
    curl -f -O "$BASE_URL/$file" -o "$DATA_DIR/$file"

    # Check if download was successful
    if [ -f "$DATA_DIR/$file" ]; then
        # Create a symlink in the symlink directory
        ### $$$ Remove the following line if not using symlinks.
        ln -sf "$DATA_DIR/$file" "$SYMLINK_DIR/$file"
    else
        echo "Failed to download $file"
    fi
done
