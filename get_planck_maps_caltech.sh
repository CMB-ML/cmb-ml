#!/bin/sh
#
# NASA/IPAC Infrared Science Archive
# Planck Data Release 3: All-Sky Maps
#
# Edit this script to select which files to download
#

#-----------------------------------
# Primary maps
#-----------------------------------

# Full mission maps

curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_353_2048_R3.01_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_full.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_full.fits

# Nominal mission maps

# Nominal
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_nominal.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_nominal.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_nominal.fits

# Multi-survey maps

# Surveys 1-3-5-6-7-8
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-1-3-5-6-7-8.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-1-3-5-6-7-8.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-1-3-5-6-7-8.fits

# Single-survey maps

# Survey 1
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-1.fits

# Survey 2
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-2.fits

# Survey 3
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-3.fits

# Survey 4
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-4.fits

# Survey 6
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-6.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-6.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-6.fits

# Survey 7
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-7.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-7.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-7.fits

# Survey 8
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-8.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-8.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-8.fits

# Survey 9
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_survey-9.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_survey-9.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_survey-9.fits

# Single-year maps

# Year 1
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-1.fits

# Year 2
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-2.fits

# Year 3
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-3.fits

# Year 4
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-4.fits

# Multi-year maps

# Years 1+2
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-1-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-1-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-1-2.fits

# Years 3+4
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-3-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-3-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-3-4.fits

# Years 1+3
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-1-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-1-3.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-1-3.fits

# Years 2+4
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_030_1024_R3.00_year-2-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_044_1024_R3.00_year-2-4.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/LFI_SkyMap_070_1024_R3.00_year-2-4.fits

# Half-mission maps

# First half
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_halfmission-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_halfmission-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_halfmission-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_353_2048_R3.01_halfmission-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_halfmission-1.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_halfmission-1.fits

# Second half
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_100_2048_R3.01_halfmission-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_143_2048_R3.01_halfmission-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_217_2048_R3.01_halfmission-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_353_2048_R3.01_halfmission-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_545_2048_R3.01_halfmission-2.fits
curl -f -O https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_857_2048_R3.01_halfmission-2.fits

# End of script