Note that the CMB-ML detector FWHM's were determined as follows:

- The Planck official FWHM's have the lowest FWHM for the 857 GHz detector, with a value of 4.638 arcmin.
- The output Nside for our simulation is 512.
- An output Nside of 512 has resolution 6.871 arcmin.
- We want to have at least 3 pixels for the beam diameter, so we use 20.6 arcmin for the 857 GHz FWHM.
- We consider 20.6 / 4.638 as a scale factor for the remaining beam FWHM.
- In this way we get values (in the table) for 100 - 545 GHz detectors.
- Applying the same scale to the 30, 44, 70 GHz FWHM gives very large values: 58, 124, and 147 arcmins. We want to provide patch-sized maps of 10 degrees by 10 degrees. The largest of those is ~2.5 degrees, which we believe to be too large, relative to the size of the patches. Because we want varied FWHM's, but also want to have more reasonable scales, we arbitrarily set FWHM's to 55, 65, and 75 arcmin (up to a maximum of 1.25 degree).
