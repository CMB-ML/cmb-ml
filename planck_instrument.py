import numpy as np
from pathlib import Path
from astropy.table import QTable
from astropy.io import fits
import healpy as hp
import pysm3
import pysm3.units as u
from astropy.cosmology import Planck15


ref_map_files = {
    # 30: "LFI_SkyMap_030-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    # 44: "LFI_SkyMap_044-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    # 70: "LFI_SkyMap_070-BPassCorrected-field-IQU_1024_R3.00_full.fits",
    30: "LFI_SkyMap_030-BPassCorrected_1024_R3.00_full.fits",
    44: "LFI_SkyMap_044-BPassCorrected_1024_R3.00_full.fits",
    70: "LFI_SkyMap_070-BPassCorrected_1024_R3.00_full.fits",
    100: "HFI_SkyMap_100_2048_R3.01_full.fits",
    143: "HFI_SkyMap_143_2048_R3.01_full.fits",
    217: "HFI_SkyMap_217_2048_R3.01_full.fits",
    353: "HFI_SkyMap_353-psb_2048_R3.01_full.fits",
    545: "HFI_SkyMap_545_2048_R3.01_full.fits",
    857: "HFI_SkyMap_857_2048_R3.01_full.fits"
}
for k in list(ref_map_files.keys()):
    ref_map_files[str(k)] = ref_map_files[k]


class PlanckInstrument:
    def __init__(self, 
                 nside, 
                 fidu_instrument_dir="planck_assets",
                 beams_info_filename="planck_deltabandpass.tbl",
                 local_files_dir="local_files",
                 noise_subdir="noise") -> None:
        self.fidu_instrument_dir = Path(fidu_instrument_dir)
        self.local_files_path = Path(local_files_dir)
        self.noise_sd_subdir = self.local_files_path / noise_subdir
        if not self.noise_sd_subdir.exists():
            self.noise_sd_subdir.mkdir(exist_ok=True, parents=True)

        planck_beam_info = QTable.read(self.fidu_instrument_dir / beams_info_filename, format="ascii.ipac")
        planck_beam_info.add_index("band")
        
        self.detectors = {}
        self.ds = self.detectors
        for band in planck_beam_info["band"]:
            center_frequency = planck_beam_info.loc[band]["center_frequency"]
            fwhm = planck_beam_info.loc[band]["fwhm"]
            map_fn = ref_map_files[band]
            if band in ["545", "857"]:
                map_fields = "T"
            else:
                map_fields = "TQU"
            self.detectors[band] = PlanckDetector(
                instrument=self,
                nom_freq=band,
                cen_freq=center_frequency,
                fwhm=fwhm,
                ref_map_fn=map_fn,
                ref_map_fields=map_fields,
                nside=nside
            )
            self.detectors[int(band)] = self.detectors[band]


class PlanckDetector:
    def __init__(self, instrument: PlanckInstrument, nom_freq, cen_freq, fwhm, ref_map_fn, ref_map_fields, nside):
        self.instrument = instrument
        self.nominal_frequency = nom_freq
        self.center_frequency = cen_freq
        self.fwhm = fwhm
        self.ref_map_fn = instrument.fidu_instrument_dir / ref_map_fn
        self.ref_map_fields = ref_map_fields
        self.nside = nside
        if ref_map_fields == "T":
            self.var_maps = {"T": None}
        else:
            self.var_maps = {"T": None, "Q": None, "U": None}
    
    def get_noise_map(self, field, rng: np.random.Generator, force_overwrite=False):
        if isinstance(field, str):
            field_str = field
        else:
            raise TypeError(f"Field must be a string. Got {field}, expected 'T', 'Q', or 'U'.")
        sd_map_path = self.get_sd_map_path(field_str)
        if self.var_maps[field] is None:
            if sd_map_path.exists() and not force_overwrite:
                sd_map = hp.read_map(sd_map_path)
            else:
                sd_map = self.make_sd_map(field_str, sd_map_path)
        noise_map = rng.normal(scale=sd_map, size=sd_map.size)
        noise_map = u.Quantity(noise_map, u.K_CMB, copy=False)
        noise_map = noise_map.to(u.uK_RJ, equivalencies=u.cmb_equivalencies(self.center_frequency))
        # noise_map = noise_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(self.center_frequency))
        return noise_map

    def get_sd_map_path(self, field_str):
        if self.nominal_frequency in ["30", "44", "70"]:
            inst = "LFI"
        elif self.nominal_frequency in ["100", "143", "217", "353", "545", "857"]:
            inst = "HFI"
        else:
            raise ValueError(f"Can't determine detector from {self.nominal_frequency}")
        fn = f"sd_map_{inst}_{self.nominal_frequency}_{field_str}_{self.nside}.fits"
        return self.instrument.noise_sd_subdir / fn
    
    def make_sd_map(self, field_str, out_filepath):
        fn = self.ref_map_fn
        field_idx = self.lookup_ref_map_var_field(field_str)
        if not fn.exists():
            raise FileNotFoundError(f"Cannot find '{fn}'")

        # Can't use PySM3's read_map() function
        # I don't know how to make the units "(K_CMB)^2" parseable by astropy's units module
        with fits.open(fn) as hdul:
            # TODO: This is based on poor assumptions. Need to check if the field actually
            #       exists in the file. 
            try:
                field_num = field_idx + 1
                # Need to use correct hdul_number instead of assuming 1?
                unit = hdul[1].header[f"TUNIT{field_num}"]
            except KeyError:
                unit = ""
                # TODO: Use logging
                # log.warning("No physical unit associated with file %s", str(path))
        ok_units_k_cmb = ["(K_CMB)^2", "Kcmb^2"]
        ok_units_mjysr = ["(Mjy/sr)^2"]
        ok_units = [*ok_units_k_cmb, *ok_units_mjysr]
        # TODO: Use logging
        # log.info(f"Units for map {self.ref_map_fn} are {unit}")
        if unit not in ok_units:
            raise ValueError(f"Wrong unit found in fits file. Found {unit}, expected one of {ok_units}.")
        if unit in ok_units_k_cmb:
            unit_sqrt = "K_CMB"
        else:
            unit_sqrt = "MJy/sr"

        m = hp.read_map(fn, field=field_idx)

        # From PySM3 template.py's read_map function, with minimal alteration:
        dtype = m.dtype
        # numba only supports little endian
        if dtype.byteorder == ">":
            dtype = dtype.newbyteorder()
        # mpi4py has issues if the dtype is a string like ">f4"
        if dtype == np.dtype(np.float32):
            dtype = np.dtype(np.float32)
        elif dtype == np.dtype(np.float64):
            dtype = np.dtype(np.float64)
        nside_in = hp.get_nside(m)
        if self.nside < nside_in:  # do downgrading in double precision
            m = hp.ud_grade(m.astype(np.float64), nside_out=self.nside)
        elif self.nside > nside_in:
            m = hp.ud_grade(m, nside_out=self.nside)
        m = m.astype(dtype, copy=False)
        # End of code from PySM3 template.py

        # TODO: (Physics) Please check my assumption.
        # I assume variance was calculated for K_CMB, but not sure.
        # If you think it was calculated for uK_CMB, uncomment the next two lines.
        # if unit_sqrt == "K_CMB":
        #     m *= 1e6
        
        # Convert variance to standard deviation, 
        #   to be used in np.random.Generator.normal as "scale" parameter
        m = np.sqrt(m)

        # Convert MJy/sr to K_CMB (I think, TODO: Verify)
        # This is an oversimplification applied to the 545 and 857 GHz bands
        # something about "very sensitive to band shape" for sub-mm bands (forgotten source)
        # This may be a suitable first-order approximation
        if unit_sqrt == "MJy/sr":
            m = (m * u.MJy / u.sr).to(
                u.K, equivalencies=u.thermodynamic_temperature(self.center_frequency, Planck15.Tcmb0)
            ).value

        col_names = {"T": "II", "Q": "QQ", "U": "UU"}
        hp.write_map(filename=str(out_filepath),
                     m=m,
                     nest=False,
                     column_names=[col_names[field_str]],
                     column_units=["K_CMB"],
                     dtype=m.dtype,
                     overwrite=True
                    # TODO: figure out how to add a comment to hp's map... or switch with astropy equivalent 
                    #  extra_header=f"Variance map pulled from {self.ref_map_fn}, {col_names[field_str]}"
                     )
        return m

    def lookup_ref_map_var_field(self, field_str):
        if self.ref_map_fields == "T":
            if field_str == "T":
                field_idx = 2
            else:
                raise ValueError("Detector only has 'T' field.")
        elif self.ref_map_fields == "TQU":
            if field_str == "T":
                field_idx = 4
            elif field_str == "Q":
                field_idx = 7
            elif field_str == "U":
                field_idx = 9
            else:
                raise ValueError(f"Field is {field_str}, expected 'T', 'Q', or 'U'")
        else:
            raise ValueError(f"Unexpected detector fields found: {self.ref_map_fields}")
        return field_idx    


def main():
    msg = "This is not the file you intended to run."
    print(msg)
    raise Exception(msg)
    planck = PlanckInstrument()
    print(planck.detectors[70].fwhm)
    print(planck.ds["70"].fwhm)


def get_info_from_planck_deltabandpass_demo():
    # From https://mapsims.readthedocs.io/en/latest/usage.html#simulate-other-instruments
    planck_beam_info = QTable.read("planck_deltabandpass.tbl", format="ascii.ipac")
    print(planck_beam_info.colnames)  # ['band', 'center_frequency', 'fwhm']
    
    print(planck_beam_info)
    # result: """
    # band center_frequency     fwhm    
    #         GHz           arcmin   
    # ---- ---------------- ------------
    #  30             28.4 33.102652125
    #  44             44.1  27.94348615
    #  70             70.4  13.07645961
    # 100           100.89        9.682
    # 143          142.876        7.303
    # 217          221.156        5.021
    # 353            357.5        4.944
    # 545            555.2        4.831
    # 857            866.8        4.638
    # """

    planck_beam_info.add_index("band")
    bands = list(planck_beam_info["band"])
    print(bands)
    print(planck_beam_info.loc["70"]["fwhm"])


if __name__ == "__main__":
    main()