from pysm3.models.dust import ModifiedBlackBody
import pysm3.units as u

class FunkyDust(ModifiedBlackBody):
    def __init__(
        self,
        map_I,
        freq_ref_I,
        freq_ref_P,
        map_mbb_index,
        map_mbb_temperature,
        nside,
        scale_fix,

        max_nside=None,
        available_nside=None,
        # map_dist=None,
        unit_mbb_temperature=None,
    ):
        super().__init__(
                         map_I=map_I,
                         freq_ref_I=freq_ref_I,
                         freq_ref_P=freq_ref_P,
                         map_mbb_index=map_mbb_index,
                         map_mbb_temperature=map_mbb_temperature,
                         nside=nside,
                         max_nside=max_nside,
                         unit_mbb_temperature=unit_mbb_temperature,
                         available_nside=available_nside,
                        #  map_dist=map_dist,
                         )
        # TODO: Use full equation instead of "scale_fix"
        #       Temporary. But I don't know if PySM3 ever uses other reference frequencies ??
        eq = u.cmb_equivalencies(u.Quantity(freq_ref_I))
        I_ref = self.I_ref.to(u.MJy / u.sr, equivalencies=eq)
        I_ref = I_ref * scale_fix
        self.I_ref = I_ref.to(u.uK_RJ, equivalencies=eq)
