# This is a reimplementation of the pysm3.ModifiedBlackBodyRealization object.
#   It enables the user to replace components more easily, instead of 
#   recreating a Sky object and all the attendant baggage (run-time).
# This is simply the last line of the __init__ method, extracted, 
#   as a stand-alone function.


from pysm3.models import ModifiedBlackBodyRealization


class FlexDust(ModifiedBlackBodyRealization):
    def replace_draw(self, 
                     seeds,
                     synalm_lmax:int = None):
        (
            self.I_ref,
            self.Q_ref,
            self.U_ref,
            self.mbb_index,
            self.mbb_temperature,
        ) = self.draw_realization(synalm_lmax, seeds)
