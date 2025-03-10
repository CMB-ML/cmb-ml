import cmbfscnn.CNN_models as cmbfscnn_models
from cmbml.utils.suppress_print import SuppressPrint

def make_cmbfscnn(level=3, in_channels=None, out_channels=1, n_feats=16):
    if level == 3:
        model_class = cmbfscnn_models.CMBFSCNN_level3
        if in_channels == None:
            in_channels = 10
    elif level == 4:
        model_class = cmbfscnn_models.CMBFSCNN_level4
        if in_channels == None:
            in_channels = 8

    with SuppressPrint():
        net = model_class(in_channels, out_channels, n_feats)
    return net