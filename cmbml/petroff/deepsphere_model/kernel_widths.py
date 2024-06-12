# Copied from https://github.com/deepsphere/paper-iclr20/blob/51260d9169b9bff2f0d71d567c99909a17efd5e9/figures/kernel_widths.py
# This was used in a slightly different form in Petroff's work, but should be functionally equivalent.
# Modified to be a function
import numpy as np


def kernel_width_optimal(query_nside):
    # I do not fully understand the derivation of the following values.
    # The appear to be from https://github.com/deepsphere/paper-deepsphere-iclr2020/blob/51260d9169b9bff2f0d71d567c99909a17efd5e9/HEALPix_equivariance_error.ipynb
    kernel_width_optimal = {
        32: 0.02500,
        64: 0.01228,
        128: 0.00614,
        256: 0.00307,
        512: 0.00154,
        1024: 0.00077,
    }
    if query_nside not in kernel_width_optimal.keys():
        kernel_wdith_fit = np.poly1d(
            np.polyfit(
                np.log(sorted(kernel_width_optimal.keys())),
                np.log(
                    [i[1] for i in sorted(kernel_width_optimal.items(), key=lambda i: i[0])]
                ),
                1,
            )
        )
        for _nside in [1, 2, 4, 8, 16]:
            # Populate kernel widths for other map sizes
            kernel_width_optimal[_nside] = np.exp(kernel_wdith_fit(np.log(_nside)))
    return kernel_width_optimal[query_nside]
