import numpy as np


def format_decimal_places(value, decimal_places):
    """Format the number to the specified number of decimal places."""
    formatted = f"{value:.{decimal_places}f}"
    return formatted

def format_sci(value, exp, decimal_places):
    v = value / 10**exp
    formatted = f"{v:.{decimal_places-1}f}"
    return formatted

def format_mean_std(mean, std, sig_digs=4, latex=False):
    """Format mean and standard deviation to the same precision, using scientific notation if needed."""
    sci_low_threshold = 0.001
    sci_high_threshold = 1000
    pm = r"\pm" if latex else "+/-"
    exp_str = r"\times 10^" if latex else "e"

    a, b = (mean, std) if np.abs(mean) > np.abs(std) else (std, mean)
    if (abs(a) < sci_low_threshold or abs(a) > sci_high_threshold):
        # determine the exponent to use for scientific notation based on the larger value
        exp = np.log10(abs(a))
        exp = np.floor(exp)
        formatted_mean = format_sci(mean, exp, sig_digs)
        formatted_std = format_sci(std, exp, sig_digs)
        val_str = f"({formatted_mean} {pm} {formatted_std}){exp_str}{int(exp)}"
    else:
        # Calculate decimal places based on the larger value's significant figures
        significant_figure_place = int(np.floor(np.log10(abs(a)))) - (sig_digs - 1)
        decimal_places = -significant_figure_place if significant_figure_place < 0 else 0
        formatted_mean = format_decimal_places(mean, decimal_places)
        formatted_std = format_decimal_places(std, decimal_places)
        val_str = f"{formatted_mean} {pm} {formatted_std}"

    return f"${val_str}$" if latex else val_str


if __name__ == "__main__":
    try_combos = [
        (12345, 12345),
        (1234.5, 1234.5),
        (123.45, 123.45),
        (12.345, 12.345),
        (1.2345, 1.2345),
        (0.12345, 0.12345),
        (0.012345, 0.012345),
        (0.0012345, 0.0012345),
        (0.00012345, 0.00012345),
        (0.0012345, 0.00012345),
        (0.000012345, 0.00012345),
        (999, 0.1)
    ]

    for combo in try_combos:
        a = (combo[0], combo[1])
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
        a = (combo[0]*10, combo[1])
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
        a = (combo[0], combo[1]*10)
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
        a = (-combo[0], combo[1])
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
        a = (combo[0]*10, -combo[1])
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
        a = (-combo[0], -combo[1]*10)
        print(f"{str(a):<40}:", format_mean_std(a[0], a[1], sig_digs=3, latex=False))
