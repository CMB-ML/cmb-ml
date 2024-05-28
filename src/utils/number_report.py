import numpy as np


def format_decimal_places(value, decimal_places):
    """Format the number to the specified number of decimal places."""
    formatted_value = f"{value:.{decimal_places}f}"
    return formatted_value


def format_scientific(value, sig_figs, latex=False):
    """Format the number in scientific notation with a given number of significant figures."""
    formatted_value = f"{value:.{sig_figs-1}e}"
    if latex:
        # Convert 'e' to LaTeX exponent format
        base, exponent = formatted_value.split('e')
        formatted_value = f"{base} \\times 10^{{{int(exponent)}}}"
        return f"{formatted_value}"
    return formatted_value


def format_number(value,
                  sig_figs=4,
                  sci_low_threshold=0.001,
                  sci_high_threshold=1000,
                  latex=False):
    """Determine the formatting method based on the magnitude of the number."""
    if abs(value) < sci_low_threshold or abs(value) > sci_high_threshold:
        return format_scientific(value, sig_figs, latex)
    else:
        if value == 0:
            return "$0.0$" if latex else "0.0"
        significant_figure_place = int(np.floor(np.log10(abs(value)))) - (sig_figs - 1)
        decimal_places = -significant_figure_place if significant_figure_place < 0 else 0
        val_str = format_decimal_places(value, decimal_places)
        return f"${val_str}$" if latex else val_str


def format_mean_std(mean, std, sig_figs=4, latex=False):
    """Format mean and standard deviation to the same precision, using scientific notation if needed."""
    sci_low_threshold = 0.001
    sci_high_threshold = 1000
    
    if (abs(mean) < sci_low_threshold or abs(mean) > sci_high_threshold) or \
       (abs(std) < sci_low_threshold or abs(std) > sci_high_threshold):
        formatted_mean = format_scientific(mean, sig_figs, latex)
        formatted_std = format_scientific(std, sig_figs, latex)
    else:
        # Calculate decimal places based on the mean's significant figures
        significant_figure_place = int(np.floor(np.log10(abs(mean)))) - (sig_figs - 1)
        decimal_places = -significant_figure_place if significant_figure_place < 0 else 0
        formatted_mean = format_decimal_places(mean, decimal_places)
        formatted_std = format_decimal_places(std, decimal_places)

    pm = r"\pm" if latex else "+/-"
    val_str = f"{formatted_mean} {pm} {formatted_std}"
    return f"${val_str}$" if latex else val_str
