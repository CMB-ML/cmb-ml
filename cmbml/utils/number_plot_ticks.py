import numpy as np


def make_tick_labels(ticks):
    tick_labels = []
    for tick in ticks:
        tick_labels.append(make_tick_label(tick))
    return tick_labels


def make_tick_label(tick):
    res = make_tick_label_dbg(tick)[0]
    res = res.replace(".0$", "$")
    return res 


def make_tick_label_dbg(tick):
    res = make_tick_label_dbg2(tick)
    r0, r1 = res
    r0 = r0.replace(".0$", "$")
    r0 = r0.replace(".0 ", " ")
    return r0, r1


def make_tick_label_dbg2(tick):
    pfx = ""
    if tick == 0:
        return "$0$", "a"
    if tick < 0:
        pfx = "-"
        tick = -tick

    log = np.log10(tick)
    if log == 0:
        return f"${pfx}1$", "b"

    if tick == 10:
        return f"${pfx}10$", "c"

    exponent = int(np.floor(np.log10(tick)))

    base = tick / 10**exponent
    base = np.round(base, 2)
    characteristic = int(base)
    mantissa = base - characteristic
    # print(log, base, characteristic, mantissa, exponent)

    if characteristic == 1 and characteristic == base:
        return f"${pfx}10^{{{exponent}}}$", "d"

    if exponent >= 3 or exponent <= -3:
        if mantissa == 0:
            return f"${pfx}{characteristic} \\times 10^{{{exponent}}}$", "e"
        return f"${pfx}{np.round(base, 1)} \\times 10^{{{exponent}}}$", "f"

    if exponent == 0:
        return f"${pfx}{base}$", "g"

    if exponent == 1:
        return f"${pfx}{base*10**exponent:.1f}$", "h"

    if exponent == 2:
        return f"${pfx}{base*10**exponent:.0f}$", "i"

    res = f"{np.round(tick, 2)}"
    res = res.split("00")[0]
    res = res[:5]
    return f"${pfx}{res}$", "j"


if __name__ == "__main__":
    # Testing the function
    check_vals = [
        0, 
        *[10**i for i in range(-3,4)],
        *[2*10**i for i in range(-3,4)],
        *[1.2*10**i for i in range(-3,4)],
        *[1.64642*10**i for i in range(-3,4)],
        *[1.9*10**i for i in range(-3,4)],
        *[2.1*10**i for i in range(-3,4)],
        *[1.99*10**i for i in range(-3,4)],
        *[2.01*10**i for i in range(-3,4)],
    ]
    for val in check_vals:
        asdf = make_tick_label_dbg(val)
        print(f"{val:<23}: {asdf[1]}, {asdf[0]}")
    for val in check_vals:
        asdf = make_tick_label_dbg(-val)
        print(f"{-val:<23}: {asdf[1]}, {asdf[0]}")
