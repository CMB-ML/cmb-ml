def add_missing_multipoles(in_cmb_ps, max_ell=None):
    out_cmb_ps = in_cmb_ps.split('\n')
    
    # Insert placeholder monopole and dipole
    pl_mono = "     0   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00"
    pl_di   = "     1   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00   0.000000E+00"
    out_cmb_ps.insert(1, pl_mono)
    out_cmb_ps.insert(2, pl_di)

    if max_ell is not None:
        # Find current maximum ell from the last line
        last_line = out_cmb_ps[-1].strip()
        if last_line == '':
            out_cmb_ps.pop()
            last_line = out_cmb_ps[-1].strip()
        try:
            current_max = int(last_line.split()[0])
        except (IndexError, ValueError):
            raise ValueError(f"Could not parse ell from last line: {last_line!r}")

        if current_max < max_ell:
            # Append missing rows with zeros up to max_ell
            for ell in range(current_max + 1, max_ell + 1):
                new_line = f"{ell:6d}   " + "   ".join(["0.000000E+00"] * (len(last_line.split()) - 1))
                out_cmb_ps.append(new_line)

        elif current_max > max_ell:
            # Truncate lines beyond max_ell
            out_cmb_ps = [
                line for line in out_cmb_ps if int(line.split()[0]) <= max_ell
            ]

    out_cmb_ps = '\n'.join(out_cmb_ps)
    return out_cmb_ps
