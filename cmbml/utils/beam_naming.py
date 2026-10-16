import math
from collections.abc import Iterable


def beam_fwhm_to_name(fwhm_arcmin: float | None) -> str:
    """Convert a beam FWHM (arcmin) to a filesystem-safe name."""
    if fwhm_arcmin is None:
        return "deconv"

    fwhm = float(fwhm_arcmin)
    if not math.isfinite(fwhm) or fwhm <= 0:
        raise ValueError(f"Invalid beam FWHM: {fwhm_arcmin!r}")

    return f"{fwhm:g}".replace(".", "p") + "arcmin"


def name_beam_fwhms(
    fwhms_arcmin: Iterable[float | None],
) -> list[tuple[str, float | None]]:
    """Return (beam_name, FWHM) pairs in input order.

    Accepts Python lists and OmegaConf ListConfig values.
    Rejects duplicate beam names.
    """
    result = []
    seen = set()

    for value in fwhms_arcmin:
        fwhm = None if value is None else float(value)
        name = beam_fwhm_to_name(fwhm)

        if name in seen:
            raise ValueError(f"Duplicate target beam name: {name}")

        seen.add(name)
        result.append((name, fwhm))

    if not result:
        raise ValueError("Beam FWHM list must not be empty")

    return result
