"""Python port of an IDL TESS quick-look light-curve pipeline.

The pipeline removes instrumental systematics by fitting them *simultaneously*
with the stellar variability, rather than flattening first -- which is what
keeps it unbiased for rapidly-rotating young stars where PDCSAP struggles.

Three entry points, sharing one correction:

:func:`quicklooktess`
    **Recommended.** Picks the fastest data available in each sector: 20 s SPOC
    if the target was on the fast list, else 120 s SPOC, else FFI photometry.

:func:`quicklooktesssc`
    SPOC short-cadence only (``quicklooksector3.pro``).  Photometry comes from
    the mission light-curve files; dilution from ``CROWDSAP``.

:func:`quicklooktessffi`
    FFI only (``quicklooktessffi.pro``).  Does its own photometry from a TESScut
    cutout, with aperture selection and a PRF scene model for dilution.

::

    from tessquicklook import quicklooktess
    result = quicklooktess(166527623, outfile="lc.csv")
"""

from .dispatch import plan_cadences, quicklooktess  # noqa: F401
from .pipeline import quicklooktessffi, write_lightcurve  # noqa: F401
from .scpipeline import quicklooktesssc  # noqa: F401

__version__ = "0.2.0"
__all__ = [
    "quicklooktess",
    "quicklooktesssc",
    "quicklooktessffi",
    "plan_cadences",
    "write_lightcurve",
]
