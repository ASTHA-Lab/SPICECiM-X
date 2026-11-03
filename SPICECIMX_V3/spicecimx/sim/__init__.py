"""
Unified backend dispatch for SPICECiM-X v3 simulation.

The ``imc_matmul`` core accepts any callable matching
``backend(A, B, scales, tag, bname, dump_dir) -> I_SPICE``.  This module
picks the right one based on ``sim.mode``:

    transient  -> spicecimx.sim.transient_backend.make_transient_backend(ctx)
    dc         -> spicecimx.sim.dc_backend.make_dc_backend(ctx)
    hybrid     -> spicecimx.sim.hybrid_backend.make_hybrid_backend(ctx)
    fastdc     -> spicecimx.sim.fastdc_backend.make_fastdc_backend(ctx)

Usage::

    ctx = build_context(hw, sim, paths, batch_id=...)   # or FastDcBackendContext for fastdc
    backend = make_backend(ctx)
    I_spice = imc_matmul(A, B, ..., sim_backend=backend)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from spicecimx.sim.transient_backend import (
    TransientBackendContext, build_context, make_transient_backend,
)
from spicecimx.sim.dc_backend import make_dc_backend


def make_backend(ctx) -> Callable[..., np.ndarray]:
    """Return the backend callable matching ``ctx.sim.mode``.

    Imports hybrid and fastdc backends lazily so callers that don't use
    them don't pay the import cost.
    """
    mode = ctx.sim.mode.lower()
    if mode == "transient":
        return make_transient_backend(ctx)
    if mode == "dc":
        return make_dc_backend(ctx)
    if mode == "hybrid":
        from spicecimx.sim.hybrid_backend import make_hybrid_backend
        return make_hybrid_backend(ctx)
    if mode == "fastdc":
        from spicecimx.sim.fastdc_backend import make_fastdc_backend
        return make_fastdc_backend(ctx)
    raise ValueError(
        f"Unknown sim.mode={ctx.sim.mode!r}; expected one of "
        "'transient', 'dc', 'hybrid', or 'fastdc'."
    )


__all__ = ["build_context", "make_backend"]