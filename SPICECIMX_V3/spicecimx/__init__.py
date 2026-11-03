"""
SPICECiM-X — SPICE-accurate simulation of Transformer inference on ReRAM
analog in-memory-computing crossbars.

Entry points (``python -m ...``):

    spicecimx.cli.infer               SPICE-accurate ViT inference + reports
    spicecimx.cli.characterize        one-time 1T1R cell I-V table (fastdc)
    spicecimx.cli.characterize_array  one-time subarray I-V table (pre-warp)
    spicecimx.cli.gui                 configuration / run GUI
    spicecimx.cli.run --selftest      numpy-only check of the IMC kernel

The simulation backends are selected through ``spicecimx.sim.make_backend``.
Importing this package is cheap: subpackages are imported on demand.
"""

__version__ = "1.1.0"
