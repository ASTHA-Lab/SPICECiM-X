"""
A drop-in replacement for the spectre binary that the integration test uses
to validate the full transient pipeline without invoking the real simulator.

Reads its input deck path from argv (just to match the real spectre CLI),
ignores the actual deck content, and writes a synthetic ``tran.tran.tran``
psfascii dump under the directory passed via ``-raw``.

The synthetic dump is driven by per-job settings, given either as environment
variables or as ``NAME=value`` command-line arguments:
    SPICECIM_FAKE_NPY     : path to the subarray .npy (gives us G_block)
    SPICECIM_FAKE_V_NPY   : path to the V matrix for THIS subarray's BL slice
    SPICECIM_FAKE_BRANCH  : 'pos' | 'neg' (sign convention)
    SPICECIM_FAKE_NCOL    : number of SL probes
    SPICECIM_FAKE_INC     : window length in seconds
    SPICECIM_FAKE_M       : number of windows (images)

The emitted psfascii has, for each window, one '"time"' sample at the
window mid-point and one '"IPRB{k}:in"' sample per probe equal to the
analytic per-row dot product V_row @ G_branch_col.

This is intentionally NOT a faithful SPICE model — it just shoves the
arithmetic answer into the place dc.py-style parsing expects, so we can
verify the assembly logic in the transient backend.
"""

import os
import sys
import numpy as np


def main():
    argv = sys.argv[1:]
    # Per-job settings may also be passed as SPICECIM_FAKE_*=value arguments
    # (portable alternative to a shell wrapper exporting env vars).
    for a in argv:
        if a.startswith("SPICECIM_FAKE_") and "=" in a:
            k, v = a.split("=", 1)
            os.environ[k] = v
    # Find the -raw <dir> argument
    raw_dir = None
    for i, a in enumerate(argv):
        if a == "-raw" and i + 1 < len(argv):
            raw_dir = argv[i + 1]
            break
    if raw_dir is None:
        sys.exit("FAKE_SPECTRE: missing -raw arg")
    os.makedirs(raw_dir, exist_ok=True)

    npy_path  = os.environ["SPICECIM_FAKE_NPY"]
    v_path    = os.environ["SPICECIM_FAKE_V_NPY"]
    branch    = os.environ["SPICECIM_FAKE_BRANCH"]
    ncol      = int(os.environ["SPICECIM_FAKE_NCOL"])
    inc       = float(os.environ["SPICECIM_FAKE_INC"])
    M         = int(os.environ["SPICECIM_FAKE_M"])

    B_block = np.load(npy_path)               # shape (rs, cs)
    V_slice = np.load(v_path)                 # shape (M, rs)

    # Conductance scaling parameters come from env too
    c = float(os.environ["SPICECIM_FAKE_C"])
    d = float(os.environ["SPICECIM_FAKE_D"])
    Gmin = float(os.environ["SPICECIM_FAKE_GMIN"])
    Tmin_val = float(os.environ["SPICECIM_FAKE_TMIN_GVAL"])  # G of HRS cell

    # Build per-branch conductance with the differential mapping the real
    # backends program (device_emit.convert_to_conductance):
    #     G+ = c·max(B, 0) + Gmin,   G- = c·max(-B, 0) + Gmin
    # so the shared baseline cancels in I+ - I-.  (``d`` and ``Tmin_val``
    # are accepted for backward compatibility but no longer used.)
    if branch == "pos":
        G = c * np.maximum(B_block, 0.0) + Gmin
    else:
        G = c * np.maximum(-B_block, 0.0) + Gmin

    # Currents this subarray would produce: I = V @ G, shape (M, cs)
    I = V_slice @ G

    # Emit the psfascii dump.  Format mimics the real Spectre output:
    #   header lines (anything; reader looks for "VALUE")
    #   VALUE
    #   ( "time"    t0 )
    #   ( "IPRB0:in" i00 )
    #   ( "IPRB1:in" i01 )
    #   ...
    #   ( "time"    t1 )
    #   ( "IPRB0:in" i10 )
    #   ...
    tran_path = os.path.join(raw_dir, "tran.tran.tran")
    with open(tran_path, "w") as f:
        f.write("// fake spectre psfascii dump\n")
        f.write("VALUE\n")
        # Emit several samples per window so the settled-value reducer has
        # something to reduce.  Put 3 samples inside the settled region.
        sl = float(os.environ.get("SPICECIM_FAKE_SL", "0.40"))
        sh = float(os.environ.get("SPICECIM_FAKE_SH", "0.65"))
        for w in range(M):
            t_start = w * inc
            for frac in (sl, 0.5 * (sl + sh), sh):
                t = t_start + frac * inc
                f.write(f'( "time" {t:.6e} )\n')
                for k in range(ncol):
                    f.write(f'( "IPRB{k}:in" {I[w, k]:.10e} )\n')
    # done
    sys.exit(0)


if __name__ == "__main__":
    main()
