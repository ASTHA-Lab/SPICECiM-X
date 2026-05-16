# SPICECiM-X v3 Tutorial

A SPICE-accurate simulation framework for Transformer inference on ReRAM-based
Analog In-Memory Computing (AIMC) crossbars.

This document covers what is new in v3 vs v2, how to install and run the
refactored tool, how to choose between the three simulation modes
(transient, DC, hybrid), and how to interpret the outputs as a diagnostic
of where analog Transformer inference fails.  A quick migration recipe at
the bottom shows how to swap v2 functions for their v3 equivalents in an
existing inference loop.

---

## 1. What is new in v3

v3 is a near-total refactor of the simulator with three high-level goals:

1. **Correctness.**  Eight specific bugs in the v2 post-processing pipeline
   were identified and fixed.  The most consequential is the *complete*
   affine de-mapping at the heart of `imc_matmul` (Eq. 31 in the paper):
   v2 implemented only one of the three terms required to recover floating
   point from the IMC current, leading to relative errors between 2.9× and
   14.9× on realistic ReRAM conductance windows whenever the mapping
   coefficient `d` was non-zero.  Other fixes include:

   * settled-current extraction using a robust median over a configurable
     plateau window, not the edge-spike peak that v2 used,
   * removal of dead code (`current_to_voltage` in `dp.py`) that clipped
     negative currents to zero and destroyed sign information,
   * per-instance polarity caches so concurrent matmuls don't share global
     state,
   * coherent inter-layer voltage scaling.

2. **Parallel simulation across Spectre licenses.**  v2 ran one Spectre
   invocation at a time.  v3 dispatches every (subarray, branch) job to a
   `JobScheduler` that pulls licenses from a counting-semaphore-backed
   `LicensePool`.  On hardware with N Spectre seats, the wall clock for
   per-layer simulation drops by very nearly a factor of N (verified up to
   N=16 with a fake Spectre stand-in — peak observed concurrency exactly
   matches license count in every test case).

   Post-processing of one subarray runs in parallel with the simulation of
   the next, on a separate process pool.  The pipelined schedule is
   automatic; set `sim.pipeline_postprocess=False` to disable it if you
   want to A/B compare.

3. **Three simulation modes** selectable via a single config flag:

   * `transient` — full transient SPICE with PWL voltage sources and
     time-domain settling.  This is the v2 default and is needed for
     timing/energy claims and any analysis that depends on RC settling.
   * `dc` — parametric DC sweep across image index.  One Spectre
     invocation per (subarray, branch) covers all M images at M DC
     operating points.  Roughly an order of magnitude faster than
     transient for inference workloads, but ignores time-domain effects.
   * `hybrid` — DC for the primary currents, transient on a configurable
     sampled subset of subarrays for a validation report.  Useful for
     answering "how much does DC deviate from transient on my workload?"
     without paying the full transient cost.

   The new "ideal SPICE" diagnostic experiment in the paper draft maps
   directly to DC mode: it isolates whether observed accuracy issues come
   from software pipeline bugs versus genuine hardware non-idealities, by
   stripping away the time-domain physics.

A summary of the v3 package layout, then the rest of this document covers
how to use each piece.

```
spicecimx_v3/
├── spicecimx/
│   ├── __init__.py
│   ├── config.py             ← Cfg / HwCfg / SimCfg / PathsCfg + loader
│   ├── core/
│   │   └── imc_matmul.py     ← The corrected affine matmul kernel
│   ├── hw/
│   │   ├── netlist.py        ← Subarray, decoders, std cells, weight tiling
│   │   └── device_emit.py    ← Verilog-A subckt emission (transient + DC)
│   ├── io/
│   │   ├── parse_transient.py← Robust psfascii reader + settled-value
│   │   └── parse_dc.py       ← Binary PSF DC reader
│   ├── sim/
│   │   ├── __init__.py       ← make_backend(ctx) — mode dispatch
│   │   ├── spectre.py        ← LicensePool, JobScheduler, run_spectre
│   │   ├── templates.py      ← PWL / DC source rendering + testbench
│   │   ├── transient_backend.py
│   │   ├── dc_backend.py
│   │   └── hybrid_backend.py
│   ├── analog/
│   │   └── tia_adc.py        ← Realistic TIA + ADC + calibrator
│   └── cli/
│       └── run.py            ← Minimal demo runner; see Section 8
├── supports/
│   └── _golden_input_dc.scs  ← DC testbench template (new in v3)
└── tests/
    ├── test_imc_matmul_roundtrip.py
    ├── test_v2_vs_v3_demap.py
    ├── test_scheduler.py
    ├── test_transient_backend_e2e.py
    └── test_dc_backend_units.py
```

The v2 files (`reram_imc_infer.py`, `utils.py`, `dp.py`, `dc.py`,
`dataext.py`, etc.) are kept untouched alongside the new package, so you
can A/B compare freely and migrate one layer at a time.

---

## 2. Installation

The package is pure Python and has no compiled extensions.  Required
third-party libraries:

* `numpy`, `pandas` — array math and CSV I/O.
* `torch`, `transformers`, `timm` — only if you're driving the ViT
  inference loop from v2's `vit_reram.py` / `torch_infer.py`.  The new
  package itself does not depend on PyTorch.

You also need:

* Cadence Spectre.  v3 is tested against Spectre 21.1 and later; older
  versions probably work but the parametric DC `table()` expression in DC
  mode requires reasonably recent Spectre.
* The PDK transistor model and a standard cell netlist — same files you
  already had for v2.

Drop the `spicecimx_v3/` directory anywhere on your `PYTHONPATH` (or `pip
install -e .` if you add a `setup.py`).  No environment setup beyond that.

---

## 3. Configuration

v3 uses a single dataclass-based configuration system.  You can build a
`Cfg` programmatically:

```python
from spicecimx.config import Cfg, ModelCfg, HwCfg, SimCfg, PathsCfg

cfg = Cfg(
    model=ModelCfg(name="vit_b_16", num_classes=10, d_model=768),
    hw=HwCfg(
        subarray_rows=16, subarray_cols=16,
        pe_rows=2,        pe_cols=2,
        tile_rows=2,      tile_cols=2,
        vmin=0.0, vmax=1.0,
        Gmin=1e-7, Gmax=5e-5,
    ),
    sim=SimCfg(
        mode="dc",                  # "transient", "dc", or "hybrid"
        license_count=16,           # Spectre seats available
        postprocess_workers=4,
        pipeline_postprocess=True,
        pulse_width=10e-9,
        trise=1e-12, tfall=1e-12,
        settling_window_lo=0.40,
        settling_window_hi=0.65,
        settling_reducer="median",
    ),
    paths=PathsCfg(
        hwsimdb_root="./hwsimdb",
        tmp_root    ="./tmp",
        stdcell_netlist        ="./supports/standard_cell.netlist",
        golden_input_transient ="./supports/_golden_input.scs",
        golden_input_dc        ="./supports/_golden_input_dc.scs",
        device_model_transient ="./supports/reram_cell.v",
        device_model_dc        ="./supports/sky130_dc_memristor_dc.v",
        transistor_model       =".../gpdk180_v3.3/models/spectre/gpdk.scs",
    ),
)
```

…or load it from a TOML/INI file via `load_cfg(path)`.

The split into `HwCfg` / `SimCfg` / `PathsCfg` is deliberate.  `HwCfg`
encodes everything about the *physical hardware being modeled*: subarray
shape, PE/tile tiling, conductance window, device model parameters.
`SimCfg` encodes everything about *how the simulation is run*: mode,
license count, settling parameters, ADC bits.  `PathsCfg` is everything
filesystem-y.  This separation means you can sweep simulation mode without
touching hardware parameters, and vice versa — a common workflow during
debugging.

Note on `mode`: the three valid values are `"transient"`, `"dc"`, and
`"hybrid"`.  Hybrid uses two extra fields, `hybrid_transient_fraction`
(default 0.0) and `hybrid_transient_seed`, which control the random
sample of subarrays that get a transient validation run.

---

## 4. Quick start — running one matrix multiply

The lowest-level operation v3 exposes is `imc_matmul`, the corrected
analog matmul kernel.  Once you have a `TransientBackendContext`, a
backend callable, and a TIA-apply callable, calling it looks like
ordinary numpy:

```python
from spicecimx.sim import build_context, make_backend
from spicecimx.core.imc_matmul import imc_matmul, PolarityCache
from spicecimx.hw.device_emit import conductance_calc

ctx = build_context(cfg.hw, cfg.sim, cfg.paths, batch_id="run_001")
backend = make_backend(ctx)        # transient / dc / hybrid per cfg.sim.mode

# Compute the conductance window from the device parameters
alpha = cfg.hw.Ik1 / cfg.hw.Vref
Gmax, Gmin = conductance_calc(alpha, cfg.hw.Tox, cfg.hw.Tref,
                              cfg.hw.Tmax, cfg.hw.Tmin)

# Per-network polarity cache — pass it across all layers so polarity flips
# are diagnosed coherently.  This replaces the global mutable cache in v2.
pcache = PolarityCache()

# TIA-apply callable: maps (I, tag) -> (V_quant, I_hat, meta).
# A real one comes from analog/tia_adc.py; the simplest passthrough is:
def passthrough_tia(I, tag):
    return I, I, {}

# A: (m_tokens, d_in) activation matrix
# W: (d_in, d_out) weight matrix
Y = imc_matmul(
    A=A, B=W,
    Gmax=Gmax, Gmin=Gmin,
    vmin=cfg.hw.vmin, vmax=cfg.hw.vmax,
    sim_backend=backend,
    polarity_cache=pcache,
    tia_apply=passthrough_tia,
    tag="Q_h0",                  # diagnostic label, used in filenames
    bname="batch_001",
    dump_dir=None,
)
```

That's the entire user-facing API for one matmul.  Internally:

1. Inputs are clipped to `[vmin, vmax]` and the affine scales
   `(a, b, c, d)` are derived from `(Gmax, Gmin, vmin, vmax)`.
2. Polarity is determined once (and cached) by comparing the ideal
   `A @ W` result to the analog measurement up to a sign.
3. The `sim_backend` callable is invoked.  It tiles `W` across subarrays,
   builds a `SpectreJob` for every (subarray, branch), dispatches the
   batch through the scheduler, and assembles the final `(m, d_out)`
   current matrix.
4. The current is run through `tia_apply` for current→voltage conversion
   and ADC quantization, then *fully* de-mapped back to floating point
   with all three terms of the affine inverse.

The output `Y` is a `(m, d_out)` numpy array in the same scale as a
straight `A @ W` would produce, modulo whatever non-idealities the chosen
simulation mode exposes.

---

## 5. Choosing a simulation mode

Use this decision tree:

* Are you in a *correctness-first* debugging phase — chasing why MNIST
  works but CIFAR-10 doesn't, or whether the software pipeline matches
  the math?  Start with **DC mode**.  It is the closest you can get to
  "ideal SPICE": you keep all hardware physics that don't require time,
  but strip out RC settling, peripheral coupling, and any transient
  artifact.  If accuracy is still bad in DC mode, the failure is in the
  software pipeline (mapping, de-mapping, polarity, scale) and you can
  find it without paying the transient cost.

* Are you producing numbers for a paper claim about *timing or energy*
  (TOPS/W, latency, power)?  Use **transient mode**.  DC has no time
  domain so it cannot produce energy.

* Are you producing inference accuracy numbers for the paper and want to
  defend them against the "your DC mode is unrealistic" reviewer
  question?  Use **hybrid mode** with
  `sim.hybrid_transient_fraction=0.05` (or so).  v3 will simulate every
  subarray in DC for the actual currents, plus run transient on a 5%
  sample, and emit a `hybrid_report.csv` you can cite to show DC and
  transient agree on your workload within X%.

The wall-clock difference is significant.  On a 16-license workstation,
on the small-ViT inference workload, the rough relative cost is:

* DC: 1× (baseline)
* hybrid (5% transient): ~1.5×
* hybrid (20% transient): ~3×
* transient: ~12×

These are approximate, set by the per-Spectre-invocation runtime which
varies with the device model complexity.  Measure on your own deck.

---

## 6. Outputs and their layout

Every run writes its artifacts under `paths.hwsimdb_root/{batch_id}/`.
The layout matches v2 closely so that any analysis script you have keeps
working:

```
hwsimdb/
└── batch_001/
    └── Q_h0/                       ← one directory per imc_matmul tag
        ├── Tile0/
        │   └── PE0/
        │       ├── Array_0_0.npy        ← B-block for subarray (0,0)
        │       ├── Array_0_0_pos/psf/   ← Spectre raw output (pos branch)
        │       └── Array_0_0_neg/psf/   ← Spectre raw output (neg branch)
        └── hybrid_report.csv         ← only in hybrid mode
```

Temporary working files (input decks, device-model decks, seed netlist)
live in `paths.tmp_root/{batch_id}/{tag}/...` per (subarray, branch) so
concurrent jobs don't collide.

The contents of each `Array_*_pos/psf/` differ by mode:

* transient: `tran.tran.tran` (psfascii)
* dc:        `dc.dc` (PSF binary)
* hybrid:    both — DC for the full run, transient on sampled subarrays

You typically never read these files directly; the backend's
post-processor consumes them as soon as Spectre finishes and only the
assembled `(m, d_out)` current matrix flows downstream.  But they are
useful for debugging — you can pull the psfascii into a notebook and
inspect any anomalous subarray.

---

## 7. Performance tuning

The two knobs that move the most weight on wall-clock time are
`sim.license_count` and `sim.mode`.

**License count.**  Set this to the number of Spectre seats you have
available.  The scheduler will saturate the pool.  The peak concurrency
observed in tests with N=16 was exactly 16 — no underutilization.  Going
*above* your real seat count just produces queueing inside Spectre's own
license manager; don't.

**Mode.**  See Section 5.

**Other knobs.**  `sim.postprocess_workers` controls the size of the
process pool used for parsing psfascii / PSF binary in parallel with sims.
The default (4) is usually fine; bumping it helps if you're running on a
many-core machine with a small license count, where post-processing
becomes the long pole.  `sim.pipeline_postprocess=False` runs all sims
first then post-processes serially; useful for A/B but slower.

**TMP directory placement.**  `paths.tmp_root` should be on a fast local
disk if possible — Spectre writes a few MB of auxiliary files per job and
NFS round-trips can dominate for short jobs.

---

## 8. Migrating from v2

The mechanical mapping from v2 functions to their v3 equivalents:

| v2 (in `utils.py` / `dp.py` / `dc.py`)               | v3 (in `spicecimx/...`)                          |
|------------------------------------------------------|--------------------------------------------------|
| `imc_matmul_v0`                                       | `spicecimx.core.imc_matmul.imc_matmul`           |
| `generate_pwl_sources`                                | `spicecimx.sim.templates.generate_pwl_sources` (back-compat) or `render_pwl_sources_text` |
| `parse_standard_cells`                                | `spicecimx.hw.netlist.parse_standard_cells`      |
| `generate_subarray`                                   | `spicecimx.hw.netlist.generate_subarray`         |
| `distribute_weights`                                  | `spicecimx.hw.netlist.distribute_weights`        |
| `convert_to_conductance{,_ptq,_continuous}`           | `spicecimx.hw.device_emit.*`                     |
| `weight_mapping`                                      | `spicecimx.hw.device_emit.write_device_models_transient` |
| `setup_simulation_args_*`, `run_sim`                  | `spicecimx.sim.spectre.build_spectre_command` + `run_spectre` |
| `map_primary_input`                                   | `spicecimx.sim.templates.render_testbench_transient` |
| `tia_adc_realistic`, `TIACalibrator`                  | `spicecimx.analog.tia_adc.*` (identical signatures) |
| `process_files_in_folder` / `process_crossbar_csv_to_image_csv` | replaced by the backend's per-subarray post-processor; you should not normally call it directly |
| `current_to_voltage`                                  | REMOVED — was dead code that clipped negative currents.  The TIA in `analog/tia_adc.py` is the real implementation. |
| `dataext.PsfDcReader`                                 | `spicecimx.io.parse_dc.PsfDcReader` (byte-identical reader, validated against the user's `dc.dc`)  |

The minimum-touch migration recipe:

1. Replace `imc_matmul_v0(...)` calls with `imc_matmul(...)` from
   `spicecimx.core.imc_matmul`.  This alone gets you the affine fix.
2. Build a `TransientBackendContext` once at the start of your inference
   run and pass `sim_backend=make_backend(ctx)` to every `imc_matmul`
   call.  Now you have parallel Spectre + mode selection.
3. Drop the old `process_files_in_folder` post-processing call entirely
   — the backend's `_postprocess_*_subarray` runs per subarray and the
   assembled current matrix is returned directly by `imc_matmul`.

If you want to stay on v2's transient flow for the first migration step
and just bolt on the affine fix, you can: `imc_matmul` will accept any
callable as `sim_backend`, so you can write a thin adapter that calls v2's
`process_files_in_folder` from inside the backend.  That gets the
correctness fix without touching dispatch.

---

## 9. Debugging recipes

A few common failure modes and how to diagnose them:

* **Output is constant or very small.**  Usually a polarity-cache
  mismatch.  Pass a fresh `PolarityCache()` to `imc_matmul` and see if
  the issue persists.  If polarity flips at every call, the ideal current
  is too small to disambiguate from noise — verify `scales.c` is
  non-degenerate.

* **Output is exactly v2's output × some constant.**  v2's affine
  de-mapping bug.  Check that `scales.d` is nonzero on your inputs.  v3's
  `derive_affine` returns the full `(a, b, c, d)` and `compute_offset_terms`
  emits all three correction terms.  If you forked v2's `imc_matmul_v0` to
  hand-patch one term, you missed the other two — drop in the v3 kernel
  outright.

* **DC mode finishes but gives all-NaN currents.**  Spectre wrote
  something other than `dc.dc` in the raw output, usually because the
  parametric sweep failed (Spectre couldn't find a DC solution for some
  `img_idx`).  Inspect `psf/spectre.out` — the log usually points at a
  convergence failure on a specific subarray.  Common causes: `vmax` set
  above the device's rupture voltage, or a `Tmin` that produces a
  conductance below `gmin` in the simulator options.

* **Transient mode finishes but settled values are noisy/wrong.**  Check
  your settling window.  The default `[0.40, 0.65]` reduces samples in
  the middle 40-65% of each pulse window using a median.  If your
  peripheral RC is slow, the plateau may not have arrived by 40% — widen
  the start and/or extend `sim.pulse_width`.

* **Spectre licenses idle, scheduler runs serially.**  Usually a
  `sim.license_count` set to 1 or a `pipeline_postprocess` toggle issue.
  Confirm with `pool.stats()['peak_concurrent']` after a run — it should
  match `license_count`.

---

## 10. Tests included

Five suites ship in `tests/`:

* `test_imc_matmul_roundtrip.py` — sanity check that `imc_matmul` recovers
  exact `A @ B` when the simulator backend is a perfect `A @ B`
  passthrough.  Should run in < 1 second.
* `test_v2_vs_v3_demap.py` — directly quantifies the v2 affine-demapping
  bug across realistic conductance windows.  Reports 2.9–14.9× relative
  error.
* `test_scheduler.py` — exercises the `LicensePool` + `JobScheduler` with
  a `/bin/sleep` stand-in and asserts peak-concurrency == license_count
  for several configurations.
* `test_transient_backend_e2e.py` — runs the full transient backend
  pipeline (tiling → dispatch → post-process → assembly) against a fake
  Spectre that emits synthetic psfascii.  Recovers analytic results to
  machine precision modulo text-format truncation.
* `test_dc_backend_units.py` — exercises the DC-mode-specific code paths
  (source rendering, testbench filling, PSF reading on the user's real
  `dc.dc`).

All five run in under a minute total and produce no output other than
"OK" lines.  Run them after pulling the package or making any change:

```bash
cd spicecimx_v3
python -m tests.test_scheduler
python tests/test_imc_matmul_roundtrip.py
python tests/test_v2_vs_v3_demap.py
python tests/test_transient_backend_e2e.py
python tests/test_dc_backend_units.py
```

---

## 11. Known limitations and future work

* The hybrid backend currently only *reports* DC↔transient agreement; it
  does not apply a per-subarray correction back to the DC output.  A
  natural extension is to fit a scalar `alpha` such that
  `alpha · I_dc[sa] ≈ I_tran[sa]` from the sampled subarrays, then apply
  it layer-wide.  The infrastructure to do this lives inside
  `hybrid_backend._hybrid_imc` and would add maybe 20 lines.
* The DC mode assumes Spectre's `table()` parameter expression is
  available.  For older Spectre versions, switch to the `per_image`
  encoding (one Spectre call per image) by setting
  `sim.dc_input_encoding = "per_image"` — this is sketched in the config
  but the backend implementation is left to a follow-up.
* PVT corners other than the configured `process_corner` are not
  automatically simulated; sweep `sim.process_corner` over multiple runs
  to characterize.

---

## 12. Quick reference: end-to-end inference recipe

```python
import numpy as np
from spicecimx.config import Cfg, ModelCfg, HwCfg, SimCfg, PathsCfg
from spicecimx.sim import build_context, make_backend
from spicecimx.core.imc_matmul import imc_matmul, PolarityCache
from spicecimx.hw.device_emit import conductance_calc

# 1. Build config
cfg = Cfg(
    model=ModelCfg(name="vit_b_16", num_classes=10, d_model=768),
    hw=HwCfg(subarray_rows=16, subarray_cols=16),
    sim=SimCfg(mode="dc", license_count=16),
    paths=PathsCfg(hwsimdb_root="./hwsimdb", tmp_root="./tmp"),
)

# 2. Build context (one per inference batch)
ctx = build_context(cfg.hw, cfg.sim, cfg.paths, batch_id="my_run")
backend = make_backend(ctx)
pcache = PolarityCache()

# 3. Derive conductance window
alpha = cfg.hw.Ik1 / cfg.hw.Vref
Gmax, Gmin = conductance_calc(alpha, cfg.hw.Tox, cfg.hw.Tref,
                              cfg.hw.Tmax, cfg.hw.Tmin)

# 4. Build a TIA-apply callable.  The shape is (I_diff, tag) -> (V_quant,
#    I_hat, meta).  The simplest passthrough (good for getting started or
#    for diagnostic runs where you want to remove ADC error from the
#    picture) is:
def passthrough_tia(I, tag):
    return I, I, {}

# For the real production TIA, see analog/tia_adc.py:tia_adc_realistic
# and the TIACalibrator class.  Wire them up the same way your v2 code
# does — the API is identical; only the import path changed.
tia_apply = passthrough_tia

# 5. Run the inference loop, replacing every linear layer's matmul with
#    imc_matmul.
def imc_linear(x, W, layer_tag):
    return imc_matmul(
        A=x, B=W,
        Gmax=Gmax, Gmin=Gmin,
        vmin=cfg.hw.vmin, vmax=cfg.hw.vmax,
        sim_backend=backend,
        polarity_cache=pcache,
        tia_apply=tia_apply,
        tag=layer_tag, bname="my_run", dump_dir=None,
    )

# 6. Plug into your existing ViT forward — swap `np.matmul(x, W)` for
#    `imc_linear(x, W, "Q_h0")` (and similar tags) everywhere relevant.
```

That is the whole API.  Everything else — Spectre invocation, license
scheduling, per-subarray post-processing, three modes of physics, full
affine de-mapping — is handled inside `imc_matmul` and the backend.

For paper-writing, the natural diagnostic experiments fall right out of
this API:

* "Software vs hardware":  swap `sim.mode='dc'` and compare results to a
  pure-numpy baseline.  If DC matches numpy, the v2 software-pipeline
  bugs were the cause of any accuracy gap.
* "DC vs transient":  swap `sim.mode='hybrid'` with a 5% sample, look at
  `hybrid_report.csv`.  Reports magnitude and sign of the transient
  correction needed.
* "Subarray-level localization":  read the per-subarray `.npy` weight
  blocks and `psf/` raw outputs, plot per-subarray pos-neg current
  distributions, identify outliers.



# SPICECiM-X v3 — Run Guide

How to run the v3 simulator against your existing ViT trainer project +
saved checkpoint.  This is the practical companion to `TUTORIAL.md`.

---

## 1. Directory layout

v3 keeps **two separate trees**: your trainer (unchanged) and the
SPICECiM-X package.  They live side-by-side and never modify each other.

```
~/scratch/proj/
├── NNA_SIMULATION_FRAMEWORK/        ← your trainer project (unchanged from v2)
│   ├── config.ini                   ← extended with [HW] / [SIM] / [PATHS]
│   ├── main.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── vit_reram.py
│   │   ├── train.py
│   │   ├── reram_imc_infer.py       ← v2, kept for A/B comparison
│   │   ├── dc.py, dp.py             ← v2 helpers (no longer used by v3)
│   │   └── utils.py
│   ├── outputs/
│   │   ├── best.pt                  ← trained checkpoint
│   │   └── last.pt
│   ├── Test_Image/
│   │   └── t1/                      ← batch ID inferred from this folder name
│   │       ├── img001.jpg
│   │       └── ...
│   ├── supports/                    ← SPICE / Verilog-A / netlists (see §3)
│   │   ├── _golden_input.scs
│   │   ├── _golden_input_dc.scs
│   │   ├── standard_cell.netlist
│   │   ├── reram_cell.v
│   │   └── sky130_dc_memristor_dc.v
│   └── SPICECIMX_V3/                ← v3 package, sits inside the trainer project
│       ├── pyproject.toml
│       ├── conftest.py
│       ├── spicecimx/
│       ├── tests/
│       └── supports/                ← (optional duplicate; see §3)
│
└── PDK/                             ← your Cadence PDK (absolute path, unchanged)
    └── GPDK/gpdk180_v3.3/models/spectre/gpdk.scs
```

**Generated at runtime** (created fresh per run, scoped by batch ID):
```
~/scratch/proj/NNA_SIMULATION_FRAMEWORK/
├── hwsimdb/t1/                      ← SPICE outputs (.psf, .npy weight tiles)
│   ├── h0_Q__block_00/
│   │   └── Tile0/PE0/Array_0_0_pos/psf/
│   ├── FFN_W1__block_00/
│   └── ...
├── tmp/t1/                          ← per-job working dirs
│   ├── h0_Q__block_00/Array_0_0/pos/   ← isolated tmp for one Spectre run
│   │   ├── seed_netlist.scs            ← symlink
│   │   ├── input.scs                   ← rendered deck for this job
│   │   └── psf/                        ← raw Spectre output
│   ├── ...
│   └── Final_report.txt             ← match/no-match summary
└── imc_traces/t1/                   ← per-image CSV dumps (logits, softmax, intermediates)
    ├── img001/
    │   ├── logits_imc.csv
    │   ├── logits_torch.csv
    │   ├── softmax_imc.csv
    │   ├── softmax_torch.csv
    │   └── block_00/                ← per-block intermediates
    └── img002/...
```

The batch ID (`t1` above) comes from the **leaf folder name of
`--images_dir`** — same convention as v2.  Run two batches in parallel
with `--images_dir Test_Image/t1` and `--images_dir Test_Image/t2` and
they never collide.

---

## 2. Install once

```bash
module load cadence/2025
conda activate spicecim

cd ~/scratch/proj/NNA_SIMULATION_FRAMEWORK/SPICECIMX_V3
pip install -e .
```

After that, `python -m spicecimx.cli.infer` works from anywhere — though
you'll usually run it from the trainer project root so relative paths in
`config.ini` resolve correctly.

---

## 3. Where to place the support files

v3 reads the support file paths from the **`[PATHS]`** section of
`config.ini`.  Defaults are relative to the directory you run from
(usually `NNA_SIMULATION_FRAMEWORK/`):

```ini
[PATHS]
project_dir            = .
images_dir             = ./Test_Image/t1
dump_root              = imc_traces
hwsimdb_root           = ./hwsimdb
tmp_root               = ./tmp
golden_input_transient = ./supports/_golden_input.scs
golden_input_dc        = ./supports/_golden_input_dc.scs
stdcell_netlist        = ./supports/standard_cell.netlist
device_model_transient = ./supports/reram_cell.v
device_model_dc        = ./supports/sky130_dc_memristor_dc.v
transistor_model       = /home/s550a945/scratch/proj/PDK/GPDK/gpdk180_v3.3/models/spectre/gpdk.scs
```

**Recommended placement**: keep ONE copy of the support files in the
trainer project's `supports/` directory (next to `config.ini`).  v2
already expects them there, and v3's defaults match.  Don't keep a
second copy inside `SPICECIMX_V3/supports/` — that one is only used by
the test suite (it ships with a minimal `_golden_input_dc.scs` example
to keep the tests self-contained).

If your support files live elsewhere (e.g. on shared scratch), edit the
`[PATHS]` section.  All paths can be absolute.

| File                             | Purpose                                              | Used by mode      |
|----------------------------------|------------------------------------------------------|-------------------|
| `_golden_input.scs`              | Transient testbench template (PWL voltage sources)   | transient, hybrid |
| `_golden_input_dc.scs`           | DC testbench template (param sweep over `img_idx`)   | dc, hybrid        |
| `standard_cell.netlist`          | Decoder / TIA / peripheral standard cells            | all               |
| `reram_cell.v`                   | Transient Verilog-A ReRAM model                      | transient, hybrid |
| `sky130_dc_memristor_dc.v`       | DC Verilog-A ReRAM model                             | dc, hybrid        |
| `<PDK>/gpdk.scs` (absolute)      | Transistor models from Cadence PDK                   | all               |

---

## 4. Update your `config.ini`

Take your existing v2 `config.ini` and **add three sections** at the
end.  Everything you have (`[DATA]`, `[MODEL]`, `[TRAIN]`, `[SAVE]`)
keeps working unchanged.

```ini
# ── existing sections from your trainer ─────────────────────────────
[DATA]
dataset = fashion_mnist
data_root = ./data
img_height = 16
img_width = 16
channels = 1
val_split = 0.1
num_workers = 2
augment = false

[MODEL]
patch_size = 4
d_model = 32
n_heads = 2
d_k = 16
d_v = 16
depth = 2
mlp_ratio = 2.0
dropout = 0.1
num_classes = 10
use_class_token = true
pos_embed = learnable

[TRAIN]
epochs = 30
batch_size = 256
lr = 3e-4
weight_decay = 0.05
betas = 0.9, 0.999
warmup_epochs = 2
lr_scheduler = cosine
seed = 42
amp = true
device = auto
log_interval = 50

[SAVE]
out_dir = outputs
save_weights = true
save_best = true

# ── NEW sections for SPICECiM-X v3 ──────────────────────────────────
[HW]
# ReRAM device physics (Cu/HfO2 from your original setup)
Tox  = 5.0e-9
Tref = 4.7249e-9
Ik1  = 6.14e-5
Vref = 0.43
Tmax = 4.9e-9
Tmin = 3.3e-9
# Crossbar tiling
subarray_rows = 16
subarray_cols = 16
pe_rows = 2
pe_cols = 2
tile_rows = 2
tile_cols = 2
# Input voltage range for activation encoding
vmin = 0.0
vmax = 1.0

[SIM]
# transient | dc | hybrid — start with 'dc' for fastest sweeps
mode = dc
# Cap on concurrent Spectre invocations.  Set to # of available seats.
license_count = 4
pipeline_postprocess = true
# Transient settling window (fraction of pulse — middle plateau)
settling_window_lo = 0.40
settling_window_hi = 0.65
settling_reducer = median
# Hybrid mode: fraction of subarrays cross-checked with transient (0 disables)
hybrid_transient_fraction = 0.0
hybrid_transient_seed = 0
# ADC / TIA
adc_bits = 16
vmin_tia = 0.0
vmax_tia = 1.0
vmax_adc = 0.95
enable_qerror = false

[PATHS]
project_dir            = .
images_dir             = ./Test_Image/t1
dump_root              = imc_traces
hwsimdb_root           = ./hwsimdb
tmp_root               = ./tmp
golden_input_transient = ./supports/_golden_input.scs
golden_input_dc        = ./supports/_golden_input_dc.scs
stdcell_netlist        = ./supports/standard_cell.netlist
device_model_transient = ./supports/reram_cell.v
device_model_dc        = ./supports/sky130_dc_memristor_dc.v
transistor_model       = /home/s550a945/scratch/proj/PDK/GPDK/gpdk180_v3.3/models/spectre/gpdk.scs
```

---

## 5. Run it — three ways

### Drop-in replacement for v2

v2 command:
```bash
module load cadence/2025
conda activate spicecim
python utils/reram_imc_infer.py --batch_all
```

v3 equivalent:
```bash
module load cadence/2025
conda activate spicecim
python -m spicecimx.cli.infer --batch_all
```

Reads `./config.ini`, finds `outputs/best.pt`, processes
`./Test_Image/t1/`, writes results to `./imc_traces/t1/`,
`./hwsimdb/t1/`, `./tmp/t1/Final_report.txt`.

### Try a different mode without editing config

```bash
# Fastest — DC operating point per matmul
python -m spicecimx.cli.infer --batch_all --mode dc --licenses 4

# Most accurate — full transient (v2-style but parallelised)
python -m spicecimx.cli.infer --batch_all --mode transient --licenses 8

# Diagnostic — DC primary + transient on 20% of subarrays for validation
python -m spicecimx.cli.infer --batch_all --mode hybrid --licenses 4
```

### Sweep a different image batch in parallel

```bash
# Terminal 1
python -m spicecimx.cli.infer --images_dir ./Test_Image/t1 --mode dc --licenses 4 &

# Terminal 2 — different batch ID, isolated dirs, no collision
python -m spicecimx.cli.infer --images_dir ./Test_Image/t2 --mode dc --licenses 4 &
```

---

## 6. Common flags

| Flag                       | Effect                                                     |
|----------------------------|------------------------------------------------------------|
| `--mode {transient,dc,hybrid}` | Override `sim.mode`                                  |
| `--licenses N`             | Override `sim.license_count` (Spectre seats)               |
| `--batch_all`              | Time-multiplex all images through each matmul (v2-style)   |
| `--images_dir DIR`         | Override input dir (sets batch ID from leaf folder name)   |
| `--checkpoint PATH`        | Force a specific `.pt` file                                |
| `--limit N`                | Only process the first N images (good for smoke tests)     |
| `--no_intermediates`       | Skip per-block CSV dumps (faster, less disk)               |
| `--output_classes K`       | Trim final logits to first K columns                       |

---

## 7. Quick sanity check (no Spectre needed)

Before launching a big run, verify the v3 pipeline is wired correctly:

```bash
cd ~/scratch/proj/NNA_SIMULATION_FRAMEWORK/SPICECIMX_V3
python -m spicecimx.cli.run --selftest --m 4 --k 8 --n 6
```

Should print `PASS` with rel error ~1e-16.  This skips Spectre and runs
an analytic-ideal-current backend through the full `imc_matmul` kernel —
it validates the affine de-mapping and config plumbing without needing
a license.

---

## 8. Outputs you'll get

For each image (`img001.jpg`):

```
imc_traces/t1/img001/
├── logits_torch.csv      ← reference (pure PyTorch)
├── logits_imc.csv        ← SPICE-accurate IMC result
├── softmax_torch.csv
├── softmax_imc.csv
├── 01_patch_grid.csv     ← intermediate (only if --no_intermediates not set)
├── 02_patch_tokens.csv
├── ...
└── block_00/
    ├── h0_Q.csv          ← per-head Q matrix (numpy IMC output)
    ├── h0_scores.csv     ← Q @ K.T / sqrt(d)
    ├── h0_probs.csv      ← softmax
    ├── h0_context.csv    ← attention output per head
    ├── FFN_W1.npy        ← weight tile (saved by backend)
    ├── after_ln1.csv
    └── after_ln2.csv
```

Plus the run-level summary:
```
tmp/t1/Final_report.txt
# img001.jpg | torch: 7 (p=0.962) | imc: 7 (p=0.951) | match=True
# img002.jpg | torch: 3 (p=0.887) | imc: 9 (p=0.421) | match=False
# ...
```

---

## 9. Migrating from v2 — function-by-function

If you have any v2 scripts (other than `reram_imc_infer.py`) that call
into the simulator directly, here's the swap table:

| v2                                                        | v3                                                                      |
|-----------------------------------------------------------|-------------------------------------------------------------------------|
| `from utils.utils import imc_matmul`                      | `from spicecimx.core.imc_matmul import imc_matmul`                      |
| `from utils.utils import TIACalibrator`                   | `from spicecimx.analog.tia_adc import TIACalibrator, tia_adc_realistic` |
| `from utils.dc import process_files_in_folder`            | `from spicecimx.io.parse_transient import process_files_in_folder`      |
| `from utils.utils import generate_crossbar_netlist, ...`  | `from spicecimx.hw.netlist import generate_crossbar_netlist, ...`       |
| `from utils.utils import conductance_calc, ...`           | `from spicecimx.hw.device_emit import conductance_calc, ...`            |

Note that v3 `imc_matmul` requires `sim_backend`, `polarity_cache`, and
`tia_apply` callables — see `cli/infer.py` for how to assemble them.

---

## 10. Where to look when something breaks

| Symptom                                 | Where to look                                              |
|-----------------------------------------|------------------------------------------------------------|
| Import errors                           | `pyproject.toml` installed? `conftest.py` present?         |
| "No checkpoint found"                   | Place `best.pt` under `<project>/outputs/` or pass `--checkpoint` |
| "No images found"                       | `paths.images_dir` correct?  Try `--images_dir <abs path>` |
| Spectre exit code != 0                  | Check `tmp/<batch>/<tag>/<Array>/<branch>/input.scs` + raw Spectre stderr in `psf/` |
| All-zero currents                       | `transistor_model` path?  `process_corner` in `[SIM]`?     |
| Huge `imc - torch` mismatch             | Try `--mode dc` to isolate physics from settling artifacts; check `polarity_cache` is shared across calls (CLI does this) |
| Tests fail with "No module named spicecimx" | `pip install -e .` from the SPICECIMX_V3 dir            |



cd ~/scratch/proj/NNA_SIMULATION_FRAMEWORK/SPICECIMX_V3
# pull in the four fixed files (templates.py, device_emit.py, and the two .scs templates)
pip install -e . --force-reinstall --no-deps

## Fast DC mode

Build the table (once per device)
Either reuse your existing CSV directly via config:
```
[PATHS]
char_table_path = supports/1T1R_dc_char.csv
[SIM]
fastdc_V_min  = 0.1     # match the CSV's implicit grid
fastdc_V_max  = 1.0
fastdc_V_step = 0.01
```
---

Or build a fresh native .npz over the full grid:
```
python -m spicecimx.cli.characterize \
    --config config.ini \
    --veriloga /lustre/.../sky130_dc_memristor/veriloga/veriloga.va \
    --output supports/cell_table.npz \
    --V_min 0.0 --V_max 1.0 --V_step 0.01 \
    --T_min 3.3e-9 --T_max 4.9e-9 --T_step 0.01e-9
```
---

cd ~/scratch/proj/NNA_SIMULATION_FRAMEWORK/VIT_TEST/SPICECIMX_V3
# overwrite/add files as listed above
pip install -e . --force-reinstall --no-deps --break-system-packages
