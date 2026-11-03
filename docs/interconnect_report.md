# Tile loading and NoC in SPICECiM-X: method and validation

## 1. What the RSN leaves out

The reusable subarray netlist (RSN) simulates every 16×16 subarray of a layer
in its own Spectre deck. Each deck has ideal row sources and every column
ends in an ideal ground (`IPRB<i> (SL VSS)`). The per-subarray column
currents are then summed in Python. Three physical effects sit outside those
ideal boundaries:

| Missing effect | Where it sits | What it changes |
|---|---|---|
| **Column-line loading** | The op ledger counts **one TIA/ADC per output column per layer**. So the `R = ⌈k/16⌉` row-blocks of a column are summed in the analog domain, by KCL, on one shared line into one TIA, with no peripheral in between. The summed current flows through the line resistance and the TIA's finite input resistance, which raises every subarray's source-line end. | column currents (accuracy), crossbar conduction energy, settling |
| **Input-bus loading** | One DAC drives each activation row across all `C = ⌈n/16⌉` column-blocks and both ± arrays. | DAC/driver energy (wire C·V²), settling |
| **Inter-tile network (NoC)** | Activations travel between the tiles of every matmul and the host/global buffer that runs softmax, LayerNorm and residuals (executed digitally on the host). | energy and latency of data movement |

Varying the RSN seed size does not expose these effects: with ideal
boundaries, every decomposition sums the same currents.

## 2. What was added to the tool

All of it sits behind one config section, `[INTERCONNECT]`
(`SPICECIMX_V3/TUTORIAL.md` §3, §5.1). With `enable = false`, results are
bit-identical to the ideal RSN.

### 2.1 Physical floorplan (`spicecimx/arch/floorplan.py`)

A layer's `R × C` subarray grid is laid out as indexed, with the ± arrays of
each position side by side:
- `pe_rows × pe_cols` positions form a PE, and `tile_rows × tile_cols` PEs form a tile (4×4 positions in the reference configuration);
- lines cross a 5 µm gap between subarrays and an extra 20 µm at PE/tile boundaries;
- the cell pitch comes from the PDK: `W + 4F` across columns, `8F` across rows (2.18 × 0.36 µm at 45 nm);
- line R is semi-global metal per node (0.06 / 0.19 / 0.80 Ω/µm at 180 / 90 / 45 nm), with C = 0.2 fF/µm.

Every one of these values can be overridden in the config.

### 2.2 Loaded RSN: boundary relaxation in SPICE (`spicecimx/arch/loading.py`, `sim/transient_backend.py`)

The coupling between the decks is a classic domain-decomposition problem. It
is resolved with a waveform-relaxation-style iteration that keeps every
Spectre deck at 16×16:

1. **Pass 1:** the ideal RSN. Its waveforms give the
   ideal column currents `I0` and the "ideal RSN" crossbar energy.
2. **Line solve:** each column line is a ladder: TIA input R (auto: R_f/(1+A),
   7.4 Ω at 45 nm), then the wire segments up to every row-block tap. Its exact
   response is `d = Z·I`, with `Z[t,u]` the common-path resistance. Each
   subarray port is a Norton source `I = I0 − g·d`, with the small-signal cell
   conductance `g = G·h′(V)` taken from the pre-warp curve. The coupled linear
   system gives the predicted boundary of every subarray.
3. **Pass 2:** every deck is re-simulated with each of its column ends
   terminated in the **Thévenin equivalent of the rest of the line**: a
   resistor `Z[t,t]` (its own path to the TIA) in series with a source
   carrying the other subarrays' contribution, `V_oc = d − Z[t,t]·I_pred`.
   SPICE therefore resolves each subarray's own loading exactly, at every
   time point.
4. **Check:** the exact line response to the currents measured in pass 2 is
   compared with the boundary the decks actually saw. A third pass with
   secant-updated port conductances runs when the residual exceeds
   `loading_tol_V` (1 mV). The accepted pass replaces the layer's waveforms,
   so crossbar energy and the measured settle time include the loading.

Two implementation details matter for correctness and were found on real
Spectre:
- An ideal-voltage-source termination rings (±0.5 mA sample to sample) under
  trapezoidal integration. The Thévenin termination's series resistance
  removes the ringing.
- In the first vector of every deck, the input sources' terminal current
  includes the decoder/BL-driver start-up transient (up to 2× the cell
  current). Row currents are therefore taken from the KCL-consistent cell
  model, scaled to the measured column total, not from the sources.

**Input bus.** With row drivers per PE/subarray (the SPICECiM-X architecture, and the
default `input_bus = buffered`), the DAC bus only feeds driver inputs and
carries no DC current. Each subarray's driver supplies its own cells, which
is the RSN's BL driver fed by a source. The bus is charged once per vector:
wire C plus 2 fF per driver input, `E = C·V²`. Its Elmore constant (DAC
R_out 50 Ω) adds to the settle time. `input_bus = shared` (one unbuffered
pass-gate line) is available and solved the same way. For the reference ViT's layer
sizes it drops more than the drive voltage on minimum-width metal, which is
why the buffered architecture is the physical one.

**Capacitive line terms.** The C·V² of every bus and column line per vector
is added to the energy, and `τ·ln(1/settle_band)` of the slowest line
(Elmore) is added to the MVM settle time.

**fastdc** (the accuracy pass of `auto`) applies the same line solve to its
table currents.

### 2.3 NoC (`spicecimx/arch/noc.py`, `arch/booksim.py`)

- **Placement.** Every weight-stationary matmul's tiles are shelf-packed in
  execution order onto a 2D mesh. A global buffer / host sits at the centre
  router.
- **Traffic.** Inputs go buffer → the layer's left-column tiles (DACs), and
  outputs go bottom-row tiles (ADCs) → buffer. The NoC volume therefore
  equals the op ledger's DAC/ADC conversions times their bits; nothing is
  double counted with the analog lines inside the layer. The Q, K and V
  matmuls of all heads read the same LayerNorm output, so it is multicast
  along the XY tree (`noc_multicast`).
- **Energy.**
  - Routers: 0.110 pJ/bit per router traversal. This value is measured with
    BookSim2's Orion power model (2007-ITRS 32 nm techfile, 5-port, 4 VC × 4
    flits, 64-bit flits; 0.110–0.112 pJ/bit for loads 0.05–0.3) and C·V²-scaled
    to each node and corner supply.
  - Links: 0.5·C·V² per bit per mm, over the real tile pitch.
- **Latency.** Serialisation of each phase at the single buffer port (one
  64-bit flit per cycle at 1 GHz) plus the pipeline fill (2 router + 1 link
  cycles per hop). It is added serially to the conversion latency; the
  `noc_overlap` option counts only the pipeline fill instead.
- **BookSim2.** Upstream BookSim2 has only synthetic traffic patterns. A
  86-line patch (`scripts/booksim/patch_booksim.py`) adds a `matrix` traffic
  pattern and injection process that read the workload's source/destination
  matrix. `build_booksim.sh` clones the pinned commit, patches and builds it.
  With `noc = booksim`, each unicast NoC phase of the inference is replayed
  cycle-accurately at the highest stable load, and its time and Orion router
  energy are reported next to the analytic model as a validation.

### 2.4 Report

`power_energy_latency_report.{json,csv}` gains `tile_lines` and `noc`
energy rows, a NoC latency term and an `interconnect` block. The block holds
the ideal-RSN vs loaded-RSN crossbar energy, the line energies, the line
settle, and the NoC mesh, latency and hops. `interconnect_report.json` keeps
the per-layer records: boundary voltages, residual statistics, line R/C,
Elmore constants and passes.

### 2.5 Tests

`SPICECIMX_V3/tests/test_interconnect.py` (10 tests, 16 parametrised cases) checks:
- the ladder algebra against nodal analysis;
- the coupled layer solve against a brute-force MNA of a whole layer (buffered and shared bus);
- the floorplan and NoC volume bookkeeping (NoC bits = DAC/ADC conversions × bits), and multicast;
- config parsing, and the netlist and testbench changes;
- the full transient backend in loaded-RSN mode, driven by a deck-reading fake Spectre with linear cells. It must reproduce the exact MNA solution of the whole layer after two passes, where the ideal RSN visibly does not.

### 2.6 Validation against a monolithic deck

`python -m spicecimx.cli.check_loading --config <cfg> --k 64 --n 32` builds
one monolithic Spectre deck of a whole layer as the reference and compares it
with the ideal and the loaded RSN. On GPDK 45/90/180 nm layers with 4 and 8
row-blocks (`k` = 64 and 128, `input_bus = buffered`), the loaded RSN's
column currents agree with the monolithic deck within 0.01–0.10 % (RMSE of
the current range), and its crossbar energy within +0.0 to +0.53 %. The ideal
RSN is off by up to 1.4 % in current and −5 % in energy.
