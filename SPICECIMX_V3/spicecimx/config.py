"""
Configuration for SPICECiM-X v3.

Backward-compatible with the existing config.ini (DATA / MODEL / TRAIN / SAVE),
plus three new sections:

    [HW]   — ReRAM device + crossbar geometry (was hardcoded in reram_imc_infer.py)
    [SIM]  — simulator backend: mode, licenses, settling window, ADC
    [PATHS]— absolute paths to PDK, support files, output roots

Anything not present in config.ini falls back to a sensible default, so existing
config.ini files for ViT training keep working unchanged.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Existing sections (mirror of utils/config.py — kept identical for parity)
# ---------------------------------------------------------------------------
@dataclass
class DataCfg:
    dataset: str
    data_root: str
    img_height: int
    img_width: int
    channels: str
    val_split: float
    num_workers: int
    augment: bool


@dataclass
class ModelCfg:
    patch_size: int
    d_model: int
    n_heads: int
    d_k: int
    d_v: int
    depth: int
    mlp_ratio: float
    dropout: float
    num_classes: int
    use_class_token: bool
    pos_embed: str


@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    betas: tuple
    warmup_epochs: int
    lr_scheduler: str
    seed: int
    amp: bool
    device: str
    log_interval: int


@dataclass
class SaveCfg:
    out_dir: str
    save_weights: bool
    save_best: bool
    # Wipe hwsimdb/imc_traces/tmp at the start of each run (fast rename to
    # ./trash/<id> + background delete) so stale traces don't conflict or fill
    # disk.  The characterization cache in supports/ is never touched.
    clean_workspace: bool = True


# ---------------------------------------------------------------------------
# NEW sections — drive the IMC inference pipeline
# ---------------------------------------------------------------------------
@dataclass
class HwCfg:
    """ReRAM device parameters and crossbar geometry."""
    # device physics
    Tox:  float = 5.0e-9
    Tref: float = 4.7249e-9
    Ik1:  float = 6.14e-5
    Vref: float = 0.43
    Tmax: float = 4.9e-9
    Tmin: float = 3.3e-9
    # 1T1R access transistor ON-resistance (ohms).
    # In a 1T1R cell the total cell conductance is the series combination
    # G_cell = G_R / (1 + R_access * G_R), which is NOT affine in G_R.
    # When R_access > 0, programming compensates so that G_cell (not G_R)
    # is the affine target — see device_emit.compensate_for_transistor.
    # Typical value: 1-10 kohm for 180nm processes; set to 0 to disable
    # compensation (back-compat with v2 behavior).
    # Effective per-cell series resistance (access transistor + BL driver +
    # wire).  -1 = 'auto': derive it from the SPICE array table at run time
    # (strongly node dependent -- 0.4k was a placeholder that made every
    # node look identical and under-compensated the weight mapping).
    R_access: float = -1.0
    # parasitics (per-cell; usually zero in deck, real values in PDK)
    res_val: float = 0.0
    cap_val: float = 0.0
    # crossbar tiling
    subarray_rows: int = 16
    subarray_cols: int = 16
    pe_rows: int = 2
    pe_cols: int = 2
    tile_rows: int = 2
    tile_cols: int = 2
    # input voltage range
    vmin: float = 0.0
    vmax: float = 1.0


@dataclass
class SimCfg:
    """Simulator backend selection and runtime knobs."""
    # 'transient' | 'dc' | 'hybrid' | 'fastdc' | 'auto'
    mode: str = "transient"

    # ── parallelism ────────────────────────────────────────────────────────
    # Hard cap on concurrent Spectre invocations. Set to the number of
    # Spectre seats your cluster grants; 1 keeps the old serial behaviour.
    license_count: int = 1
    # If True, post-processing of subarray N runs concurrently with the
    # simulation of subarray N+1.
    pipeline_postprocess: bool = True
    # Total worker threads for the post-process pool (CSV reduction,
    # PSF parsing). Independent of license_count.
    postprocess_workers: int = 8

    # ── transient mode ─────────────────────────────────────────────────────
    # Pulse-width / edge times for the PWL input. The window length is
    # 3*pulse_width + trise + tfall by construction in generate_pwl_sources.
    pulse_width: float = 1e-9
    trise:       float = 1e-12
    tfall:       float = 1e-12
    # Fraction of each window over which the "settled" current is extracted.
    # Default uses the middle 60% of the plateau — robust against
    # displacement-current spikes at edges.
    settling_window_lo: float = 0.40
    settling_window_hi: float = 0.65
    # How to summarise the settled samples: 'median' | 'mean' | 'last'
    settling_reducer: str = "median"
    # Maximum transient time step (s) appended to the tran analysis; 0 = the
    # simulator's own step control (Spectre X presets ignore it).
    tran_maxstep: float = 0.0

    # ── DC mode ────────────────────────────────────────────────────────────
    # Image-vector encoding: 'sweep' (parametric DC, one Spectre call per
    # subarray-branch, M op-points inside) or 'per_image' (M separate calls).
    dc_input_encoding: str = "sweep"
    # In DC mode each BL gets a fixed input voltage per image. The mapping
    # from raw activation to BL voltage is the same affine A→V used in
    # transient mode (V = a·A + b within [vmin, vmax]).

    # ── Hybrid mode ────────────────────────────────────────────────────────
    # In hybrid mode DC is used for the primary I_SPICE, and a fraction of
    # subarrays is additionally simulated with transient mode for a
    # validation report.  0.0 disables transient sampling (degenerates to
    # pure DC).  Sampling is deterministic given the seed.
    hybrid_transient_fraction: float = 0.0
    hybrid_transient_seed:     int   = 0

    # ── failure handling ──────────────────────────────────────────────────
    # What to do when a Spectre job (subarray, branch) still fails after its
    # retries: 'raise' stops the run with the failing job ids (default);
    # 'nan' marks that subarray's output columns NaN and keeps going.
    on_job_failure: str = "raise"
    # Wall-clock limit (s) for one inference Spectre job; 0 = no limit.
    spectre_timeout_s: float = 0.0
    # ── auto mode (transient power/energy + fastdc accuracy) ──
    # which image feeds the transient power/energy pass: -1 = random (seeded),
    # >=0 = that index into the sorted image list.
    auto_transient_image_index: int = -1
    auto_transient_seed:        int = 0
    # auto-mode transient image selection (extended):
    #   auto_transient_image      = filename -> transient runs ONLY that image
    #   auto_transient_num_images = N        -> N distinct random images; the
    #                                           reported crossbar energy/power
    #                                           is the AVERAGE over them
    auto_transient_image:       str = ""
    auto_transient_num_images:  int = 1
    # hwsimdb disk retention for transient runs:
    #   all        = keep every block's raw sim data (waveforms/netlists) [legacy]
    #   last_block = as each new block starts, reduce the previous block's
    #                _pwr/_eng CSVs then DELETE its raw data; only the final
    #                block (+ non-block layers like HEAD) keeps raw files.
    #   none       = reduce + delete raw data right after every layer.
    hwsimdb_retention:          str = "all"

    # ── Spectre invocation ─────────────────────────────────────────────────
    spectre_args:  Tuple[str, ...] = ("+preset=mx", "+mt=16", "+escchars",
                                      "+lqtimeout", "900", "-maxw", "5",
                                      "-maxn", "5", "+logstatus")
    process_corner: str = "stat"
    vdd:  float = 1.8
    vss:  float = 0.0
    temperature: float = 25.0
    tnom: float = 27.0          # nominal (model-extraction) temperature, deg C

    # ── TIA calibration percentiles (ADC/TIA voltages moved to [PERIPHERALS]) ──
    tia_calib_p_lo: float = 0.1
    tia_calib_p_hi: float = 99.9

    # ── fastdc characterisation grid ──────────────────────────────────────
    # Only consulted by `python -m spicecimx.cli.characterize` and by the
    # CSV loader (which has an implicit V grid).  At fastdc inference time
    # the table itself carries the grid in metadata; these defaults just
    # control the build step.
    fastdc_T_min:  float = 3.3e-9
    fastdc_T_max:  float = 4.9e-9
    fastdc_T_step: float = 0.01e-9
    fastdc_V_min:  float = 0.0
    fastdc_V_max:  float = 1.0
    fastdc_V_step: float = 0.01

    # ── ARRAY characterisation grid ───────────────────────────────────────
    # Used by `python -m spicecimx.cli.characterize_array` to build a
    # subarray-level I-V table that captures BOTH cell I-V nonlinearity and
    # the BL_DRIVER pass-gate IR drop (array loading).  Two characterisations
    # are run — one at the primary subarray size (matches the user's actual
    # inference subarray) and one at the secondary size — so the V-linearizer
    # can interpolate to any in-between size at inference time.  Cached under
    # paths.array_table_path with a config-derived hash; auto-rebuild only
    # fires if any hashed input changed (transistor model, corner, vdd, T/V
    # grid, subarray size pair, cell model parameters).
    array_char_V_min:        float = 0.0
    array_char_V_max:        float = 1.0
    array_char_V_step:       float = 0.01
    array_char_T_step:       float = 0.01e-9
    array_char_size_primary:   int = 16
    array_char_size_secondary: int = 32
    array_char_probe_column:   int = 0     # which IPRB to read (all are identical at uniform T)

    # ── Spectre per-invocation timeouts for the one-time characterisations ──
    # A single Spectre call must finish within this many seconds or it is
    # killed.  The array characterisation (N×N, full T×V grid) is the long one;
    # raise it for large arrays / fine grids on a loaded cluster.  The single-
    # cell characterisation is much shorter.  These bound only the build-time
    # characterisation calls, not inference.
    array_char_timeout_s: float = 7200.0
    cell_char_timeout_s:  float = 3600.0


@dataclass
class PeripheralsCfg:
    """DAC / ADC / TIA energy model parameters (consumed by analog.pwr_energy).

    Defaults are 180 nm-flavoured.  The ADC energy can use either the Walden
    figure-of-merit from Murmann's ADC survey (``adc_energy_model=walden``,
    survey SAR median ~22.5 fJ/conv-step) or the tech-node-aware "best
    possible" envelope from the Accelergy ADC plug-in (``accelergy``).  All
    numbers are tunable per technology node without touching code.
    """
    # master switch: produce the consolidated power/energy/latency report
    generate_energy_report: bool = False
    tech_node_nm: float = 180.0
    dac_bits: int = 8
    # ── ADC / TIA operating range (moved here from [SIM]) ──
    adc_bits:   int   = 16
    vmin_tia:   float = 0.0
    vmax_tia:   float = 1.0
    vmax_adc:   float = 0.95                    # ADC full-scale voltage (V)
    enable_qerror: bool = False
    # TIA/ADC calibration: each layer's current range [Imin, Imax] is learned
    # from its first N calls (images; one call per batch with --batch_all) and
    # then frozen, like a calibrate-then-run ADC.  0 = never freeze (the range
    # keeps widening, so results depend on image order).
    tia_calib_images: int = 1
    # ── ADC energy ──
    adc_energy_model: str = "walden"            # 'walden' | 'accelergy'
    adc_fom_fJ_per_convstep: float = 22.5       # Walden FOMW (SAR survey median)
    adc_enob: float = -1.0                      # <0 → use peripherals.adc_bits as ENOB
    # Full-scale column current (A) for the physical TIA+ADC characterisation
    # sweep (feature 5).  Must bracket the crossbar's max column current.
    adc_i_fullscale: float = 8e-4
    # ── DAC energy ──
    dac_energy_pJ_per_conv: float = -1.0        # <0 → switched-cap model below
    dac_cap_fF: float = 500.0
    dac_vref_V: float = 0.2
    dac_switching_activity: float = 1.0 / 3.0   # SAR/binary cap array average
    # ── TIA energy ──
    tia_bias_current_A: float = 20e-6
    tia_supply_V: float = 1.8
    tia_active_time_s: float = -1.0             # <0 → one clock/sample period
    # ── multiplexer (converter sharing) energy ──
    # A mux is charged only when num_dac/num_adc < a layer's I/O width.
    mux_energy_pJ_per_op: float = -1.0          # <0 → transmission-gate model below
    mux_cap_fF: float = 2.0                     # per-stage transmission-gate cap
    mux_vdd_V: float = 1.8                      # mux switching rail
    # ── subtractor (separate stage: TIA outputs → subtractor → ADC) ──
    differential_columns: bool = True           # +col/-col → 2 TIAs + 1 subtractor/output
    subtractor_energy_pJ_per_op: float = -1.0   # <0 → switched-cap model below
    subtractor_cap_fF: float = 10.0
    subtractor_vdd_V: float = 1.8
    # ── bit-slicing ──
    n_bit_slices: int = 1
    adc_per_slice: bool = False                 # digitise each slice separately


@dataclass
class LatencyCfg:
    """Cycle-accurate latency model.  Latency = data-conversion cycles
    (DAC + ADC) + MAC/VMM settling cycles in the IMC, at the configured clock.
    """
    clock_freq_hz: float = 1.0e9                # 1 GHz → 1 ns clock
    num_dac: int = 16                           # physical DACs working in parallel
    num_adc: int = 16                           # physical ADCs working in parallel
    adc_type: str = "sar"                       # sar|flash|pipeline|sigma_delta|integrating
    adc_cycles_per_conv: int = 0                # 0 → derive from adc_type & adc_bits
    dac_cycles_per_conv: int = 1
    mac_cycles_per_vector: int = 1              # VMM settle cycles per input vector
    vmm_parallel_subarrays: int = 0             # 0 → pe_rows*pe_cols*tile_rows*tile_cols
    adc_osr: int = 0                            # sigma-delta oversampling (0 → 2^bits)
    mux_cycles: int = 0                         # extra settle cycles per muxed wave
    # --- technology / corner / temperature aware timing ---
    periph_delay_model: str = "fixed"           # fixed | node_scaled
    ref_node_nm: float = 0.0                    # node where clock_freq_hz holds
    node_delay_exponent: float = 1.0            # (node/ref)^e
    corner_delay_ff: float = 0.85               # ff timing multiplier
    corner_delay_ss: float = 1.30               # ss timing multiplier
    temp_delay_coeff_per_C: float = 0.001       # ~+0.1%/degC mobility slowdown
    settle_band: float = 0.02                   # measured-settle tolerance


@dataclass
class SubcircuitsCfg:
    """Declares which subckt implements each logical component and whether it
    is software-modelled (S) or hardware-simulated (H).

    Stored as ``role -> (subckt_name, mode)``.  The subckt's pin list and bus/
    bit width are *not* stored here — they are parsed from standard_cell.netlist
    at run time by ``hw.subckt_registry.SubcktRegistry`` so a different cell
    library can be dropped in without code changes.
    """
    entries: dict = field(default_factory=dict)   # role -> (name, 'S'|'H')

    def name(self, role: str, default: str = "") -> str:
        v = self.entries.get(role)
        return v[0] if v else default

    def mode(self, role: str, default: str = "S") -> str:
        v = self.entries.get(role)
        return v[1] if v else default

    def is_hardware(self, role: str) -> bool:
        return self.mode(role, "S").strip().upper() == "H"


@dataclass
class PdkCfg:
    """Process/PDK-specific knobs that used to be hard-coded in device_emit.py,
    netlist.py and the characterisation decks.  Changing technology node or
    cell library should require editing only this section — nothing in code.

    Access-transistor geometry is emitted verbatim into the Spectre instance
    line, so the values are strings (they carry units like ``2u``, ``180n``).
    """
    # ── access transistor (instance line: w/l/as/ad/ps/pd/m + model) ──
    transistor_model_name: str = "nmos1"        # device model in transistor_model file
    transistor_w:  str = "2u"
    transistor_l:  str = "180n"
    transistor_as: str = "1.2p"
    transistor_ad: str = "1.2p"
    transistor_ps: str = "5.2u"
    transistor_pd: str = "5.2u"
    transistor_m:  str = "(1)*(1)"
    # ── ReRAM Verilog-A primitive cell names ──
    va_cell_transient: str = "sky130_fd_pr_reram__reram_cell"
    va_cell_dc:        str = "sky130_dc_memristor_dc"
    # ── 1T1R cell subckt naming (MUST match across device_emit + netlist) ──
    cell_prefix: str = "rram_cell_1T1R_180N_STD"
    # ── Verilog-A device params NOT already in [HW] ──
    # (Tox/Tmax/Tmin/Tref/Ik1/Vref come from [HW]; these are the rest.)
    va_area_ox:            float = 1.024e-13
    va_eact_generation:    float = 1.501
    va_eact_recombination: float = 1.5
    va_velocity_k1:        float = 150.0
    va_gamma_k0:           float = 16.5
    va_gamma_k1:           float = -1.25
    va_temperature_0:      float = 300.0
    va_c_thermal:          float = 3.1825e-16
    va_tau_thermal:        float = 2.3e-10
    va_t_step:             float = 1.0e-9
    # ReRAM read/program mode for the TRANSIENT decks:

    #   0 = read mode (default): the filament is FROZEN at the programmed

    #       weight -> no read disturb, no drift, and the transient currents

    #       agree with the DC array table.

    #   1 = full dynamic switching model (needed only to simulate SET/RESET

    #       programming; during a long MVM read it SETs every cell toward

    #       Tmax and the drift is time-step dependent, so crossbar energy

    #       becomes saturated and non-comparable across nodes).

    va_enable_switching: int = 0
    # access-transistor gate drive during read/characterisation (V)
    wl_read_V:             float = 2.4


@dataclass
class ParasiticsCfg:
    """Crossbar interconnect (wire) parasitics.

    Each bit-line (row input) and source-line (column output) is broken into
    per-cell segments with a series resistance ``r_wire_per_cell_ohm`` and an
    optional node capacitance ``c_wire_per_cell_fF`` to ground.  This reproduces
    the IR-drop / line-loading that makes far-corner cells read low (weights
    "decay across the diagonal").  Values are strongly node-dependent: ~1-2.5 ohm
    and ~0.1-0.2 fF per cell at 180 nm (wide metal), rising to kohm at sub-30 nm.
    Set ``enable=false`` (or r=c=0) for the ideal zero-parasitic crossbar.

    If you prefer to derive R from the metal stack:
        r_wire_per_cell_ohm = sheet_res_ohm_per_sq * (cell_pitch / wire_width)
    """
    enable: bool = False
    r_wire_per_cell_ohm: float = 2.5
    c_wire_per_cell_fF:  float = 0.2
    model_wl: bool = False                  # also add WL series R (timing only)
    r_wl_per_cell_ohm: float = 2.5
    driver_at_origin: bool = True           # driver at index-0 end of each line


@dataclass
class InterconnectCfg:
    """Tile-level interconnect that the RSN decomposition leaves out.

    Every RSN deck is a 16x16 (subarray_rows x subarray_cols) crossbar driven
    by ideal sources and terminated into an ideal ground.  In the tile, a
    layer's subarray grid (R row-blocks x C column-blocks, +/- arrays side by
    side) is wired together:

      * input bus  — each activation row runs from its DAC across all C
        column-blocks (both branches).  ``input_bus = buffered`` (default,
        Fig. 2: row drivers per PE/subarray): the bus only feeds the drivers'
        inputs, so it is a capacitive load (C V^2 energy, Elmore delay) and
        every subarray's own driver supplies its cells — the RSN's BL driver
        fed by a source.  ``shared``: one unbuffered pass-gate line carries
        the row current of every subarray (resistive loading, for studies).
      * column line — each output column runs down all R row-blocks into one
        TIA virtual ground with NO peripheral in between: the partial sums
        add by KCL (the op ledger counts one TIA/ADC per output column), so
        the summed current flows through the shared wire and the TIA input
        resistance.

    ``loading`` makes the transient (and fastdc) backends solve these shared
    lines and re-simulate every RSN deck with the resulting boundary voltages
    (see ``arch.loading``).  The capacitive part of the lines (charging
    energy, Elmore settling) is added in closed form.  ``noc`` adds the
    inter-tile network that carries activations between the tiles and the
    global buffer (``arch.noc``).

    Geometry/electrical values left at 0 (or 'auto') are derived per node in
    ``arch.floorplan.resolve_interconnect`` from [PDK] / [PERIPHERALS].
    """
    enable: bool = False
    # ── analog shared-line loading (intra-layer, between RSN subarrays) ──
    loading: bool = True
    input_bus: str = "buffered"         # buffered | shared
    cell_pitch_x_um: float = 0.0        # 0 = auto from [PDK] transistor_w + node
    cell_pitch_y_um: float = 0.0        # 0 = auto (8 F)
    subarray_gap_um: float = 5.0        # routing/periphery gap between subarrays
    tile_gap_um: float = 20.0           # extra gap at PE/tile boundaries
    wire_r_ohm_per_um: float = 0.0      # 0 = auto (per-node semi-global metal)
    wire_c_fF_per_um: float = 0.2
    driver_r_out_ohm: float = 50.0      # DAC / bus-driver output resistance
    bus_tap_c_fF: float = 2.0           # input cap of each subarray row driver
    tia_r_in_ohm: float = -1.0          # <0 = auto: R_f / (1 + A_ol)
    tia_open_loop_gain: float = 100.0   # A_ol for the auto TIA input resistance
    loading_max_passes: int = 2         # SPICE passes (1 = ideal RSN only)
    loading_tol_V: float = 1e-3         # boundary-voltage residual target
    # ── inter-tile network-on-chip ──
    noc: str = "analytic"               # none | analytic | booksim
    noc_flit_bits: int = 64
    noc_packet_flits: int = 4
    noc_clock_hz: float = 1.0e9
    noc_router_cycles: int = 2          # router pipeline depth (cycles/hop)
    noc_link_cycles: int = 1
    noc_router_energy_pJ_per_bit: float = -1.0   # <0 = BookSim2/Orion 32 nm value, node-scaled
    noc_link_energy_pJ_per_bit_mm: float = -1.0  # <0 = C_w V^2 activity, node-scaled
    noc_buffer: str = "center"          # global-buffer position: center | corner
    noc_mesh_cols: int = 0              # 0 = smallest square that fits all tiles
    noc_overlap: bool = False           # True: NoC hides under conversion time
    noc_multicast: bool = True          # Q/K/V inputs sent once (XY multicast tree)
    booksim_binary: str = ""            # patched BookSim2 (noc = booksim)


@dataclass
class AccuracyCfg:
    """Accuracy report (Torch-vs-GT, IMC-vs-GT, SW-HW agreement).

    ``task`` selects the metric set via ``core.metrics`` so the same pipeline
    serves future problem types (classification today; LLM token-prediction
    later by registering a new Task).  Ground-truth labels are optional — when
    none can be resolved, only the SW-HW agreement is reported.
    """
    generate_accuracy_report: bool = True
    task: str = "classification"
    # GT sources (optional). labels_file: CSV "filename,label" or JSON map.
    # label_from_filename: regex whose first group captures the label.
    labels_file: str = ""
    label_from_filename: str = ""
    topk: Tuple[int, ...] = (1,)


@dataclass
class PathsCfg:
    """Filesystem locations.  Everything else is derived from these.

    ``load_cfg`` makes every path absolute: ``project_dir`` is resolved
    against the directory of the config file, and every other relative path
    against ``project_dir`` (see :func:`resolve_paths`).
    """
    project_dir:        str = "."
    images_dir:         str = "./Test_Image"
    dump_root:          str = "imc_traces"
    hwsimdb_root:       str = "./hwsimdb"
    tmp_root:           str = "./tmp"
    # Support files (Verilog-A, std-cell, testbench template)
    golden_input_transient: str = "./supports/_golden_input.scs"
    golden_input_dc:        str = "./supports/_golden_input_dc.scs"
    stdcell_netlist:        str = "./supports/standard_cell.netlist"
    device_model_transient: str = "./supports/reram_cell.v"
    device_model_dc:        str = "./supports/sky130_dc_memristor_dc.v"
    # PDK transistor model (Spectre .scs) — required for every SPICE run;
    # it ships with your PDK and cannot be distributed with this repo.
    transistor_model: str = ""
    # fastdc cell I-V table (built once via `python -m spicecimx.cli.characterize`).
    # Either an .npz (native) or .csv (the user's flat-CSV format with
    # implicit V grid).  Required when sim.mode == "fastdc".
    char_table_path:    str = "./supports/cell_table.npz"
    # Array-level I-V table (built once via `python -m spicecimx.cli.characterize_array`).
    # Captures cell I-V nonlinearity + BL_DRIVER IR drop loading at two
    # subarray sizes for cross-size interpolation.  When set and present,
    # all three backends (fastdc/dc/transient) build their V-linearization
    # from this table; when unset or missing, they fall back to char_table_path
    # (single-cell only).  See spicecimx/hw/array_table.py for the layout.
    array_table_path:   str = "./supports/array_table.npz"
    # Optional netlist with the hardware (H) peripheral subckts (DAC/TIA/
    # SUBTRACTOR/ADC) referenced in [SUBCIRCUITS].  Kept separate from the
    # std-cell library because it may ahdl_include Verilog-A models; it is
    # only included by the converter characterisation decks.
    peripheral_netlist: str = ""


@dataclass
class Cfg:
    """Top-level configuration container."""
    data:  DataCfg
    model: ModelCfg
    train: TrainCfg
    save:  SaveCfg
    hw:    HwCfg    = field(default_factory=HwCfg)
    sim:   SimCfg   = field(default_factory=SimCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    peripherals: PeripheralsCfg = field(default_factory=PeripheralsCfg)
    latency:     LatencyCfg     = field(default_factory=LatencyCfg)
    subcircuits: SubcircuitsCfg = field(default_factory=SubcircuitsCfg)
    pdk:         PdkCfg         = field(default_factory=PdkCfg)
    parasitics:  ParasiticsCfg  = field(default_factory=ParasiticsCfg)
    accuracy:    AccuracyCfg    = field(default_factory=AccuracyCfg)
    interconnect: InterconnectCfg = field(default_factory=InterconnectCfg)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _get_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y", "t")


def _get_with_default(cp, section, key, getter, default):
    """Read a typed value from cp[section][key] or fall back to default."""
    if cp.has_option(section, key):
        return getter(section, key)
    return default


_PATH_KEYS = ("images_dir", "dump_root", "hwsimdb_root", "tmp_root",
              "golden_input_transient", "golden_input_dc", "stdcell_netlist",
              "device_model_transient", "device_model_dc", "transistor_model",
              "char_table_path", "array_table_path", "peripheral_netlist")


def resolve_paths(paths: PathsCfg, base_dir: str) -> PathsCfg:
    """Make every path in ``paths`` absolute, in place.

    ``project_dir`` is resolved against ``base_dir`` (normally the directory
    holding the config file); every other relative path against
    ``project_dir``.  Empty values stay empty.  Call again after changing
    ``project_dir`` (e.g. from ``--project``) — absolute paths are kept.
    """
    import os
    pd = os.path.expanduser(paths.project_dir or ".")
    if not os.path.isabs(pd):
        pd = os.path.join(base_dir, pd)
    paths.project_dir = os.path.normpath(pd)
    for k in _PATH_KEYS:
        v = getattr(paths, k)
        if not v:
            continue
        v = os.path.expanduser(v)
        if not os.path.isabs(v):
            v = os.path.join(paths.project_dir, v)
        setattr(paths, k, os.path.normpath(v))
    return paths


def make_parser() -> configparser.ConfigParser:
    """The INI parser used everywhere (trainer, inference, GUI): ``#``/``;``
    inline comments are stripped and ``%`` is literal (regexes in
    ``label_from_filename`` must not be interpolated)."""
    return configparser.ConfigParser(inline_comment_prefixes=("#", ";"),
                                     interpolation=None)


def load_cfg(path: str) -> Cfg:
    """
    Load an INI file. Sections DATA/MODEL/TRAIN/SAVE are required (same as v2).
    Sections HW/SIM/PATHS are optional — defaults fill in any missing keys.
    All [PATHS] entries come back absolute (see :func:`resolve_paths`).
    """
    import os
    cp = make_parser()
    with open(path, "r", encoding="utf-8-sig") as f:     # tolerate a BOM
        cp.read_file(f)

    # --- existing sections (unchanged) ---
    betas = tuple(float(x) for x in cp.get("TRAIN", "betas").split(","))
    data = DataCfg(
        dataset       = cp.get("DATA", "dataset"),
        data_root     = cp.get("DATA", "data_root"),
        img_height    = cp.getint("DATA", "img_height"),
        img_width     = cp.getint("DATA", "img_width"),
        channels      = cp.get("DATA", "channels"),
        val_split     = cp.getfloat("DATA", "val_split"),
        num_workers   = cp.getint("DATA", "num_workers"),
        augment       = _get_bool(cp.get("DATA", "augment")),
    )
    model = ModelCfg(
        patch_size      = cp.getint("MODEL", "patch_size"),
        d_model         = cp.getint("MODEL", "d_model"),
        n_heads         = cp.getint("MODEL", "n_heads"),
        d_k             = cp.getint("MODEL", "d_k"),
        d_v             = cp.getint("MODEL", "d_v"),
        depth           = cp.getint("MODEL", "depth"),
        mlp_ratio       = cp.getfloat("MODEL", "mlp_ratio"),
        dropout         = cp.getfloat("MODEL", "dropout"),
        num_classes     = cp.getint("MODEL", "num_classes"),
        use_class_token = _get_bool(cp.get("MODEL", "use_class_token")),
        pos_embed       = cp.get("MODEL", "pos_embed"),
    )
    train = TrainCfg(
        epochs        = cp.getint("TRAIN", "epochs"),
        batch_size    = cp.getint("TRAIN", "batch_size"),
        lr            = cp.getfloat("TRAIN", "lr"),
        weight_decay  = cp.getfloat("TRAIN", "weight_decay"),
        betas         = betas,
        warmup_epochs = cp.getint("TRAIN", "warmup_epochs"),
        lr_scheduler  = cp.get("TRAIN", "lr_scheduler"),
        seed          = cp.getint("TRAIN", "seed"),
        amp           = _get_bool(cp.get("TRAIN", "amp")),
        device        = cp.get("TRAIN", "device"),
        log_interval  = cp.getint("TRAIN", "log_interval"),
    )
    save = SaveCfg(
        out_dir      = cp.get("SAVE", "out_dir"),
        save_weights = _get_bool(cp.get("SAVE", "save_weights")),
        save_best    = _get_bool(cp.get("SAVE", "save_best")),
        clean_workspace = (_get_bool(cp.get("SAVE", "clean_workspace"))
                           if cp.has_option("SAVE", "clean_workspace") else True),
    )

    # --- new sections (every key has a default; the section may be absent) ---
    hw = HwCfg()
    if cp.has_section("HW"):
        for k in ("Tox", "Tref", "Ik1", "Vref", "Tmax", "Tmin",
                  "res_val", "cap_val", "vmin", "vmax"):
            if cp.has_option("HW", k):
                setattr(hw, k, cp.getfloat("HW", k))
        # R_access accepts a number OR 'auto' (-1): derive the effective
        # per-cell series resistance from the SPICE array table at run time.
        if cp.has_option("HW", "R_access"):
            raw = cp.get("HW", "R_access").strip().lower()
            hw.R_access = -1.0 if raw in ("auto", "-1") else float(raw)
        for k in ("subarray_rows", "subarray_cols",
                  "pe_rows", "pe_cols", "tile_rows", "tile_cols"):
            if cp.has_option("HW", k):
                setattr(hw, k, cp.getint("HW", k))

    sim = SimCfg()
    if cp.has_section("SIM"):
        for k in ("mode", "settling_reducer", "dc_input_encoding",
                  "on_job_failure"):
            if cp.has_option("SIM", k):
                setattr(sim, k, cp.get("SIM", k).strip().lower())
        if cp.has_option("SIM", "process_corner"):
            sim.process_corner = cp.get("SIM", "process_corner").strip()
        for k in ("license_count", "postprocess_workers", "hybrid_transient_seed"):
            if cp.has_option("SIM", k):
                setattr(sim, k, cp.getint("SIM", k))
        for k in ("pulse_width", "trise", "tfall", "tran_maxstep",
                  "settling_window_lo", "settling_window_hi",
                  "vdd", "vss", "temperature", "tnom",
                  "tia_calib_p_lo", "tia_calib_p_hi",
                  "hybrid_transient_fraction", "spectre_timeout_s",
                  "fastdc_T_min", "fastdc_T_max", "fastdc_T_step",
                  "fastdc_V_min", "fastdc_V_max", "fastdc_V_step",
                  "array_char_V_min", "array_char_V_max",
                  "array_char_V_step", "array_char_T_step",
                  "array_char_timeout_s", "cell_char_timeout_s"):
            if cp.has_option("SIM", k):
                setattr(sim, k, cp.getfloat("SIM", k))
        for k in ("array_char_size_primary", "array_char_size_secondary",
                  "array_char_probe_column",
                  "auto_transient_image_index", "auto_transient_seed",
                  "auto_transient_num_images"):
            if cp.has_option("SIM", k):
                setattr(sim, k, cp.getint("SIM", k))
        if cp.has_option("SIM", "auto_transient_image"):
            sim.auto_transient_image = cp.get("SIM", "auto_transient_image").strip()
        if cp.has_option("SIM", "hwsimdb_retention"):
            sim.hwsimdb_retention = cp.get("SIM", "hwsimdb_retention").strip().lower()
        for k in ("pipeline_postprocess",):
            if cp.has_option("SIM", k):
                setattr(sim, k, _get_bool(cp.get("SIM", k)))
        if cp.has_option("SIM", "spectre_args"):
            sim.spectre_args = tuple(cp.get("SIM", "spectre_args").split())

    paths = PathsCfg()
    if cp.has_section("PATHS"):
        for k in ("project_dir", "images_dir", "dump_root",
                  "hwsimdb_root", "tmp_root",
                  "golden_input_transient", "golden_input_dc",
                  "stdcell_netlist",
                  "device_model_transient", "device_model_dc",
                  "transistor_model", "char_table_path",
                  "array_table_path", "peripheral_netlist"):
            if cp.has_option("PATHS", k):
                setattr(paths, k, cp.get("PATHS", k).strip())
    resolve_paths(paths, os.path.dirname(os.path.abspath(path)))

    # --- [PERIPHERALS] (energy model) ---
    periph = PeripheralsCfg()
    if cp.has_section("PERIPHERALS"):
        for k in ("adc_energy_model",):
            if cp.has_option("PERIPHERALS", k):
                setattr(periph, k, cp.get("PERIPHERALS", k).strip())
        for k in ("dac_bits", "n_bit_slices", "adc_bits", "tia_calib_images"):
            if cp.has_option("PERIPHERALS", k):
                setattr(periph, k, cp.getint("PERIPHERALS", k))
        for k in ("tech_node_nm", "adc_fom_fJ_per_convstep", "adc_enob",
                  "dac_energy_pJ_per_conv", "dac_cap_fF", "dac_vref_V",
                  "dac_switching_activity", "tia_bias_current_A",
                  "tia_supply_V", "tia_active_time_s",
                  "mux_energy_pJ_per_op", "mux_cap_fF", "mux_vdd_V",
                  "adc_i_fullscale", "subtractor_energy_pJ_per_op",
                  "subtractor_cap_fF", "subtractor_vdd_V",
                  "vmin_tia", "vmax_tia", "vmax_adc"):
            if cp.has_option("PERIPHERALS", k):
                setattr(periph, k, cp.getfloat("PERIPHERALS", k))
        for k in ("generate_energy_report", "adc_per_slice",
                  "differential_columns", "enable_qerror"):
            if cp.has_option("PERIPHERALS", k):
                setattr(periph, k, _get_bool(cp.get("PERIPHERALS", k)))

    # --- [LATENCY] (cycle-accurate model) ---
    latency = LatencyCfg()
    if cp.has_section("LATENCY"):
        for k in ("adc_type", "periph_delay_model"):
            if cp.has_option("LATENCY", k):
                setattr(latency, k, cp.get("LATENCY", k).strip())
        for k in ("num_dac", "num_adc", "adc_cycles_per_conv",
                  "dac_cycles_per_conv", "mac_cycles_per_vector",
                  "vmm_parallel_subarrays", "adc_osr", "mux_cycles"):
            if cp.has_option("LATENCY", k):
                setattr(latency, k, cp.getint("LATENCY", k))
        for k in ("clock_freq_hz", "ref_node_nm", "node_delay_exponent",
                  "corner_delay_ff", "corner_delay_ss",
                  "temp_delay_coeff_per_C", "settle_band"):
            if cp.has_option("LATENCY", k):
                setattr(latency, k, cp.getfloat("LATENCY", k))

    # --- [SUBCIRCUITS] (role -> subckt_name, S|H) ---
    # Defaults preserve the current all-software peripherals; CELL and
    # WL_DECODER_DRIVER are hardware (they're already real subckts in
    # standard_cell.netlist).
    subcircuits = SubcircuitsCfg(entries={
        "multiplexer":  ("MUX",                       "S"),
        "dac":          ("DAC_8b",                    "S"),
        "adc":          ("ADC_8b",                    "S"),
        "tia":          ("TIA_8b",                    "S"),
        "subtractor":   ("SUBTRACTOR_8b",             "S"),
        "cell":         ("rram_cell_1T1R_180N_STD",   "H"),
        "bl_driver":    ("BL_DRIVER",                 "H"),
    })
    if cp.has_section("SUBCIRCUITS"):
        for role, raw in cp.items("SUBCIRCUITS"):
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 2 and parts[0]:
                name, mode = parts[0], parts[1].upper()
            elif len(parts) == 1 and parts[0]:
                # only a mode given -> keep existing/role name, set mode
                existing = subcircuits.entries.get(role, (role, "S"))
                if parts[0].upper() in ("S", "H"):
                    name, mode = existing[0], parts[0].upper()
                else:
                    name, mode = parts[0], "S"
            else:
                continue
            if mode not in ("S", "H"):
                mode = "S"
            subcircuits.entries[role.strip().lower()] = (name, mode)

    # --- sanity checks ---
    if sim.mode not in ("transient", "dc", "hybrid", "fastdc", "auto"):
        raise ValueError(
            f"[SIM] mode must be 'transient'|'dc'|'fastdc'|'hybrid'|'auto', "
            f"got '{sim.mode}'"
        )
    if not (0.0 <= sim.settling_window_lo < sim.settling_window_hi <= 1.0):
        raise ValueError(
            f"[SIM] settling_window_[lo,hi] must satisfy 0 <= lo < hi <= 1, "
            f"got [{sim.settling_window_lo}, {sim.settling_window_hi}]"
        )
    if sim.license_count < 1:
        raise ValueError(f"[SIM] license_count must be >= 1, got {sim.license_count}")
    if sim.dc_input_encoding not in ("sweep", "per_image"):
        raise ValueError(
            f"[SIM] dc_input_encoding must be 'sweep'|'per_image', "
            f"got '{sim.dc_input_encoding}'"
        )
    if sim.on_job_failure not in ("raise", "nan"):
        raise ValueError(f"[SIM] on_job_failure must be 'raise'|'nan', "
                         f"got '{sim.on_job_failure}'")
    if sim.settling_reducer not in ("median", "mean", "last"):
        raise ValueError(f"[SIM] settling_reducer must be 'median'|'mean'|'last', "
                         f"got '{sim.settling_reducer}'")
    if sim.hwsimdb_retention not in ("all", "last_block", "none"):
        raise ValueError(f"[SIM] hwsimdb_retention must be 'all'|'last_block'|'none', "
                         f"got '{sim.hwsimdb_retention}'")
    if sim.trise + sim.tfall >= sim.pulse_width:
        raise ValueError(f"[SIM] trise + tfall ({sim.trise + sim.tfall:g} s) must be "
                         f"shorter than pulse_width ({sim.pulse_width:g} s)")
    # The settled plateau of each PWL window is [(pw+trise)/W, 2pw/W] with
    # W = 3pw+trise+tfall (sim.templates); warn if the reduction window leaves it.
    _W = 3 * sim.pulse_width + sim.trise + sim.tfall
    _lo, _hi = (sim.pulse_width + sim.trise) / _W, 2 * sim.pulse_width / _W
    if sim.settling_window_lo < _lo - 1e-9 or sim.settling_window_hi > _hi + 1e-9:
        import warnings
        warnings.warn(f"[SIM] settling window [{sim.settling_window_lo}, "
                      f"{sim.settling_window_hi}] extends outside the pulse plateau "
                      f"[{_lo:.3f}, {_hi:.3f}] for these pulse/edge times; the settled "
                      f"current will include edge samples.")

    # --- [PDK] (process/transistor/VA params; was hardcoded in device_emit) ---
    pdk = PdkCfg()
    if cp.has_section("PDK"):
        str_keys = ("transistor_model_name", "transistor_w", "transistor_l",
                    "transistor_as", "transistor_ad", "transistor_ps",
                    "transistor_pd", "transistor_m", "va_cell_transient",
                    "va_cell_dc", "cell_prefix")
        for k in str_keys:
            if cp.has_option("PDK", k):
                setattr(pdk, k, cp.get("PDK", k).strip())
        float_keys = ("va_area_ox", "va_eact_generation", "va_eact_recombination",
                      "va_velocity_k1", "va_gamma_k0", "va_gamma_k1",
                      "va_temperature_0", "va_c_thermal", "va_tau_thermal",
                      "va_t_step", "wl_read_V")
        for k in float_keys:
            if cp.has_option("PDK", k):
                setattr(pdk, k, cp.getfloat("PDK", k))
        if cp.has_option("PDK", "va_enable_switching"):
            pdk.va_enable_switching = cp.getint("PDK", "va_enable_switching")

    # --- [PARASITICS] (crossbar wire R/C) ---
    parasitics = ParasiticsCfg()
    if cp.has_section("PARASITICS"):
        for k in ("r_wire_per_cell_ohm", "c_wire_per_cell_fF", "r_wl_per_cell_ohm"):
            if cp.has_option("PARASITICS", k):
                setattr(parasitics, k, cp.getfloat("PARASITICS", k))
        for k in ("enable", "model_wl", "driver_at_origin"):
            if cp.has_option("PARASITICS", k):
                setattr(parasitics, k, _get_bool(cp.get("PARASITICS", k)))

    # --- [ACCURACY] (Torch/IMC/GT report; extensible task) ---
    accuracy = AccuracyCfg()
    if cp.has_section("ACCURACY"):
        for k in ("task", "labels_file", "label_from_filename"):
            if cp.has_option("ACCURACY", k):
                setattr(accuracy, k, cp.get("ACCURACY", k).strip())
        if cp.has_option("ACCURACY", "generate_accuracy_report"):
            accuracy.generate_accuracy_report = _get_bool(
                cp.get("ACCURACY", "generate_accuracy_report"))
        if cp.has_option("ACCURACY", "topk"):
            accuracy.topk = tuple(int(x) for x in
                                  cp.get("ACCURACY", "topk").replace(" ", "").split(",") if x)

    # --- [INTERCONNECT] (tile loading + NoC) ---
    ic = InterconnectCfg()
    if cp.has_section("INTERCONNECT"):
        for k in ("enable", "loading", "noc_overlap", "noc_multicast"):
            if cp.has_option("INTERCONNECT", k):
                setattr(ic, k, _get_bool(cp.get("INTERCONNECT", k)))
        for k in ("cell_pitch_x_um", "cell_pitch_y_um", "subarray_gap_um",
                  "tile_gap_um", "wire_r_ohm_per_um", "wire_c_fF_per_um",
                  "driver_r_out_ohm", "bus_tap_c_fF",
                  "tia_open_loop_gain", "loading_tol_V",
                  "noc_clock_hz", "noc_router_energy_pJ_per_bit",
                  "noc_link_energy_pJ_per_bit_mm"):
            if cp.has_option("INTERCONNECT", k):
                setattr(ic, k, cp.getfloat("INTERCONNECT", k))
        if cp.has_option("INTERCONNECT", "tia_r_in_ohm"):
            raw = cp.get("INTERCONNECT", "tia_r_in_ohm").strip().lower()
            ic.tia_r_in_ohm = -1.0 if raw == "auto" else float(raw)
        for k in ("loading_max_passes", "noc_flit_bits", "noc_packet_flits",
                  "noc_router_cycles", "noc_link_cycles", "noc_mesh_cols"):
            if cp.has_option("INTERCONNECT", k):
                setattr(ic, k, cp.getint("INTERCONNECT", k))
        for k in ("noc", "noc_buffer", "booksim_binary", "input_bus"):
            if cp.has_option("INTERCONNECT", k):
                setattr(ic, k, cp.get("INTERCONNECT", k).strip())
        ic.noc = ic.noc.lower()
        ic.noc_buffer = ic.noc_buffer.lower()
        ic.input_bus = ic.input_bus.lower()
    if ic.input_bus not in ("buffered", "shared"):
        raise ValueError(f"[INTERCONNECT] input_bus must be 'buffered'|'shared', "
                         f"got '{ic.input_bus}'")
    if ic.noc not in ("none", "analytic", "booksim"):
        raise ValueError(f"[INTERCONNECT] noc must be 'none'|'analytic'|'booksim', "
                         f"got '{ic.noc}'")
    if ic.noc_buffer not in ("center", "corner"):
        raise ValueError(f"[INTERCONNECT] noc_buffer must be 'center'|'corner', "
                         f"got '{ic.noc_buffer}'")
    if ic.loading_max_passes < 1:
        raise ValueError(f"[INTERCONNECT] loading_max_passes must be >= 1, "
                         f"got {ic.loading_max_passes}")
    if ic.booksim_binary:
        ic.booksim_binary = os.path.expanduser(ic.booksim_binary)
        if not os.path.isabs(ic.booksim_binary):
            ic.booksim_binary = os.path.normpath(
                os.path.join(paths.project_dir, ic.booksim_binary))

    return Cfg(data=data, model=model, train=train, save=save,
               hw=hw, sim=sim, paths=paths,
               peripherals=periph, latency=latency,
               subcircuits=subcircuits, pdk=pdk, parasitics=parasitics,
               accuracy=accuracy, interconnect=ic)