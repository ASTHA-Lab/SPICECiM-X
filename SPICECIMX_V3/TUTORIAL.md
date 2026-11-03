# SPICECiM-X Tutorial

SPICE-accurate simulation of Vision Transformer (ViT) inference on ReRAM (1T1R)
analog in-memory-computing crossbars. This tutorial goes from a trained
checkpoint and a config file to accuracy and energy reports. It then covers
every config section, the simulation modes, hardware peripherals, outputs,
how to switch datasets and how to debug a run.

All commands are run from the **repository root**. The reference
configuration is `config_cifar10.ini`, which is annotated line by line. The
simulator package is in `SPICECIMX_V3/spicecimx/`.

---

## 1. Overview and pipeline

SPICECiM-X maps every static-weight matrix multiplication of the ViT onto
differential 1T1R crossbars. That covers Q/K/V, W_O, FFN W1/W2 and the
classifier head. Softmax, LayerNorm and the activation × activation products
run digitally. Each matmul is tiled into
`subarray_rows × subarray_cols` subarrays. The +column and −column subarrays
are simulated with Cadence Spectre, or looked up from SPICE-built tables in
`fastdc`. The jobs run in parallel across your Spectre licenses. The column
currents go through the TIA → subtractor → ADC, and an affine de-map turns
them back into floating point. A voltage pre-warp `h⁻¹(V)`, built from the
array characterisation, cancels the 1T1R I–V non-linearity and the
bit-line-driver loading.

```
  python main.py --config config_cifar10.ini           train the ViT -> [SAVE] out_dir/best.pt, last.pt
                 │
                 ▼
  spicecimx.cli.characterize        (1T1R cell table, fastdc)       one-time
  spicecimx.cli.characterize_array  (subarray table, pre-warp)      one-time, cached by config hash,
                 │                                                  auto-built by infer if missing/stale
                 ▼
  spicecimx.cli.infer               SPICE-accurate forward pass
                 │
                 ▼
  <dump_root>/<batch>/  Final_report.txt, accuracy_report.json,
                        power_energy_latency_report.{json,csv}
```

`<batch>` is the leaf folder name of `images_dir`. For example,
`Test_Image/t1_ds` gives `t1_ds`.

The hardware sections (`[HW]`, `[SIM]`, `[PDK]`, `[PARASITICS]`,
`[PERIPHERALS]`, `[LATENCY]`, `[SUBCIRCUITS]`) do not depend on the dataset.
The workload sections (`[DATA]`, `[MODEL]`, `[ACCURACY]`) plus the checkpoint
are what change when you switch datasets.

---

## 2. Installation

Requirements:

* Python ≥ 3.9 with `numpy` and `pandas`. Training and inference also need
  `torch`, `torchvision` and `Pillow`.
* **Cadence Spectre** (tested with Spectre 24.1) on `PATH`. Every mode needs
  it except `python -m spicecimx.cli.run --selftest`. `fastdc` inference
  reads tables only, but Spectre builds those tables once.
* Your PDK's Spectre transistor models, set in `[PATHS] transistor_model`.
  The examples use Cadence GPDK 45 nm / 180 nm, which cannot be
  redistributed.
* Tkinter, only for the GUI. On Linux it is an OS package such as
  `python3-tk`, not a pip package.

```bash
pip install -e "SPICECIMX_V3[torch,test]"
```

The package directory is `SPICECIMX_V3`, and the name is case-sensitive on
Linux. The extras are `torch` (torch, torchvision, Pillow), `test` (pytest)
and `all`.

Sanity check without Spectre or torch:

```bash
python -m spicecimx.cli.run --selftest
```

---

## 3. Configuration

One INI file drives the trainer, the simulator and the GUI. `[DATA]`,
`[MODEL]`, `[TRAIN]` and `[SAVE]` are required. Every other key has a default
in `spicecimx/config.py`, so you can omit it. The tables below give the
**code default**, meaning the value used when the key is absent. Where
`config_cifar10.ini` uses a different value, the table notes it as *ref*.
Inline `#` / `;` comments are allowed, and `%` is literal.

### Path resolution

Every relative `[PATHS]` entry is resolved against `project_dir`.
`project_dir` itself is resolved against the **folder that holds the config
file**, not the current directory. So `config_cifar10.ini` with
`project_dir = .` works from any working directory. Paths given on the
command line (`--images_dir`, `--dump_root`, `--checkpoint`, `--project`)
are relative to the current directory. `project_dir` must also contain the
trainer code (`main.py` and `utils/`), because `infer` imports `build_model`
and the config loader from there.

### `[DATA]`, `[MODEL]`, `[TRAIN]` (required, read by the trainer)

These must match the checkpoint. Reference values:

| key | ref | meaning |
|---|---|---|
| `dataset` | `cifar10` | `cifar10`, `mnist`, `fashion_mnist` (also `fashion`, `fashionmnist`, `mnist-fashion`) |
| `data_root` | `./data` | torchvision download/cache folder |
| `img_height`, `img_width` | `32`, `32` | input size; inference resizes test images to it |
| `channels` | `auto` | `auto` = 1 for *mnist* datasets, else 3. At inference the channel count is taken from the checkpoint (`patch_proj.weight`) |
| `val_split`, `num_workers`, `augment` | `0.1`, `4`, `true` | trainer only |
| `patch_size` | `4` | must divide `img_height` and `img_width` |
| `d_model`, `n_heads`, `depth` | `192`, `6`, `9` | ViT width / heads / blocks |
| `d_k`, `d_v` | `0`, `0` | per-head dims (0 = `d_model / n_heads`) |
| `mlp_ratio`, `dropout` | `2.0`, `0.1` | FFN expansion; dropout (inactive at inference) |
| `num_classes` | `10` | classifier width |
| `use_class_token`, `pos_embed` | `true`, `learnable` | CLS token / positional embedding |
| `[TRAIN]` `epochs`, `batch_size`, `lr`, `weight_decay`, `betas`, `warmup_epochs`, `lr_scheduler`, `seed`, `amp`, `device`, `log_interval` | see file | trainer only |

### `[SAVE]`

| key | default | meaning |
|---|---|---|
| `out_dir` | required (ref `./cifar10/cifar10_large_vit_outputs`) | trainer writes `best.pt` / `last.pt` here; `infer` looks here first |
| `save_weights`, `save_best` | required | trainer |
| `clean_workspace` | `true` | at the start of each `infer` run, clear **only this batch's** folders: `<hwsimdb_root>/<batch>`, `<dump_root>/<batch>` and `<tmp_root>/<batch>`. They are moved to `<project_dir>/trash/<run_id>` and deleted in the background. The run refuses a target that is or contains the project dir, the images, `supports/`, the table files or `out_dir`. Other batches and the characterisation tables are never touched |

### `[HW]`: device and crossbar

| key | default | meaning |
|---|---|---|
| `Tox`, `Tref`, `Ik1`, `Vref`, `Tmax`, `Tmin` | `5.0e-9`, `4.7249e-9`, `6.14e-5`, `0.43`, `4.9e-9`, `3.3e-9` | ReRAM physics; `Tmin..Tmax` is the programmable filament range (HRS..LRS) |
| `R_access` | `auto` (`-1`) | effective per-cell series resistance (Ω). `auto` fits it from the array table; a number uses it as given |
| `res_val`, `cap_val` | `0`, `0` | per-cell R/C inside the transient cell model |
| `subarray_rows`, `subarray_cols` | `16`, `16` | subarray size |
| `pe_rows`, `pe_cols`, `tile_rows`, `tile_cols` | `2` each | subarrays per PE, PEs per tile |
| `vmin`, `vmax` | `0.0`, `1.0` (ref `0.6`) | **crossbar drive range**. Activations are encoded as `V = a·A + b` within `[vmin, vmax]` in every mode. It is also the H-DAC full scale |

### `[SIM]`: simulator and operating point

| key | default | meaning |
|---|---|---|
| `mode` | `transient` (ref `auto`) | `transient` \| `dc` \| `hybrid` \| `fastdc` \| `auto` (case-insensitive). See §5 |
| `process_corner` | `stat` (ref `tt`) | section name in the transistor model `.scs` |
| `vdd`, `vss` | `1.8`, `0.0` (ref `1.1`) | supply rails (V) |
| `temperature`, `tnom` | `25.0`, `27.0` | ambient and nominal model temperature (°C) |
| `license_count` | `1` (ref `16`) | max concurrent Spectre jobs (≥ 1) |
| `pipeline_postprocess` | `true` | post-process subarray N while N+1 simulates |
| `postprocess_workers` | `8` | post-processing pool size |
| `spectre_args` | `+preset=mx +mt=16 …` | extra Spectre arguments (space-separated) |
| `on_job_failure` | `raise` | a Spectre job that still fails after retries either stops the run with the failing job ids (`raise`) or turns that subarray's output columns into NaN (`nan`) |
| `spectre_timeout_s` | `0` | wall-clock limit per inference Spectre job (0 = none) |
| `pulse_width`, `trise`, `tfall` | `1e-9`, `1e-12`, `1e-12` | transient PWL drive per input vector. `trise + tfall` must be `< pulse_width` |
| `settling_window_lo`, `settling_window_hi` | `0.40`, `0.65` | fraction of each PWL window that is reduced to the settled current (`0 ≤ lo < hi ≤ 1`). A warning is printed if it leaves the pulse plateau `[(pw+trise)/W, 2pw/W]` with `W = 3pw+trise+tfall` |
| `settling_reducer` | `median` | `median` \| `mean` \| `last` |
| `tran_maxstep` | `0` | maximum transient time step (s) appended to the `tran` analysis; 0 = the simulator's own step control (Spectre X presets ignore it) |
| `hybrid_transient_fraction`, `hybrid_transient_seed` | `0.0`, `0` | hybrid mode: fraction of subarrays also simulated in transient, and the RNG seed |
| `dc_input_encoding` | `sweep` | validated (`sweep` \| `per_image`) but currently unused; the DC backend always uses the parametric sweep |
| `tia_calib_p_lo`, `tia_calib_p_hi` | `0.1`, `99.9` | percentiles used for the per-layer TIA/ADC current range |
| `auto_transient_image` | empty | `auto` mode: run the transient pass on exactly this file name |
| `auto_transient_image_index` | `-1` | `auto`: ≥ 0 picks that index in the sorted image list (legacy) |
| `auto_transient_num_images`, `auto_transient_seed` | `1`, `0` | `auto`: number of distinct random images for the transient pass (crossbar energy is averaged over them) and their seed |
| `hwsimdb_retention` | `all` (ref `last_block`) | transient raw data kept: `all`, `last_block` (reduce, then delete earlier blocks' raw files) or `none` |
| `fastdc_T_min/T_max/T_step` | `3.3e-9`, `4.9e-9`, `0.01e-9` | cell-table T grid (`characterize`) |
| `fastdc_V_min/V_max/V_step` | `0.0`, `1.0`, `0.01` | cell-table V grid. An `.npz` from `characterize` stores its own grid. **For a `.csv` table it must match the CSV's column count**, e.g. 0..1.0 V step 0.01 = 101 columns |
| `array_char_V_min/V_max/V_step` | `0.0`, `1.0`, `0.01` (ref `0.0`, `0.6`, `0.1`) | array-table bit-line voltage grid |
| `array_char_T_step` | `0.01e-9` (ref `0.1e-9`) | array-table T step (the T range is `[HW] Tmin..Tmax`) |
| `array_char_size_primary`, `array_char_size_secondary` | `16`, `32` | the two subarray sizes characterised (interpolated in between) |
| `array_char_probe_column` | `0` | which column probe to record |
| `array_char_timeout_s`, `cell_char_timeout_s` | `7200`, `3600` | per-call timeout of the one-time characterisations |

### `[PATHS]`

All relative entries resolve against `project_dir` (see above).

| key | default | meaning |
|---|---|---|
| `project_dir` | `.` | relative to the config file's folder |
| `images_dir` | `./Test_Image` (ref `./Test_Image/t1`) | test images, searched recursively (`.jpg .jpeg .png .bmp .tif .tiff`); its leaf name is `<batch>` |
| `dump_root` | `imc_traces` | reports and per-image dumps (`<dump_root>/<batch>/`) |
| `hwsimdb_root`, `tmp_root` | `./hwsimdb`, `./tmp` | weight tiles + raw Spectre output; per-job scratch |
| `golden_input_transient`, `golden_input_dc` | `./supports/_golden_input.scs`, `./supports/_golden_input_dc.scs` | testbench templates |
| `stdcell_netlist` | `./supports/standard_cell.netlist` (ref `standard_cell_45nm.netlist`, which you create for your PDK; see §4.0) | std cells (1T1R cell, BL_DRIVER, decoders, …) |
| `device_model_transient`, `device_model_dc` | `./supports/reram_cell.v`, `./supports/sky130_dc_memristor_dc.v` | ReRAM Verilog-A |
| `transistor_model` | empty | your PDK's Spectre model file (**required** for any SPICE run) |
| `char_table_path` | `./supports/cell_table.npz` (ref `cell_table_45nm.npz`) | fastdc cell table, `.npz` (written by `characterize`) or `.csv` |
| `array_table_path` | `./supports/array_table.npz` (ref `array_table_45nm_tt.npz`) | array table, auto-built and cached |
| `peripheral_netlist` | empty | optional netlist with the H-mode DAC/TIA/SUBTRACTOR/ADC subckts (ref `./supports/peripherals/peripherals_8b.scs`); only used when a converter role is `H` |

### `[PERIPHERALS]`: readout and energy models

| key | default | meaning |
|---|---|---|
| `generate_energy_report` | `false` (ref `true`) | write `power_energy_latency_report.{json,csv}` |
| `tech_node_nm` | `180` (ref `45`) | node for the accelergy ADC model and scaling |
| `dac_bits` | `8` | DAC resolution (energy model) |
| `adc_bits` | `16` (ref `8`) | ADC resolution (software ADC model) |
| `vmin_tia`, `vmax_tia` | `0.0`, `1.0` (ref `0.6`) | TIA output window; each layer's calibrated current range maps onto it |
| `vmax_adc` | `0.95` (ref `0.6`) | **ADC full-scale voltage**: codes span `0..vmax_adc` |
| `enable_qerror` | `false` | add ADC quantisation error |
| `tia_calib_images` | `1` | per-layer TIA/ADC range learned from the first N calls (images; one call per batch with `--batch_all`), then frozen. `0` = never freeze, so results then depend on image order. Calibration and polarity are kept **per block and layer** (key `block_XX/<tag>`) |
| `adc_energy_model` | `walden` (ref `accelergy`) | `walden` (FOMW·2^ENOB) or `accelergy` (node-aware) |
| `adc_fom_fJ_per_convstep`, `adc_enob` | `22.5`, `-1` | Walden FOM; ENOB (< 0 = `adc_bits`) |
| `adc_i_fullscale` | `8e-4` | ± current range of the H-mode readout characterisation sweep (A) |
| `dac_energy_pJ_per_conv`, `dac_cap_fF`, `dac_vref_V`, `dac_switching_activity` | `-1`, `500` (ref `50`), `0.2`, `1/3` | measured pJ/conv if > 0, else switched-cap model |
| `tia_bias_current_A`, `tia_supply_V`, `tia_active_time_s` | `20e-6`, `1.8` (ref `1.1`), `-1` | static-bias TIA model (`-1` = one clock period) |
| `mux_energy_pJ_per_op`, `mux_cap_fF`, `mux_vdd_V` | `-1`, `2.0`, `1.8` (ref `1.1`) | mux energy (charged only when `num_dac`/`num_adc` < a layer's I/O width) |
| `differential_columns` | `true` | 2 TIAs + 1 subtractor per output |
| `subtractor_energy_pJ_per_op`, `subtractor_cap_fF`, `subtractor_vdd_V` | `-1`, `10.0`, `1.8` (ref `1.1`) | subtractor energy |
| `n_bit_slices`, `adc_per_slice` | `1`, `false` | bit-slicing op counts |
| `track_operating_point`, `nominal_vdd_V` | `false`, `0` (ref `true`, `1.1`) | scale modelled peripheral energy with `[SIM] vdd` (`0` = per-node table in `analog/tech_scale.py`) |
| `energy_node_scaling`, `energy_ref_node_nm` | `false`, `0` | project peripheral constants from a reference node to `tech_node_nm` |

### `[LATENCY]`

| key | default | meaning |
|---|---|---|
| `clock_freq_hz` | `1e9` (ref `500e6`) | clock |
| `num_dac`, `num_adc` | `16`, `16` (ref `8`, `8`) | parallel converters |
| `adc_type` | `sar` | `sar` (bits+1 cycles), `flash`/`pipeline` (1), `integrating` (2^bits), `sigma_delta` (OSR) |
| `adc_cycles_per_conv`, `dac_cycles_per_conv` | `0`, `1` | 0 = derive from `adc_type` |
| `mac_cycles_per_vector` | `1` | VMM settle cycles per input vector |
| `vmm_parallel_subarrays` | `0` | 0 = `pe_rows·pe_cols·tile_rows·tile_cols` |
| `adc_osr`, `mux_cycles` | `0`, `0` | sigma-delta OSR (0 = 2^bits); extra mux cycles |
| `periph_delay_model` | `fixed` (ref `node_scaled`) | `node_scaled` scales converter timing with node/corner/temperature and uses the measured transient settle time |
| `ref_node_nm`, `node_delay_exponent` | `0`, `1.0` (ref `45`) | node where `clock_freq_hz` holds; `(node/ref)^e` |
| `corner_delay_ff`, `corner_delay_ss`, `temp_delay_coeff_per_C` | `0.85`, `1.30`, `0.001` | corner/temperature timing |
| `settle_band` | `0.02` | tolerance band for the measured settle |

### `[SUBCIRCUITS]`

`role = subckt_name, S|H`. See §6. Defaults: `multiplexer = MUX, S`,
`dac = DAC_8b, S`, `adc = ADC_8b, S`, `tia = TIA_8b, S`,
`subtractor = SUBTRACTOR_8b, S`, `cell = rram_cell_1T1R_180N_STD, H`,
`bl_driver = BL_DRIVER, H`.

### `[PDK]`

Everything process-specific. Changing node or cell library means editing this
section plus `[PATHS] transistor_model` and `stdcell_netlist`.

| key | default | meaning |
|---|---|---|
| `transistor_model_name` | `nmos1` (ref `g45n1svt`) | access-transistor device name in the model file |
| `transistor_w/l/as/ad/ps/pd/m` | `2u`, `180n`, `1.2p`, `1.2p`, `5.2u`, `5.2u`, `(1)*(1)` | instance geometry (strings with units) |
| `wl_read_V` | `2.4` (ref `1.1`) | access-transistor gate drive during read/characterisation (V) |
| `va_cell_transient`, `va_cell_dc` | `sky130_fd_pr_reram__reram_cell`, `sky130_dc_memristor_dc` | Verilog-A module names |
| `cell_prefix` | `rram_cell_1T1R_180N_STD` | 1T1R cell subckt name actually emitted into the crossbar |
| `va_area_ox`, `va_eact_generation`, `va_eact_recombination`, `va_velocity_k1`, `va_gamma_k0`, `va_gamma_k1`, `va_temperature_0`, `va_c_thermal`, `va_tau_thermal`, `va_t_step` | see `config.py` | remaining Verilog-A parameters |
| `va_enable_switching` | `0` | `0` = read mode (filament frozen at the programmed weight); `1` = full SET/RESET dynamics (cells drift during long reads). Passed to `supports/reram_cell.v` as `enable_switching` |

The transient and DC decks both take the `[PDK]` geometry and VA parameters,
the `[HW]` device parameters and `[PARASITICS]`. So a non-default PDK applies
in `dc` mode as well as in `transient`.

### `[PARASITICS]`

| key | default | meaning |
|---|---|---|
| `enable` | `false` | per-cell wire R/C on bit and source lines (IR drop) |
| `r_wire_per_cell_ohm`, `c_wire_per_cell_fF` | `2.5`, `0.2` | per-segment R and node C |
| `model_wl`, `r_wl_per_cell_ohm` | `false`, `2.5` | also word-line R |
| `driver_at_origin` | `true` | driver at the index-0 end of each line |

Parasitics are part of the array-table hash, so changing them rebuilds the
table.

### `[INTERCONNECT]`

What the RSN decomposition leaves out (§5.1): the lines that join a layer's
subarrays, and the network that joins the tiles. Off by default for
back-compatibility. The reference config turns it on.

| key | default (ref) | meaning |
|---|---|---|
| `enable` | `false` (`true`) | master switch |
| `loading` | `true` | solve the shared column lines (and the input bus if `shared`) and re-simulate the RSN decks with the boundary voltages (transient). fastdc applies the same line solve to its table currents |
| `input_bus` | `buffered` | `buffered`: row drivers per subarray (Fig. 2), so the DAC bus is capacitive only. `shared`: one unbuffered pass-gate line carries every subarray's row current |
| `loading_max_passes`, `loading_tol_V` | `2`, `1e-3` | SPICE passes (1 = ideal RSN plus a predicted correction only) and the boundary residual that stops early |
| `cell_pitch_x_um`, `cell_pitch_y_um` | `0` = auto | cell pitch across columns (W + 4F) and rows (8F) |
| `subarray_gap_um`, `tile_gap_um` | `5`, `20` | routing gap between subarrays, extra gap at PE/tile boundaries |
| `wire_r_ohm_per_um`, `wire_c_fF_per_um` | `0` = auto, `0.2` | inter-subarray line R (auto: semi-global metal, a quarter of the minimum-pitch table in `arch/floorplan.py`) and C |
| `driver_r_out_ohm`, `bus_tap_c_fF` | `50`, `2` | DAC/bus-driver output R (delay; DC drop when `shared`), input C of each row driver |
| `tia_r_in_ohm`, `tia_open_loop_gain` | `auto`, `100` | TIA virtual-ground input R; auto = (vmax_tia − vmin_tia)/adc_i_fullscale/(1 + A) |
| `noc` | `analytic` | `none`, `analytic`, or `booksim` (analytic + a BookSim2 validation of the unicast traffic) |
| `noc_multicast` | `true` | send the input shared by a block's Q/K/V matmuls once (XY multicast tree) |
| `noc_flit_bits`, `noc_packet_flits`, `noc_clock_hz` | `64`, `4`, `1e9` | flit width, packet size, NoC clock |
| `noc_router_cycles`, `noc_link_cycles` | `2`, `1` | per-hop pipeline |
| `noc_router_energy_pJ_per_bit`, `noc_link_energy_pJ_per_bit_mm` | `-1` = auto | router energy (BookSim2/Orion 32 nm, 0.110 pJ/bit, C·V² scaled to the node) and link energy (0.5·C·V² per bit per mm) |
| `noc_buffer`, `noc_mesh_cols` | `center`, `0` | global-buffer router position, mesh width (0 = smallest square) |
| `noc_overlap` | `false` | `true`: only the NoC pipeline fill adds to latency |
| `booksim_binary` | empty | patched BookSim2 (`scripts/booksim/build_booksim.sh`) |

### `[ACCURACY]`

| key | default | meaning |
|---|---|---|
| `generate_accuracy_report` | `true` | write `accuracy_report.json` |
| `task` | `classification` | metric set (registry in `core/metrics.py`, §9) |
| `labels_file` | empty | ground truth: CSV `filename,label` or JSON map (relative to `project_dir`) |
| `label_from_filename` | empty | regex whose first group captures the label |
| `topk` | `1` (ref `1,5`) | top-k accuracies |

---

## 4. Running

### 4.0 Point the config at your PDK (once)

The repository ships no PDK data. Cadence PDK models, std cells built on
them, and the tables characterised from them cannot be redistributed. Before
the first SPICE run, set three things in your copy of `config_cifar10.ini`:

1. `[PATHS] transistor_model`: replace the `/path/to/...` placeholder with
   your PDK's Spectre model file. `[SIM] process_corner` must name one of its
   sections (`tt`, `ff`, `ss`, …).
2. `[PATHS] stdcell_netlist`: a std-cell netlist for your PDK. Start from
   `supports/standard_cell.netlist`, which uses 180 nm devices named `nmos1` /
   `pmos1` (as in Cadence GPDK180). Replace the device names and sizes, and
   add any instance parameters your models need, but keep the subckt names
   and pins (`rram_cell_1T1R_180N_STD`, `BL_DRIVER`, `INVX1`, `AND2_X1`,
   `OR2_X1`, …). The reference config expects it at
   `supports/standard_cell_45nm.netlist`.
3. `[PDK]`: `transistor_model_name` (the access NMOS in the model file, e.g.
   `g45n1svt` in GPDK045), the `transistor_w/l/...` geometry, `wl_read_V`, and
   `[SIM] vdd` for the node.

The cell table and the array table are then built by §4.2. See
`supports/README.md` for the full list.

### 4.1 Train (or bring your own checkpoint)

```bash
python main.py --config config_cifar10.ini      # writes best.pt / last.pt to [SAVE] out_dir
```

### 4.2 One-time characterisation

```bash
python -m spicecimx.cli.characterize       --config config_cifar10.ini
python -m spicecimx.cli.characterize_array --config config_cifar10.ini [--force]
```

* **`characterize`** builds the single-cell I–V table used by `fastdc`
  (and as the `dc` fallback pre-warp when no array table is set). It writes
  an `.npz` to `--output`, or to `[PATHS] char_table_path` if `--output` is
  not given. The DC ReRAM model comes from `--veriloga`, then
  `$SPICECIMX_VERILOGA`, then `[PATHS] device_model_dc`, so `--veriloga` is
  optional. Other flags: `--work_dir`, `--spectre`, `--corner`, and
  `--T_min/--T_max/--T_step/--V_min/--V_max/--V_step`, which override the
  `fastdc_*` grid.
  The reference config sets `char_table_path = ./supports/cell_table_45nm.npz`,
  so the plain command above writes the table where `infer` looks for it.
  A cell table exported from ADE as a CSV (`tinit` column, then one current
  column per bit-line voltage) also works if `fastdc_V_*` describes its
  columns.
* **`characterize_array`** builds the subarray table at the two anchor sizes.
  `infer` uses it for the pre-warp and for `R_access = auto`. The table is
  cached with a hash of the transistor model, corner, vdd/vss, temperature,
  tnom, `[HW]` device params, `[PDK]`, `[PARASITICS]`, the grid, the DC
  Verilog-A and the std-cell netlist. `infer` rebuilds it automatically
  when the hash changes. `--force` rebuilds it anyway. `R_access` is not part
  of the hash, because it is derived from the table.
* A failed Spectre run **aborts** characterisation; it never caches all-zero
  tables. The array deck drives `PGMEN`/`VDDP` the same way inference does.

### 4.3 Inference

```bash
python -m spicecimx.cli.infer --config config_cifar10.ini --images_dir Test_Image/t1_ds --limit 100
```

| flag | meaning |
|---|---|
| `--config PATH` | config file (default `config.ini`, so pass it explicitly) |
| `--mode M` | override `[SIM] mode`: `transient`, `dc`, `hybrid`, `fastdc`, `auto` |
| `--images_dir DIR` | override `images_dir` (relative to the current directory) |
| `--checkpoint PATH` | explicit checkpoint; **a missing file is an error** |
| `--project DIR` | re-root the config's relative `[PATHS]` under DIR |
| `--dump_root DIR` | override `dump_root` |
| `--licenses N` | override `license_count` |
| `--limit N` | first N images only (0 = all) |
| `--output_classes K` | keep the first K logits, for both the Torch and the IMC outputs |
| `--batch_all` | time-multiplex all images through each IMC matmul |
| `--no_intermediates` | skip per-layer CSV dumps |
| `--gui` | open the GUI instead (same as `python -m spicecimx.cli.gui --config …`) |

Checkpoint lookup without `--checkpoint` is
`<out_dir>/best.pt`, `<out_dir>/last.pt`, then `<project_dir>/outputs/best.pt`
and `<project_dir>/outputs/last.pt`. A relative `out_dir` is resolved against
`project_dir`. There is no "newest `*.pt` anywhere" fallback. The state dict
is loaded strictly: the `module.` and `_orig_mod.` prefixes are stripped, and
any missing or unexpected key is an error that tells you to check
`[MODEL]`/`[DATA]`. The input channel count is read from the checkpoint.

### 4.4 Auto mode

`mode = auto` (or `--mode auto`) runs two passes on the same batch:

1. a **transient** pass on 1..N images for power, energy and latency. The
   images are chosen by `auto_transient_image`, then
   `auto_transient_image_index`, then `auto_transient_num_images` random
   images seeded by `auto_transient_seed`. With N > 1, the crossbar energy is
   the average over the images, and each image's data goes under
   `timgXX_<name>/`.
2. a **fastdc** pass on all images for accuracy.

The workspace is cleaned once for the whole auto run.

### 4.5 GUI

`python -m spicecimx.cli.gui --config config_cifar10.ini` opens one tab per
section. In `[SUBCIRCUITS]` each role gets a name box and an S | H switch.
The toolbar has **Run**, **Stop**, **Save config**, **Report**,
**Help / Tutorial** and **Clear log**, plus a field for extra CLI arguments.
Run saves the config and starts `spicecimx.cli.infer` with the config's folder
as the working directory.

---

## 5. Simulation modes

| mode | what runs | Spectre | use it for |
|---|---|---|---|
| `transient` | full transient per (subarray, branch) with a PWL drive per input vector; settled current from the settling window | yes | energy, power, latency; time-domain effects |
| `dc` | one parametric DC sweep per (subarray, branch) covering all input vectors | yes | accuracy with real SPICE currents, faster than transient |
| `fastdc` | numpy bit-line sums over the SPICE-built cell table + array pre-warp | tables only | fast accuracy sweeps over many images |
| `hybrid` | `dc` for every subarray, plus transient on a `hybrid_transient_fraction` sample (seeded) | yes | DC-vs-transient validation (`hybrid_report.csv` per layer) |
| `auto` | transient (energy) on 1..N images + fastdc (accuracy) on all | yes | the full report in one command |

Notes:

* Every mode applies the same `[HW] vmin..vmax` drive range and the same
  pre-warp from the array table. If no array table is set, `fastdc` and `dc`
  fall back to the single-cell table.
* Crossbar energy comes from SPICE only in `transient` mode, and so in the
  transient pass of `auto`. The other modes report peripheral energy and
  latency only.
* The hybrid report only compares DC with transient. It does not correct the
  DC output.

### 5.1 Tile interconnect: loading between subarrays and the NoC

Each RSN deck is one 16×16 subarray with ideal sources and an ideal ground.
In the tile, a layer's `R × C` subarray grid (± arrays side by side) is
wired together without peripherals in between. `[INTERCONNECT]` adds what
that leaves out:

* **Column lines (loading).** The partial sums of the `R` row-blocks add on
  one shared column line into one TIA. The op ledger counts one TIA/ADC per
  output column. The summed current flows through the line R and the TIA
  input R, raising every subarray's source-line end. In transient mode each
  layer runs as a **loaded RSN**:
  1. pass 1 is the ideal RSN;
  2. a coupled line solve (`arch/loading.py`) predicts the boundary;
  3. pass 2 re-runs every deck with each column ending in its Thévenin
     equivalent: its own path resistance plus a source for the other
     subarrays' contribution;
  4. the exact line response to the measured currents is checked. A third
     pass (secant update) runs if the residual exceeds `loading_tol_V`.

  The accepted pass replaces the waveforms, so crossbar energy and settle
  time include the loading. fastdc applies the same line solve to its
  table currents.
* **Input bus.** `buffered` (default): row drivers per subarray, so the DAC
  bus only charges wire plus driver inputs (C·V² per vector, Elmore delay).
  `shared`: the bus is resistive and is solved together with the columns.
* **Line capacitance.** C·V² of the buses and column lines per vector, and
  the Elmore settle (×ln(1/`settle_band`)) added to the MVM settle time.
* **NoC.** Every matmul's tiles sit on a mesh with the host/global buffer
  (`arch/noc.py`). Inputs go buffer → the layer's left tiles and outputs go
  bottom tiles → buffer, so the volume equals the DAC/ADC conversion counts.
  Energy = routers × router energy + links × length × link energy. Latency
  = serialisation at the buffer port + pipeline fill. `booksim` additionally
  replays the unicast traffic in BookSim2 as a validation.

Validate the loading model against one monolithic Spectre deck of a whole
layer:

```bash
python -m spicecimx.cli.check_loading --config config_cifar10.ini --k 64 --n 32 --m 4
```

It reports the column-current error of the ideal and the loaded RSN against
the monolithic deck, the energies (deck + line terms) and the settle times.

BookSim2 (optional): `bash SPICECIMX_V3/scripts/booksim/build_booksim.sh <dir>` clones,
patches (trace-matrix traffic) and builds it. Then set
`booksim_binary = <dir>/src/booksim`.

---

## 6. Peripherals: software (S) or hardware (H)

`[SUBCIRCUITS]` declares each converter as `name, S` or `name, H`. `S` uses
the analytic TIA/ADC model in `analog/tia_adc.py` and the `[PERIPHERALS]`
energy models. `H` uses a real subckt. H converters work in **every** mode.
At the start of an inference run they are characterised once with Spectre:

* **DAC**: a code sweep. The measured code → voltage table then drives the
  crossbar inputs. Its full scale is `[HW] vmin..vmax`.
* **TIA → SUBTRACTOR → ADC**: one chained current sweep over
  `±adc_i_fullscale`. At inference, a per-layer programmable TIA gain maps
  each layer's calibrated range `[Imin, Imax]` onto that sweep. The current
  then passes through the measured transfer, and the ADC codes are the
  output.

The measured energies (`measured_converter_energy.json`) replace the modelled
ones in the energy report. A failed H characterisation stops the run; the run
never falls back silently to S.

Ready-made library: `supports/peripherals/peripherals_8b.scs` defines
`DAC_8b`, `TIA_8b`, `SUBTRACTOR_8b` and `ADC_8b`. They are behavioural,
power-annotated Verilog-A cores from the EnPower project (`dac_pwr.va`,
`opamp_pwr.va`, `sar_adc8_pwr.va`). They are sized for 45 nm / 1.1 V,
`[HW] vmax = 0.6` and `adc_i_fullscale = 8e-4`. Retune their parameters if you
change those values.

```ini
[PATHS]
peripheral_netlist = ./supports/peripherals/peripherals_8b.scs
[SUBCIRCUITS]
dac = DAC_8b, H
tia = TIA_8b, H
subtractor = SUBTRACTOR_8b, H
adc = ADC_8b, H
```

Rules:

* Pin names are discovered from the netlist. A transistor-level subckt with
  the same pins drops in unchanged: DAC `INP<7:0>`, `OUT`; TIA `IN`, `OUT`;
  subtractor `INP`, `INN`, `OUT`; ADC `IN`, `CLK`, `OUT<7:0>`; all also have
  `VDD`, `VSS`.
* `tia = H` and `subtractor = H` only take effect together with `adc = H`,
  because the chain is characterised as one. With `adc = H`, the chain uses
  the `tia` and `subtractor` subckts named in `[SUBCIRCUITS]`, even when those
  roles are `S`.
* `multiplexer = H` is not simulated. The mux stays an energy model.
* The `cell` and `bl_driver` rows are informational. The crossbar always
  uses `[PDK] cell_prefix` and the std-cell `BL_DRIVER`. `infer` prints a
  NOTE for each setting that has no effect.

---

## 7. Outputs

```
<dump_root>/<batch>/
  Final_report.txt                         per image: "<name> | torch: k (p=..) | imc: k (p=..) | match=…"
  accuracy_report.json                     Torch-vs-GT, IMC-vs-GT, SW–HW agreement, top-k, per-sample
  power_energy_latency_report.{json,csv}   per-inference energy breakdown (crossbar, DAC, TIA,
                                           subtractor, ADC, mux, tile lines, NoC), latency, TOPS/W
  peripheral_ops.json                      operation manifest (normalised to one inference)
  interconnect_report.json                 [INTERCONNECT]: per-layer loading (boundary voltages,
                                           residual, ideal vs loaded deck energy, line energy, Elmore
                                           constants), NoC (mesh, traffic, energy, latency, BookSim check)
  measured_converter_energy.json           H converters only
  crossbar_power/block_XX/<tag>_{pwr,eng}.csv   transient: per-layer crossbar power / energy
  <image stem>/                            logits_{torch,imc}.csv, softmax_{torch,imc}.csv, per-block dumps
  batch/                                   per-block dumps with --batch_all
  timgXX_<name>/                           auto mode with auto_transient_num_images > 1
  _char_dac/, _char_readout/               H-converter characterisation decks and Spectre output
<hwsimdb_root>/<batch>/<layer>/Tile<T>/PE<P>/Array_<r>_<c>.npy        weight tiles
                                         …/Array_<r>_<c>_{pos,neg}/psf/   raw Spectre output + spectre.out
                                         <layer>/hybrid_report.csv        hybrid mode
<tmp_root>/<batch>/                        per-job scratch decks
```

The energy report is per inference. Peripheral counts are normalised to one
image. With `--batch_all`, the transient crossbar energy is divided by the
number of images.

---

## 8. Switching datasets: CIFAR-10 → Fashion-MNIST

Fashion-MNIST is 28×28 grayscale with 10 classes. CIFAR-10 in
`config_cifar10.ini` is 32×32 RGB with 10 classes. The hardware stays the same;
only the workload changes.

| key | CIFAR-10 (ref) | Fashion-MNIST | why |
|---|---|---|---|
| `[DATA] dataset` | `cifar10` | `fashion_mnist` | selects the torchvision loader in the trainer |
| `[DATA] img_height` / `img_width` | `32` | `28` | native size |
| `[DATA] channels` | `auto` → 3 | `auto` → 1 | grayscale; inference takes it from the checkpoint |
| `[MODEL] patch_size` | `4` | `4` or `7` | must divide 28 |
| `[MODEL] num_classes` | `10` | `10` | unchanged |
| `[SAVE] out_dir` | `./cifar10/…` | e.g. `./fmnist/outputs` | keep the checkpoints apart |
| `[PATHS] images_dir` | `./Test_Image/t1` | your FMNIST test images | |
| checkpoint | CIFAR weights | **retrain** | patch embedding and input channels differ |

Copy the reference config and edit the workload sections:

```ini
; config_fmnist.ini  (copy of config_cifar10.ini)
[DATA]
dataset    = fashion_mnist
img_height = 28
img_width  = 28
channels   = auto

[MODEL]
patch_size  = 7          ; 28 / 7 = 4x4 patch grid
num_classes = 10

[SAVE]
out_dir = ./fmnist/outputs

[PATHS]
images_dir = ./fmnist_test

[ACCURACY]
labels_file =            ; optional: CSV "filename,label" or JSON map
```

Then run:

```bash
python main.py --config config_fmnist.ini                 # -> ./fmnist/outputs/best.pt, last.pt
python -m spicecimx.cli.infer --config config_fmnist.ini  # finds best.pt in [SAVE] out_dir
```

The cell and array tables do not need rebuilding, because their hash covers
only hardware and grid inputs. If you change `[HW] subarray_rows` or any
hashed hardware key at the same time, the array table rebuilds on its own.

Checklist when a new dataset misbehaves:

* `patch_size` must divide both image dimensions.
* `[MODEL]` must match the checkpoint. A mismatch fails at load time with the
  missing and unexpected keys.
* `num_classes` must match the classifier head. `--output_classes K` only
  trims the logits; it does not resize the head.
* If `Torch vs GT` is missing from the accuracy report, no labels were
  resolved. Check `labels_file` or the `label_from_filename` regex.

---

## 9. Adding a new Task (accuracy metric set)

The accuracy report is built on a pluggable `Task` in
`spicecimx/core/metrics.py`. Only `classification` is registered. A `Task`
implements `predict(output)` and `is_correct(output, gt)`. It can optionally
implement `agree(a, b)`, `sample_extra(output, gt)` and `has_gt(gt)`. A
language-modelling sketch, `LanguageModelingTask`, ships in `metrics.py` but
is not registered.

```python
# run_with_task.py: register before infer runs
import numpy as np
from spicecimx.core.metrics import Task, register_task
from spicecimx.cli import infer

class MyTask(Task):
    name = "my_task"
    def predict(self, output):
        return np.argmax(np.asarray(output), axis=-1)
    def is_correct(self, output, gt):
        return np.asarray(self.predict(output) == np.asarray(gt)).ravel()

register_task("my_task", MyTask)
infer.main(["--config", "config_cifar10.ini"])
```

Then set `[ACCURACY] task = my_task`. The same three-way comparison
(Torch-vs-GT, IMC-vs-GT, SW–HW agreement) is computed from whatever the task
defines as a prediction. An unregistered task name disables the report. The
run continues, and the error is printed.

---

## 10. Debugging recipes

* **Start small.** `python -m spicecimx.cli.run --selftest` checks the IMC
  kernel without Spectre. `--limit 1` runs one image. `--mode fastdc`
  separates model and config problems from Spectre problems.
* **A Spectre job failed.** With `on_job_failure = raise` (the default), the
  run stops and lists the failing job ids. Look at the job's
  `…/Array_<r>_<c>_{pos,neg}/psf/spectre.out` under `<hwsimdb_root>/<batch>/`.
  Common causes are a wrong `transistor_model` or `process_corner`, a
  `cell_prefix`/netlist name mismatch, or convergence problems. To keep a long
  run going, set `on_job_failure = nan`: the failed subarray's outputs become
  NaN and the run continues. Set `spectre_timeout_s` to stop hung jobs.
* **Characterisation fails or times out.** It aborts instead of caching bad
  tables. Raise `array_char_timeout_s` / `cell_char_timeout_s` for big grids,
  and check the Spectre output under `<tmp_root>/_array_char` or the
  `--work_dir`.
* **`CSV has N V_BL columns but the configured grid … implies M`.** A `.csv`
  table has no V axis, so `fastdc_V_min/V_max/V_step` must describe its
  columns exactly, e.g. 0..1.0 V in 0.01 V steps for 101 columns. A grid
  with the right count but the wrong range is not detected and shifts the V
  axis. An `.npz` built by `characterize` avoids this.
* **Accuracy changes with image order, or a later image looks clipped.** The
  TIA/ADC range is frozen after `tia_calib_images` calls. If the first images
  do not represent the batch, raise the value. `0` never freezes.
* **IMC output saturates or is far off in every mode.** Check that
  `[HW] vmin/vmax` (drive range), `vmin_tia/vmax_tia` and `vmax_adc` (ADC full
  scale) are consistent with your node. Also check the array table: the
  `[v-lin]` line printed at start-up reports its super-linearity.
* **Transient settled values are noisy.** Keep the settling window inside the
  pulse plateau; the config loader warns when it leaves it. Lengthen
  `pulse_width` if the column has not settled. Leave `va_enable_switching = 0`
  for inference. With `1`, the cells drift during long reads.
* **Accuracy looks wrong after a hardware change.** The array table should
  print `cache STALE … rebuilding`. Force a rebuild with
  `characterize_array --force`.
* **Scheduler runs serially.** Check `license_count` and the `--licenses`
  override.
* **Disk fills up.** Keep `clean_workspace = true`, and set
  `hwsimdb_retention = last_block` or `none`.

---

## 11. Tests

```bash
cd SPICECIMX_V3
pytest
```

The suite is portable. It needs neither Spectre nor torch: `tests/fake_spectre.py`
stands in for the backend tests, and the two torch-dependent tests are
skipped without torch. Files:

* `test_imc_matmul_roundtrip.py`: affine map/de-map recovers `A @ B`, and
  the polarity cache catches sign flips.
* `test_v2_vs_v3_demap.py`: with differential readout the d-terms cancel,
  so v2's and v3's de-maps are both exact. The single-ended 3-term expansion
  must not be applied to differential currents.
* `test_scheduler.py`: `JobScheduler` honours `license_count`.
* `test_transient_backend_e2e.py`: full transient backend against the fake
  Spectre.
* `test_dc_backend_units.py`: DC source/testbench rendering and PSF
  reading.
* `test_fastdc.py`: cell table I/O and fastdc evaluation.
* `test_hybrid_sampling.py`: hybrid subarray sampling.
* `test_interconnect.py`: shared-line ladder algebra against nodal analysis,
  the coupled layer solve against a brute-force MNA, floorplan and NoC
  bookkeeping, and the loaded-RSN transient backend against a deck-reading
  fake Spectre.
* `test_regressions_v1_1.py`: regression tests for the v1.1 fixes, including
  calibration keyed by `block_XX/tag`, `freeze_after`, `vmax_adc` scaling and
  workspace cleaning.

---

## 12. Legacy v2 code

The original v2 inference code in `utils/` (`reram_imc_infer.py`, `utils.py`,
`dp.py`, `dc.py`) is kept for comparison only. It is not used by
`spicecimx`. The trainer modules in `utils/` (`config.py`, `data.py`,
`train.py`, `vit_reram.py`) and `main.py` are still used.
