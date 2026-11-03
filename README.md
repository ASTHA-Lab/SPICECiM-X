# SPICECiM-X

SPICE-accurate simulation of Vision Transformer inference on ReRAM (1T1R)
analog in-memory-computing crossbars.

SPICECiM-X takes a trained PyTorch ViT, maps every static-weight matrix
multiplication (Q/K/V, W_O, FFN W1/W2, classifier head) onto differential
1T1R crossbars, simulates them with Cadence Spectre through a **reusable
subarray netlist (RSN)** dispatched in parallel across your Spectre licenses,
and recovers floating-point results with an affine **demapping** and a
**voltage pre-warp** that cancels the 1T1R I–V non-linearity. Softmax,
LayerNorm and the activation–activation products run digitally. It reports
SW–HW agreement / accuracy, energy (crossbar from SPICE, DAC/TIA/subtractor/ADC
from models or from SPICE-characterised hardware, tile lines and the
network-on-chip), latency and TOPS/W.

The full user guide is [`SPICECIMX_V3/TUTORIAL.md`](SPICECIMX_V3/TUTORIAL.md).
The same guide opens from the GUI's **Help** button.

## Requirements

* Python ≥ 3.9 with `numpy` and `pandas`. Training and inference also need
  `torch`, `torchvision` and `Pillow`.
* **Cadence Spectre** (tested with Spectre 24.1) on `PATH` for every mode
  except `--selftest`. The pure-table `fastdc` mode still needs Spectre once,
  to build its tables.
* **Your PDK.** No PDK data ships with this repository. The reference
  configuration was written for Cadence GPDK 45 nm, whose models, std cells
  and tables characterised from them cannot be redistributed. See
  [Set up your PDK](#set-up-your-pdk).
* Tkinter, only for the GUI (`python3-tk` on most Linux distributions).

```bash
git clone https://github.com/ASTHA-Lab/SPICECiM-X.git
cd SPICECiM-X
pip install -e "SPICECIMX_V3[torch,test]"
```

## Set up your PDK

Edit `config_cifar10.ini` (or a copy of it). Three settings are
placeholders that must point at your installation:

| setting | what to put there |
|---|---|
| `[PATHS] transistor_model` | your PDK's Spectre model file, replacing `/path/to/gpdk045_v_6_0/models/spectre/gpdk045.scs`. `[SIM] process_corner` must name one of its sections |
| `[PATHS] stdcell_netlist` | a std-cell netlist for your PDK (1T1R cell, `BL_DRIVER`, decoder gates). Start from `supports/standard_cell.netlist` (180 nm devices `nmos1`/`pmos1`) and keep its subckt names and pins. The reference config expects `supports/standard_cell_45nm.netlist` |
| `[PDK] transistor_model_name` + geometry, `[SIM] vdd` | the access-NMOS name in your model file (e.g. `g45n1svt` for GPDK045), its W/L, and the supply of your node |

The cell table (`char_table_path`) and the array table
(`array_table_path`) are generated for your PDK by the characterisation
step below. [`supports/README.md`](supports/README.md) lists every support
file and where it comes from.

## Quick start

All commands run from the repository root. Every relative path in a config's
`[PATHS]` is resolved against `project_dir`, which is itself relative to the
config file.

```bash
# 0. Sanity check of the IMC kernel (numpy only, no Spectre)
python -m spicecimx.cli.run --selftest

# 1. Train a ViT (or use your own checkpoint): writes best.pt / last.pt to [SAVE] out_dir
python main.py --config config_cifar10.ini

# 2. One-time characterisation for your PDK (cached; rebuilt automatically when the hardware config changes)
python -m spicecimx.cli.characterize       --config config_cifar10.ini   # 1T1R cell table (fastdc)
python -m spicecimx.cli.characterize_array --config config_cifar10.ini   # subarray table (pre-warp)

# 3. SPICE-accurate inference on the bundled CIFAR-10 sample images
python -m spicecimx.cli.infer --config config_cifar10.ini --images_dir Test_Image/t1_ds --limit 100
```

`cli.infer` options: `--mode transient|dc|hybrid|fastdc|auto`, `--licenses N`,
`--checkpoint PATH`, `--batch_all` (time-multiplex all images through each
matmul), `--limit N`, `--output_classes K`, `--gui`.

## Simulation modes (`[SIM] mode`)

| mode | what runs | use it for |
|---|---|---|
| `transient` | full transient Spectre with PWL drive per image | energy, power, latency |
| `dc` | one parametric DC sweep per (subarray, branch) | accuracy with real SPICE currents |
| `fastdc` | SPICE-built cell/array tables, numpy bit-line sums | fast accuracy sweeps |
| `hybrid` | `dc` + transient on a sampled fraction of subarrays (`hybrid_transient_fraction`) | DC-vs-transient validation report |
| `auto` | `transient` on 1..N images (energy) + `fastdc` on all images (accuracy) | full accuracy + energy report in one command |

## Peripherals: software (S) or hardware (H)

`[SUBCIRCUITS]` declares each converter as `name, S` (analytic model) or
`name, H` (a real subckt simulated in Spectre). H converters are characterised
once (DAC code sweep; TIA → subtractor → ADC current sweep) and their measured
transfer and energy are used in every mode. A ready-to-use behavioural,
power-annotated library (45 nm / 1.1 V) is in `supports/peripherals/`:

```ini
[PATHS]
peripheral_netlist = ./supports/peripherals/peripherals_8b.scs
[SUBCIRCUITS]
dac = DAC_8b, H
tia = TIA_8b, H
subtractor = SUBTRACTOR_8b, H
adc = ADC_8b, H
```

Any transistor-level subckt with the same pin names (`INP<7:0>`/`OUT` for the
DAC, `IN`/`OUT` for the TIA, `INP`/`INN`/`OUT` for the subtractor,
`IN`/`CLK`/`OUT<7:0>` for the ADC, plus `VDD`/`VSS`) drops in unchanged. The
`cell` and `bl_driver` rows are informational (the crossbar always uses
`[PDK] cell_prefix` and the std-cell `BL_DRIVER`); the multiplexer is energy-
modelled only.

## Tile interconnect: loading between subarrays and the NoC (`[INTERCONNECT]`)

The RSN simulates every 16×16 subarray with ideal sources and an ideal
ground. In the tile, the row-blocks of a layer share each output column line
into one TIA (the partial sums add by KCL), and the tiles exchange
activations with the host/global buffer over a network-on-chip.
`[INTERCONNECT]` models both:

* **Loaded RSN** (transient): pass 1 is the ideal RSN. A line solve predicts
  each subarray's boundary, and pass 2 (and pass 3 if needed) re-simulates
  every deck with its column ending in the Thévenin equivalent of the shared
  line (line R + TIA input R). The wire capacitance adds C·V² energy and
  Elmore settling. `python -m spicecimx.cli.check_loading` compares the
  result with one monolithic Spectre deck of a whole layer.
* **NoC**: mesh of tiles plus a global buffer. The traffic equals the DAC/ADC
  conversions, with Q/K/V inputs multicast. Router energy comes from
  BookSim2/Orion and link energy from C·V². `noc = booksim` replays the
  traffic in a patched BookSim2 (`SPICECIMX_V3/scripts/booksim/`) as a
  cycle-accurate check.

The method and its validation are described in
[`docs/interconnect_report.md`](docs/interconnect_report.md).

## Outputs

* `<dump_root>/<batch>/Final_report.txt`: per-image Torch vs IMC prediction
* `<dump_root>/<batch>/accuracy_report.json`: Torch-vs-GT, IMC-vs-GT, SW–HW agreement
* `<dump_root>/<batch>/power_energy_latency_report.{json,csv}`: per-inference energy breakdown (incl. tile lines and NoC), latency, TOPS/W
* `<dump_root>/<batch>/interconnect_report.json`: per-layer loading and the NoC
* `<hwsimdb_root>/<batch>/`: weight tiles and raw Spectre output per layer

`<batch>` is the folder name of `images_dir`. With `[SAVE] clean_workspace =
true` a run clears only its own batch folders. Two runs on the same image
folder at the same time collide, so give concurrent runs differently named
image folders.

## Configuration

`config_cifar10.ini` is the annotated reference configuration (45 nm, CIFAR-10
ViT); `SPICECIMX_V3/TUTORIAL.md` walks through every section and how to switch
datasets. The keys most worth knowing:

* `[HW] vmin/vmax`: crossbar drive range. `R_access = auto` fits the series
  resistance from the array table.
* `[PERIPHERALS] vmin_tia/vmax_tia` (TIA window), `vmax_adc` (ADC full-scale
  voltage), `adc_bits`, `tia_calib_images` (per-layer ADC range learned from
  the first N images, then frozen).
* `[SIM] license_count` (Spectre seats to use), `on_job_failure = raise|nan`,
  `spectre_timeout_s`.
* `[PDK] va_enable_switching`: `0` freezes the ReRAM filament during reads
  (default), `1` runs the full SET/RESET dynamics.

## Tests

```bash
cd SPICECIMX_V3 && pytest
```

The suite runs without Spectre, a PDK or torch (a fake Spectre stands in for
the backend tests; two tests are skipped without torch).

## Repository layout

```
main.py, utils/          ViT trainer (ViT_ReRAM model, data, config loader);
                         utils/{reram_imc_infer,utils,dp,dc}.py are the legacy v2 flow, kept for comparison
config_cifar10.ini       annotated reference configuration
SPICECIMX_V3/spicecimx/  the simulator package
  core/                  imc_matmul (affine map / demap), ViT forward, metrics
  sim/                   Spectre scheduler, transient/dc/hybrid/fastdc backends
  hw/                    netlist + device emission, characterisation tables, subckt registry
  analog/                TIA/ADC model, H-mode converter chain, energy/latency report
  arch/                  floorplan, shared-line loading, NoC (analytic + BookSim2)
  cli/                   infer, characterize, characterize_array, check_loading, gui, run
SPICECIMX_V3/tests/      pytest suite (fake Spectre, no PDK needed)
SPICECIMX_V3/scripts/    booksim/ (patch + build BookSim2)
supports/                Verilog-A ReRAM models, std-cell netlist template, golden decks
supports/peripherals/    behavioural power-annotated DAC/TIA/subtractor/ADC (H mode)
docs/                    tile-interconnect method report
Test_Image/              sample CIFAR-10 test images
```

## Citation

If you use SPICECiM-X, please cite:

> S. M. M. Ahsan, M. S. Shahriar, T. Hossain, M. S. Hasan, T. Hoque,
> "SPICECiM-X: An Accurate Circuit-Level Design and Optimization Framework for
> Analog In-Memory Transformer Inference."

## License

MIT; see [`LICENSE`](LICENSE). The ReRAM Verilog-A models
`supports/reram_cell.v` and `supports/sky130_dc_memristor_dc.v` are derived
from the SkyWater SKY130 PDK and keep their Apache-2.0 headers.
