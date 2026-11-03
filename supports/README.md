# Support files

Everything the SPICE decks include, except your PDK.

| file | what it is | used by |
|---|---|---|
| `reram_cell.v` | transient ReRAM Verilog-A (SkyWater SKY130 ReRAM model, Apache-2.0) with an `enable_switching` read-mode parameter | transient, hybrid |
| `sky130_dc_memristor_dc.v` | stateless DC variant of the same model | dc, hybrid, `characterize`, `characterize_array` |
| `standard_cell.netlist` | 1T1R cell `rram_cell_1T1R_180N_STD`, `BL_DRIVER`, decoder gates (`INVX1`, `AND2_X1`, `OR2_X1`, …) for 180 nm devices `nmos1` / `pmos1` | every SPICE run (`[PATHS] stdcell_netlist`) |
| `_golden_input.scs`, `_golden_input_dc.scs` | transient and DC testbench templates (`<...>` fields are filled at run time) | transient / dc |
| `peripherals/peripherals_8b.scs` + `*.va` | behavioural, power-annotated 8-bit DAC, TIA, subtractor and SAR ADC (45 nm / 1.1 V) | H-mode converters (`[SUBCIRCUITS] … , H`) |

## Not included: generate these for your PDK

PDK models, netlists sized for them, and tables characterised from them are
covered by the PDK license and are not distributed. Create them once:

1. **Std-cell netlist for your node.** Copy `standard_cell.netlist` (for
   example to `standard_cell_45nm.netlist`, the name `config_cifar10.ini`
   uses). Replace `nmos1` / `pmos1` and the sizes with your PDK's devices and
   add any instance parameters they need. Keep every subckt name and pin
   list. The file is pasted verbatim into each crossbar deck, so do not put
   `ahdl_include` in it.
2. **Point the config at your PDK:** `[PATHS] transistor_model`,
   `[SIM] process_corner` / `vdd`, `[PDK] transistor_model_name` and the
   `transistor_w/l/…` geometry.
3. **Cell table** (fastdc and the `dc` fallback pre-warp):

   ```bash
   python -m spicecimx.cli.characterize --config config_cifar10.ini
   ```

   writes `[PATHS] char_table_path` (`cell_table_45nm.npz` in the reference
   config).
4. **Array table** (voltage pre-warp in every mode, `R_access = auto`):

   ```bash
   python -m spicecimx.cli.characterize_array --config config_cifar10.ini
   ```

   writes `[PATHS] array_table_path`. It is cached under a hash of the
   hardware configuration, and `cli.infer` rebuilds it automatically when
   that changes.

Both characterisations abort on a failed Spectre run instead of caching an
incomplete table.
