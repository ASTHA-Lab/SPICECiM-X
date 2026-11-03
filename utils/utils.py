import pandas as pd
import os, math, re, json
import shlex
import subprocess


synaptic_array_size= (16,16)
golden_input_path = "./supports/_golden_input.scs"
#base_dir = weights_distribution
stdcell_file_path = "./supports/standard_cell.netlist"
device_model_path = "./supports/reram_cell.v"
transistor_model = "/path/to/gpdk180_v3.3/models/spectre/gpdk.scs"  # set to your PDK

Tox = 5.0e-9
Tref = 4.7249e-9
Ik1 = 6.14e-5
Vref = 0.43
Tmax = 4.9e-9
Tmin = 3.3e-9
res_val = 0
cap_val = 0


start_time_tick = 0
increment = 3e-9

def conductance_calc(alpha, Tox, Tref, Tmax, Tmin):
    Gmax = alpha * np.exp(-((Tox - Tmax) / (Tox - Tref)))
    Gmin = alpha * np.exp(-((Tox - Tmin) / (Tox - Tref)))
    return Gmax, Gmin

def transpose_csv(file_path: str):
    """
    Reads a CSV (2D matrix), transposes it, and saves it as <original>_transposed.csv.

    Example:
        h0_Q_V.csv → h0_Q_V_transposed.csv
    """
    # Read CSV into DataFrame (auto-detect headers or not)
    df = pd.read_csv(file_path, header=None)

    # Transpose the matrix
    df_t = df.transpose()

    # Construct new filename
    base, ext = os.path.splitext(file_path)
    new_path = f"{base}_transposed{ext}"

    # Save transposed CSV (no index, no header)
    df_t.to_csv(new_path, index=False, header=False)

    #print(f"✅ Transposed file saved as: {new_path}")
    return new_path

def generate_pwl_sources(csv_path, outpath, pulse_width, trise, tfall, scol):
    # Load CSV file
    df = pd.read_csv(csv_path, index_col=0)  # Assuming the first column is image names and used as index

    # Initialize the content to be written to the file
    content = ""
    node_counter = 0  # Initialize counter for node numbering

    # Process each column (each voltage source)
    for col in df.columns:
        time_start = 0
        voltage_start = 0
        waveform_data = []

        # Append each point in the waveform
        for voltage in df[col]:
            waveform_data.extend([
                (time_start, voltage_start),
                (time_start + pulse_width, voltage_start),
                (time_start + pulse_width + trise, voltage),
                (time_start + 2 * pulse_width, voltage),
                (time_start + 2 * pulse_width + tfall, voltage_start)
            ])
            time_start += 3 * pulse_width + trise + tfall

        # Generate the source string using modulus operation for node counter
        waveform_str = " ".join(f"{time} {voltage}" for time, voltage in waveform_data)
        content += f"VI{col} (IN{node_counter % (scol)} 0) vsource wave=[{waveform_str}] pwl_period={time_start} type=pwl\n"
        node_counter += 1  # Increment the node counter for the next source

    # Write to file
    #print(f"PWL signals are written to {outpath}")
    with open(outpath, 'w') as file:
        file.write(content)
    total_rows = df.shape[0]
    return total_rows, time_start


def generate_sl_decoder_netlist(num_address_pins, standard_cells):
    num_io = 2 ** num_address_pins
    Sub_ckt_name=f"BL_SL_DECODER_{num_address_pins}x{num_io}"
    Pins=" ".join([f"A{i}" for i in range(num_address_pins)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN"
    # Start of the netlist
    netlist = f"// Library name: Custom_Library\n// Cell name: BL_SL_DECODER_{num_address_pins}x{num_io}\n// View name: schematic\nsubckt BL_SL_DECODER_{num_address_pins}x{num_io} " + " ".join([f"A{i}" for i in range(num_address_pins)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN\n"

    # Generating Inverted Inputs using INVX1 cell
    inv_pins = standard_cells['INVX1']
    for i in range(num_address_pins):
        netlist += f"I_inv{i} ({' '.join(['A' + str(i) if pin == 'A' else 'X' + str(i) if pin == 'Y' else pin for pin in inv_pins])}) INVX1\n"

    # Helper function to generate AND gates
    def generate_and_gates(input_list, output_net, netlist, standard_cells):
        if len(input_list) == 1:
            return input_list[0], netlist, output_net
        else:
            new_output_net = f"net{output_net}"
            and_pins = standard_cells['AND2_X1']
            formatted_pins = ' '.join([input_list[0] if pin == and_pins[0] else input_list[1] if pin == and_pins[1] else new_output_net if pin == 'Y' else pin for pin in and_pins])
            netlist += f"I_and{output_net} ({formatted_pins}) AND2_X1\n"
            output_net += 1
            return generate_and_gates([new_output_net] + input_list[2:], output_net, netlist, standard_cells)

    
      # Generating AND Gates Logic for all combinations
    and_output_nets = []
    output_net_counter = 0
    for i in range(2**num_address_pins):
        # Reverse the binary string to switch MSB and LSB
        binary_str = format(i, f"0{num_address_pins}b")[::-1]
        and_inputs = [f"{'X' if binary_str[j] == '0' else 'A'}{j}" for j in range(num_address_pins)]
        
        final_output, netlist, output_net_counter = generate_and_gates(and_inputs, output_net_counter, netlist, standard_cells)
        and_output_nets.append(final_output)
    
    # Connecting outputs of AND gates to BL_DRIVER block
    bl_driver_pins = standard_cells['BL_DRIVER']
    for i, and_output in enumerate(and_output_nets):
        formatted_pins = ' '.join(['Y' + str(i) if pin == 'OUT' else and_output if pin == 'RDEN' else 'PGMEN' if pin == 'PGMEN' else 'VDDP' if pin == 'VD_PGM' else f'IO{i}' if pin == 'VD_READ' else pin for pin in bl_driver_pins])
        netlist += f"I_bl_driver{i} ({formatted_pins}) BL_DRIVER\n"

    # End of the netlist
    netlist += f"ends BLDECODER_{num_address_pins}x{num_io}\n// End of subcircuit definition.\n"

    return netlist, Sub_ckt_name, Pins


# Example usage
#standard_cells = parse_standard_cells('supports/standard_cell.netlist')
#decoder_netlist = generate_bl_decoder_netlist(4, standard_cells)
#print(decoder_netlist)


def generate_bl_wl_decoder_netlist(num_inputs, standard_cells):
    num_io = 2 ** num_inputs
    Sub_ckt_name=f"BL_WL_DECODER_{num_inputs}x{num_io}"
    Pins = (f"AE " + " ".join([f"A{i}" for i in range(num_inputs)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN " + " ".join([f"T{i}" for i in range(num_io)]))
    # Start of the netlist
    netlist = f"// Library name: Custom_Library\n// Cell name: BL_WL_DECODER_{num_inputs}x{num_io}\n// View name: schematic\nsubckt BL_WL_DECODER_{num_inputs}x{num_io} AE " + " ".join([f"A{i}" for i in range(num_inputs)]) + " VDD VSS " + " ".join([f"Y{i}" for i in range(num_io)]) + " " + " ".join([f"IO{i}" for i in range(num_io)]) + " VDDP PGMEN " + " ".join([f"T{i}" for i in range(num_io)]) + "\n"

    # Generating Inverted Inputs using INVX1 cell
    inv_pins = standard_cells['INVX1']
    for i in range(num_inputs):
        netlist += f"I_inv{i} ({' '.join(['A' + str(i) if pin == 'A' else 'X' + str(i) if pin == 'Y' else pin for pin in inv_pins])}) INVX1\n"

    # Helper function to generate AND gates
    def generate_and_gates(input_list, output_net, netlist, standard_cells):
        if len(input_list) == 1:
            return input_list[0], netlist, output_net
        else:
            new_output_net = f"net{output_net}"
            and_pins = standard_cells['AND2_X1']
            formatted_pins = ' '.join([input_list[0] if pin == and_pins[0] else input_list[1] if pin == and_pins[1] else new_output_net if pin == 'Y' else pin for pin in and_pins])
            netlist += f"I_and{output_net} ({formatted_pins}) AND2_X1\n"
            output_net += 1
            return generate_and_gates([new_output_net] + input_list[2:], output_net, netlist, standard_cells)

    # Generating AND Gates Logic for all combinations
    and_output_nets = []
    output_net_counter = 0
    for i in range(num_io):
        binary_str = format(i, f"0{num_inputs}b")[::-1]
        and_inputs = [f"{'X' if binary_str[j] == '0' else 'A'}{j}" for j in range(num_inputs)]
        
        final_output, netlist, output_net_counter = generate_and_gates(and_inputs, output_net_counter, netlist, standard_cells)
        and_output_nets.append(final_output)

    # Connecting outputs of AND gates to BL_DRIVER block and OR gates
    bl_driver_pins = standard_cells['BL_DRIVER']
    or_pins = standard_cells['OR2_X1']
    for i, and_output in enumerate(and_output_nets):
        formatted_pins_bl_driver = ' '.join(['Y' + str(i) if pin == 'OUT' else and_output if pin == 'RDEN' else 'PGMEN' if pin == 'PGMEN' else 'VDDP' if pin == 'VD_PGM' else f'IO{i}' if pin == 'VD_READ' else pin for pin in bl_driver_pins])
        netlist += f"I_bl_driver{i} ({formatted_pins_bl_driver}) BL_DRIVER\n"
        formatted_pins_or = ' '.join([and_output if pin == or_pins[0] else 'AE' if pin == or_pins[1] else 'T' + str(i) if pin == 'Y' else pin for pin in or_pins])
        netlist += f"I_or{i} ({formatted_pins_or}) OR2_X1\n"

    # End of the netlist
    netlist += f"ends BL_WL_DECODER_{num_inputs}x{num_io}\n// End of subcircuit definition.\n"

    return netlist, Sub_ckt_name, Pins

def generate_crossbar_netlist(size_bwl, size_sl):
    subcircuit_name = f"rram_array_{size_bwl}x{size_sl}"
    bl_pins = [f"BL{i}" for i in range(size_bwl)]
    sl_pins = [f"SL{i}" for i in range(size_sl)]
    wl_pins = [f"WL{i}" for i in range(size_bwl)]

    netlist = f"// Library name: Custom Library\n"
    netlist += f"// Cell name: {subcircuit_name}\n"
    netlist += f"// View name: schematic\n"
    netlist += f"subckt {subcircuit_name} {' '.join(bl_pins + sl_pins + wl_pins)}\n"

    for i in range(size_sl):
        for j in range(size_bwl):
            instance_name = f"I{i * size_bwl + j}"
            bl_pin = bl_pins[j]
            sl_pin = sl_pins[i]
            wl_pin = wl_pins[j]
            # Modify the cell_subckt_name to include row and column indices
            cell_subckt_name = f"rram_cell_1T1R_180N_STD_R{j}C{i}"
            netlist += f"    {instance_name} ({bl_pin} {sl_pin} {wl_pin}) {cell_subckt_name}\n"

    netlist += f"ends {subcircuit_name}\n"
    netlist += "// End of subcircuit definition."

    return netlist, subcircuit_name, wl_pins, bl_pins, sl_pins


def parse_standard_cells(file_path):
    # Dictionary to store the pin sequences of standard cells
    standard_cells = {}
    current_cell = None
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('subckt'):
                parts = line.split()
                current_cell = parts[1]
                standard_cells[current_cell] = parts[2:]  # Store the pin sequence
            elif line.startswith('ends') and current_cell:
                current_cell = None
    #print(standard_cells)
    return standard_cells

def generate_subarray(input_length, num_tensor_out, stdcell_file_path):
    #column size
    bl_wl_row_size = input_length  # Or any other size (Size of input shape) # Need to change to automatic sizing of bl_wl_row
    sl_col_size =num_tensor_out # Row size (Size of Output shape)
    subarray_capacity = bl_wl_row_size * sl_col_size
   
    # Calculate the number of inputs for the decoders
    bl_wl_input_size = int(math.ceil(math.log2(bl_wl_row_size)))
    sl_input_size = int(math.ceil(math.log2(sl_col_size)))
    
    # Read and copy the standard cell netlist
    with open(stdcell_file_path, 'r') as file:
        standard_cell_netlist = file.read()

    # Generate the crossbar array netlist
    crossbar_netlist, xSC_name, xWL_pins, xBL_pins, xSL_pins = generate_crossbar_netlist(bl_wl_row_size, sl_col_size)
    
    #print(xSC_name, xWL_pins, xBL_pins, xSL_pins)

    # Generate the decoders netlist
    standard_cells = parse_standard_cells(stdcell_file_path)
    
    
    #wl_decoder_netlist = generate_decoder_netlist(sl_input_size, standard_cells)
    bl_decoder_netlist, bwdecSC_name, bwdec_pins = generate_bl_wl_decoder_netlist(bl_wl_input_size, standard_cells)
    #print(bwdecSC_name, bwdec_pins)
    sl_decoder_netlist, bsdecSC_name, bsdec_pins = generate_sl_decoder_netlist(sl_input_size, standard_cells)
    #print(bsdecSC_name, bsdec_pins)

    # Combine all netlists
    subarray_netlist = standard_cell_netlist + "\n\n" + crossbar_netlist + "\n\n"
    subarray_netlist += bl_decoder_netlist + "\n\n" + sl_decoder_netlist + "\n\n"
    
    
    #extra_pins_count = (sl_input_size**2) - sl_col_size
    VSS_map = sl_col_size

     # Top level subcircuit calls
    subarray_netlist += f"I1 ({' '.join(xBL_pins)} {' '.join(['net' + str(i*2 + 1) + 'i' for i in range(len(xSL_pins))])} {' '.join(xWL_pins)}) {xSC_name}\n"
    subarray_netlist += f"I2 (AE {' '.join(['A' + str(i) for i in range(bl_wl_input_size)])} VDD VSS {' '.join(['BL' + str(i) for i in range(2**bl_wl_input_size)])} {' '.join(['IN' + str(i) for i in range(2**bl_wl_input_size)])} VDDP PGMEN {' '.join(['WL' + str(i) for i in range(2**bl_wl_input_size)])}) {bwdecSC_name}\n"
    subarray_netlist += f"I3 ({' '.join(['SA' + str(i) for i in range(sl_input_size)])} VDD VSS {' '.join(['net' + str(i*2) + 'i' for i in range(sl_col_size)])} {' '.join(['VSS' for _ in range(VSS_map)])} VDDP_SL SLP) {bsdecSC_name}\n"
    
    # Adding iprobes
    for i in range(sl_col_size):
        subarray_netlist += f"IPRB{i} (net{i*2 + 1}i net{i*2}i) iprobe\n"
    
    return subarray_netlist


import os
import numpy as np
from typing import Tuple, Dict

def distribute_weights(
    layer_name: str,
    layer_weights: np.ndarray,
    synaptic_array_size: Tuple[int, int],
    pe_size: Tuple[int, int],
    tile_size: Tuple[int, int],
    base_dir: str,
) -> Dict[str, int]:
    """
    Split a 2D weight matrix into rs×cs synaptic arrays, pad edges with zeros,
    and save under:
      <base_dir>/<layer_name>/Tile<tile_idx>/PE<pe_in_tile_idx>/Array_<sa_row>_<sa_col>.npy

    Args:
        layer_name: Folder name for this layer.
        layer_weights: 2D NumPy array (m×n).
        synaptic_array_size: (rs, cs) synaptic array dimensions.
        pe_size: (rpe, cpe) arrays-per-PE grid; arrays_per_pe = rpe*cpe.
        tile_size: (rt, ct) PEs-per-tile grid; pes_per_tile = rt*ct.
        base_dir: Root output directory.

    Returns:
        dict summary with counts for convenience.
    """
    if layer_weights.ndim != 2:
        raise ValueError("layer_weights must be a 2D NumPy array.")

    rs, cs = map(int, synaptic_array_size)
    rpe, cpe = map(int, pe_size)
    rt, ct = map(int, tile_size)

    if rs <= 0 or cs <= 0 or rpe <= 0 or cpe <= 0 or rt <= 0 or ct <= 0:
        raise ValueError("All size parameters must be positive integers.")

    m, n = layer_weights.shape

    # Number of synaptic arrays needed in each dimension
    num_arrays_vertically   = (m + rs - 1) // rs
    num_arrays_horizontally = (n + cs - 1) // cs
    total_arrays = num_arrays_vertically * num_arrays_horizontally

    arrays_per_pe = rpe * cpe
    pes_per_tile  = rt * ct

    total_pes   = (total_arrays + arrays_per_pe - 1) // arrays_per_pe
    total_tiles = (total_pes   + pes_per_tile  - 1) // pes_per_tile

    # Ensure base path exists
    layer_root = os.path.join(base_dir, layer_name)
    os.makedirs(layer_root, exist_ok=True)

    # Iterate Tiles → PEs → arrays (by global array index)
    array_written = 0
    for tile_idx in range(total_tiles):
        # PEs in this tile (last tile might be partial)
        pe_start = tile_idx * pes_per_tile
        pe_end   = min(pe_start + pes_per_tile, total_pes)

        for pe_in_tile_idx, pe_global_idx in enumerate(range(pe_start, pe_end)):
            start_array_idx = pe_global_idx * arrays_per_pe
            end_array_idx   = min(start_array_idx + arrays_per_pe, total_arrays)

            # Directory for this PE
            pe_dir = os.path.join(layer_root, f"Tile{tile_idx}", f"PE{pe_in_tile_idx}")
            os.makedirs(pe_dir, exist_ok=True)

            for array_idx in range(start_array_idx, end_array_idx):
                sa_row = array_idx // num_arrays_horizontally
                sa_col = array_idx %  num_arrays_horizontally

                # Slice bounds in the original weight matrix
                r0 = sa_row * rs
                c0 = sa_col * cs
                r1 = min(r0 + rs, m)
                c1 = min(c0 + cs, n)

                block = layer_weights[r0:r1, c0:c1]

                # Pad if at right/bottom edges
                if block.shape != (rs, cs):
                    padded = np.zeros((rs, cs), dtype=layer_weights.dtype)
                    padded[:block.shape[0], :block.shape[1]] = block
                    save_block = padded
                else:
                    save_block = block

                np.save(os.path.join(pe_dir, f"Array_{sa_row}_{sa_col}.npy"), save_block)
                array_written += 1

                if array_written >= total_arrays:
                    break

            if array_written >= total_arrays:
                break

        if array_written >= total_arrays:
            break

    return {
        "m": m,
        "n": n,
        "rs": rs,
        "cs": cs,
        "num_arrays_vertically": num_arrays_vertically,
        "num_arrays_horizontally": num_arrays_horizontally,
        "total_arrays": total_arrays,
        "arrays_per_pe": arrays_per_pe,
        "total_pes": total_pes,
        "pes_per_tile": pes_per_tile,
        "total_tiles": total_tiles,
    }


def quantize_weights_to_int(weights, bits):
    """
    Symmetric signed PTQ quantization.

    weights : np.ndarray of floats
    bits    : int, e.g. 8 or 16
    returns : np.ndarray of ints in [-L, L], where L = 2^{bits-1} - 1
    """
    import numpy as np

    levels_pos = 2**(bits - 1) - 1   # e.g. 7, 127, 32767
    max_abs = np.max(np.abs(weights))

    if max_abs < 1e-12:
        # All-zero (or almost) weights -> just return zeros
        return np.zeros_like(weights, dtype=int)

    # Scale to [-levels_pos, levels_pos]
    scaled = weights / max_abs * levels_pos
    q = np.round(scaled)
    q = np.clip(q, -levels_pos, levels_pos)

    return q.astype(int)

def convert_to_conductance(B, DeltaG, Gmin, Tmin, Tox, Tref, alpha):
    T_del = Tox + ((Tox - Tref) * np.log(((DeltaG * np.abs(B)) + Gmin) / alpha))

    T_pos = np.where(B >= 0, T_del, Tmin)
    T_neg = np.where(B < 0, T_del, Tmin)
    
    return T_pos, T_neg


def convert_to_conductance_ptq(quantized_weights, Gmax, Gmin, Tmin, qbit, Tox, Tref, alpha):
    """
    Map signed integer weights to device conductance via uniform steps.

    quantized_weights : np.ndarray of ints in [-L, L], L = 2^{qbit-1}-1
    Gmax, Gmin        : device conductance window
    Tmin              : minimum programming time
    qbit              : quantization bits
    Tox, Tref, alpha  : device parameters (same as continuous case)
    """

    levels_pos = 2**(qbit - 1) - 1     # max |integer| value
    if levels_pos <= 0:
        raise ValueError(f"qbit must be >= 2, got {qbit}")

    # Step in conductance per integer LSB
    delG = (Gmax - Gmin) / levels_pos

    # Device mapping
    T_del = Tox + ((Tox - Tref) * np.log(((delG * np.abs(quantized_weights)) + Gmin) / alpha))

    T_pos = np.where(quantized_weights >= 0, T_del, Tmin)
    T_neg = np.where(quantized_weights < 0, T_del, Tmin)

    return T_pos, T_neg

    
def convert_to_conductance_continuous(weights, Gmax, Gmin, Tmin, Tox, Tref, alpha):
    """
    Continuous (no quant) mapping: w→G→T.
    weights: np.ndarray of floats (can be any real range)
    """
    abs_w = np.abs(weights)
    Wmax  = abs_w.max() if abs_w.max() != 0 else 1.0

    # 1) normalize into [0,1]
    w_norm    = abs_w / Wmax

    # 2) linear map into device conductance window
    G_levels  = Gmin + w_norm * (Gmax - Gmin)

    # 3) thickness from conductance
    T_fil     = Tox + (Tox - Tref) * np.log(G_levels / alpha)

    # 4) branch assignments
    T_pos     = np.where(weights >= 0, T_fil, Tmin)
    T_neg     = np.where(weights  < 0, T_fil, Tmin)

    return T_pos, T_neg

def weight_mapping(T_pos, T_neg, baseline_file_path, res_val, cap_val, tmp_dir="./tmp"):
    """
    Generate device model netlists for positive and negative conductance branches.

    Args:
        T_pos: 2D array of positive-branch programming thicknesses
        T_neg: 2D array of negative-branch programming thicknesses
        baseline_file_path: path to the base ReRAM Verilog-A model
        res_val: series resistance value
        cap_val: parasitic capacitance value
        tmp_dir: batch-scoped temporary directory (default: "./tmp" for
                 backward compatibility; callers should pass their own)
    """
    # Ensure the tmp_dir exists
    os.makedirs(tmp_dir, exist_ok=True)

    # Initialize the content for the device_model_pos.scs file
    device_model_pos_content = f"ahdl_include \"{baseline_file_path}\" \n"
    # Initialize the content for the device_model_neg.scs file
    device_model_neg_content = f"ahdl_include \"{baseline_file_path}\" \n"
    
    # Iterate over the T_pos and T_neg matrices
    for row_index, (row_pos, row_neg) in enumerate(zip(T_pos, T_neg)):
        for col_index, (T_pos_val, T_neg_val) in enumerate(zip(row_pos, row_neg)):
            device_model_content_pos = ""
            device_model_content_neg = ""
            
            # Device name
            device_name = f"rram_cell_1T1R_180N_STD_R{row_index}C{col_index}"
            
            # Add content for T_pos
            device_model_content_pos += f"// Cell name: {device_name}\n"
            device_model_content_pos += f"// View name: schematic\n"
            device_model_content_pos += f"subckt {device_name} BL SL WL\n"
            device_model_content_pos += "    NM0 (net1 WL net2 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n"
            device_model_content_pos += "        m=(1)*(1)\n"
            device_model_content_pos += f"   R0 (net2 SL) resistor r={res_val} \nC1 (SL 0) capacitor c={cap_val/2} \nC0 (net2 0) capacitor c={cap_val/2}\n"
            device_model_content_pos += f"   I0 (BL net1) sky130_fd_pr_reram__reram_cell area_ox=1.024e-13 \\\n"
            device_model_content_pos += f"        Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 \\\n"
            device_model_content_pos += f"        Tfilament_0=3.3e-09 Eact_generation=1.501 Eact_recombination=1.5 \\\n"
            device_model_content_pos += f"        I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 velocity_k1=150 \\\n"
            device_model_content_pos += f"        gamma_k0=16.5 gamma_k1=-1.25 Temperature_0=300 \\\n"
            device_model_content_pos += f"        C_thermal=3.1825e-16 tau_thermal=2.3e-10 t_step=1e-09 initial_state={T_pos_val:.4e}\n"
            device_model_content_pos += f"ends {device_name}\n"
            device_model_content_pos += "// End of subcircuit definition.\n\n"

            device_model_pos_content += device_model_content_pos
            
            # Add content for T_neg
            device_model_content_neg += f"// Cell name: {device_name}\n"
            device_model_content_neg += f"// View name: schematic\n"
            device_model_content_neg += f"subckt {device_name} BL SL WL\n"
            device_model_content_neg += "    NM0 (net1 WL SL 0) nmos1 w=(2u) l=180n as=1.2p ad=1.2p ps=5.2u pd=5.2u \\\n"
            device_model_content_neg += "        m=(1)*(1)\n"
            device_model_content_neg += f"    I0 (BL net1) sky130_fd_pr_reram__reram_cell area_ox=1.024e-13 \\\n"
            device_model_content_neg += f"        Tox=5e-09 Tfilament_max=4.9e-09 Tfilament_min=3.3e-09 \\\n"
            device_model_content_neg += f"        Tfilament_0=3.3e-09 Eact_generation=1.501 Eact_recombination=1.5 \\\n"
            device_model_content_neg += f"        I_k1=6.14e-05 Tfilament_ref=4.7249e-09 V_ref=0.43 velocity_k1=150 \\\n"
            device_model_content_neg += f"        gamma_k0=16.5 gamma_k1=-1.25 Temperature_0=300 \\\n"
            device_model_content_neg += f"        C_thermal=3.1825e-16 tau_thermal=2.3e-10 t_step=1e-09 initial_state={T_neg_val:.4e}\n"
            device_model_content_neg += f"ends {device_name}\n"
            device_model_content_neg += "// End of subcircuit definition.\n\n"

            device_model_neg_content += device_model_content_neg

    # ── BATCH-SAFE: write into the scoped tmp_dir ──
    pos_path = os.path.join(tmp_dir, "device_model_pos.scs")
    neg_path = os.path.join(tmp_dir, "device_model_neg.scs")

    with open(pos_path, 'w') as device_model_pos_file:
        device_model_pos_file.write(device_model_pos_content)
        
    with open(neg_path, 'w') as device_model_neg_file:
        device_model_neg_file.write(device_model_neg_content)


#Run Sim

def setup_simulation_args_spectre(siminput_path, infile_noext, suffix):
    spectre_args = ["spectre -64",
                    siminput_path,
                    "+mt=8",
                    "+escchars",
                    "=log ./" + infile_noext + f"{suffix}/psf/spectre.out",
                    "-format psfascii",
                    "-raw ./" + infile_noext + f"{suffix}/psf",
                    "+lqtimeout 900",
                    "-maxw 5",
                    "-maxn 5",
                    "+logstatus"]
                    
                    
    run_string = " ".join(spectre_args)
    return run_string

def setup_simulation_args_spectrex(siminput_path, infile_noext, suffix):
    spectre_args = ["spectre -64 +preset=mx",
                    siminput_path,
                    "+mt=16",
                    "+escchars",
                    "=log ./" + infile_noext + f"{suffix}/psf/spectre.out",
                    "-format psfascii",
                    "-raw ./" + infile_noext + f"{suffix}/psf",
                    "+lqtimeout 900",
                    "-maxw 5",
                    "-maxn 5",
                    "+logstatus"]
                    
                    
    run_string = " ".join(spectre_args)
    return run_string

def setup_simulation_args_xps(siminput_path, infile_noext, suffix):
    # spectre +xps +cktpreset=dram -64 -f fsdb +spice +mt=N -outdir run1 -outname xps_out top.sp
    spectre_args = ["spectre +xps",
                    "+cktpreset=dram -64", #dram/sram/pcram
                    "+speed=1",
                    "+mt=8",
                    "+escchars",
                    "=log ./" + infile_noext + f"{suffix}/psf/spectre.out",
                    "-format psfascii",
                    "-raw ./" + infile_noext + f"{suffix}/psf",
                    "+lqtimeout 900",
                    "-maxw 5",
                    "-maxn 5",
                    "+logstatus",
                    "-outdir ./" + infile_noext + f"{suffix}",
                    siminput_path]
                    
                    
    run_string = " ".join(spectre_args)
    return run_string

PW = 1e-9
Trise = 1e-12
Tfall = 1e-12
process_corner = "NN"
vdd_val = "1.8"
vss_val = "0"
temp_val = "27"

def map_primary_input(input_path_pos, input_path_neg, full_path, pwl_source, sim_time, synaptic_array_size, golden_input_path, transistor_model, tmp_dir="./tmp"):
    """
    Generate SPICE input decks for positive and negative conductance branches.

    Args:
        input_path_pos: output path for the positive-branch input deck
        input_path_neg: output path for the negative-branch input deck
        full_path: path to the .npy array file (used to extract row/col indices)
        pwl_source: path to the PWL source file
        sim_time: total simulation time
        synaptic_array_size: (rows, cols) of the synaptic array
        golden_input_path: path to the golden input template
        transistor_model: path to the transistor model file
        tmp_dir: batch-scoped temporary directory (default: "./tmp" for
                 backward compatibility; callers should pass their own)
    """
    # Extract row and column from the filename
    filename = os.path.basename(full_path)
    match = re.search(r"Array_(\d+)_(\d+).npy", filename)
    if match:
        row_number = int(match.group(1))

    # Calculate the start and end lines for source fetching
    start_line = row_number * synaptic_array_size[0]
    end_line = start_line + synaptic_array_size[0]

    # Read the necessary source lines from _pwl_sources.scs
    with open(pwl_source, 'r') as file:
        lines = file.readlines()
    selected_sources = "".join(lines[start_line:end_line])
    
    csnames = " ".join([f"IPRB{i}:in" for i in range(synaptic_array_size[1])])

    # ── BATCH-SAFE: use scoped tmp_dir for device model paths ──
    dev_mod_pos_path = os.path.join(tmp_dir, "device_model_pos.scs")
    dev_mod_neg_path = os.path.join(tmp_dir, "device_model_neg.scs")

    # Read the golden input template and replace the placeholder with the selected sources
    with open(golden_input_path, 'r') as file:
        content = file.read()
    content = content.replace('<sim_time>', str(sim_time))
    content = content.replace('<input_source>', selected_sources)
    content = content.replace('<dev_mod>', dev_mod_pos_path)
    content = content.replace('<transistor_mod>', transistor_model)
    content = content.replace('<pcorner>', process_corner)
    content = content.replace('<vdd>', vdd_val)
    content = content.replace('<vss>', vss_val)
    content = content.replace('<temp>', temp_val)
    content = content.replace('<csnames>', csnames)
    
    # Write the modified content to the new input.scs file
    with open(input_path_pos, 'w') as file:
        file.write(content)
        
    content = content.replace(dev_mod_pos_path, dev_mod_neg_path)
    with open(input_path_neg, 'w') as file:
        file.write(content)


def run_sim(run_string, infile_noext):
    run_command = shlex.split(run_string)
    #print(f"Running simulation for {infile_noext.split('/')[-1]}")
    #Simulation Run Call
    subprocess.call(run_command, stdout=subprocess.DEVNULL)  # Running the simulation
    #print("Done.")

    output_file = "./" + infile_noext + "/psf/tran.tran.tran"
    header = True
    all_data = []

    # Prepare a dictionary to store IPRB values indexed by column number
    iprb_values = {f'IPRB{col}:in': [] for col in range(synaptic_array_size[1])}



def tia_adc_roundtrip(I_diff: np.ndarray,
                      Vmin: float, Vmax: float,
                      Nbits: int,
                      Imin: float=None, Imax: float=None,
                      Vfs: float=None,
                      add_qerror: bool=False,
                      rng: np.random.Generator=None):
    """
    I_diff: (m, n) matrix of differential currents (after pos-neg and row-sum).
    Returns: (V_quant, I_hat) where V_quant is the quantized voltage
             and I_hat is the current reconstructed from quantized voltage.
    """
    m, n = I_diff.shape
    if Imin is None: Imin = float(I_diff.min())
    if Imax is None: Imax = float(I_diff.max())
    if Vfs  is None: Vfs  = Vmax  # common choice; can be <= Vmax if you want headroom

    # Guard against degenerate ranges
    eps = 1e-12
    Ispan = max(Imax - Imin, eps)
    Vspan = max(Vmax - Vmin, eps)

    # TIA: I -> V (affine)
    k = Vspan / Ispan
    V_tia = Vmin + k * (I_diff - Imin)

    # Clip to ADC input range [0, Vfs] (or [Vmin, Vmax] if you prefer)
    V_adc_in = np.clip(V_tia, 0.0, Vfs)

    # Ideal ADC quantization
    L = (1 << Nbits) - 1
    D = np.rint((V_adc_in / Vfs) * L).astype(np.int32)
    D = np.clip(D, 0, L)

    V_quant = (D.astype(np.float64) / L) * Vfs

    if add_qerror:
        if rng is None:
            rng = np.random.default_rng()
        lsb = Vfs / L
        V_quant = V_quant + rng.uniform(-0.5 * lsb, 0.5 * lsb, size=V_quant.shape)

    # Invert TIA to get current from quantized voltage
    I_hat = Imin + (V_quant - Vmin) / k

    return V_quant, I_hat, (k, Imin, Imax, Vfs)



def tia_adc_realistic(
    I_diff: np.ndarray,
    Vmin: float, Vmax: float,
    Nbits: int,
    Imin: float=None, Imax: float=None,
    Vfs: float=None,
    add_qerror: bool=False,
    rng: np.random.Generator=None,
    *,
    # --- Calibration & headroom ---
    use_batch_calib: bool = True,            # if False, expect fixed Imin/Imax
    headroom: float = 1.0,                   # e.g., 0.9 -> compress FS by 10%
    clip_mode: str = "fs",                   # "fs" -> clip to [0, Vfs], "range" -> [Vmin, Vmax]
    # --- Quantization behavior ---
    stochastic_rounding: bool = False,       # stochastic vs .rint()
    # --- Noise / dither ---
    sigma_v: float = 0.0,                    # Gaussian voltage noise (pre-ADC), volts rms
    # add_qerror (uniform ±0.5 LSB) already supported via flag
    # --- TIA mismatch (per column) ---
    k_sigma_rel: float = 0.0,                # relative stddev of TIA gain per column (e.g., 0.02 = 2%)
    o_sigma_abs: float = 0.0,                # absolute stddev of TIA offset per column (volts)
    apply_mismatch_per: str = "col",         # "col", "row", or "none"
    # --- Nonlinearity (soft saturation) ---
    nonlin: str = None,                      # None or "tanh"
    nonlin_gain: float = 3.0,                # sharpness of tanh soft-clip
    # --- Inversion model (what calibration you assume when converting back) ---
    invert_with: str = "nominal",            # "nominal" or "mismatch" (use the true k/offset used above)
    # --- Diagnostics ---
    return_debug: bool = False
):
    """
    I_diff: (m, n) differential currents matrix (after pos-neg & row-sum).
    Returns:
      V_quant, I_hat, meta
    where:
      V_quant: quantized voltages (after ADC/DAC view, plus optional qerror)
      I_hat:   reconstructed currents after inverse TIA using chosen calibration
      meta:    dict with (k_nom, Imin, Imax, Vfs, codes_used, rail_frac, max_abs_I_err, etc.)
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = I_diff.shape

    # --- Calibration selection ---
    if use_batch_calib:
        Imin_eff = float(I_diff.min()) if Imin is None else float(Imin)
        Imax_eff = float(I_diff.max()) if Imax is None else float(Imax)
    else:
        if Imin is None or Imax is None:
            raise ValueError("Fixed calibration requested: please pass Imin and Imax.")
        Imin_eff, Imax_eff = float(Imin), float(Imax)

    # Guard against degenerate ranges
    eps = 1e-12
    Ispan = max(Imax_eff - Imin_eff, eps)
    Vspan = max(Vmax - Vmin, eps)

    # Nominal TIA gain
    k_nom = Vspan / Ispan

    # --- Apply TIA gain/offset mismatch (forward path) ---
    if apply_mismatch_per == "col":
        k_vec = np.ones((1, n)) * k_nom
        if k_sigma_rel > 0.0:
            k_vec = k_vec * (1.0 + rng.normal(0.0, k_sigma_rel, size=(1, n)))
        o_vec = np.zeros((1, n))
        if o_sigma_abs > 0.0:
            o_vec = o_vec + rng.normal(0.0, o_sigma_abs, size=(1, n))
        V_lin = Vmin + k_vec * (I_diff - Imin_eff) + o_vec
        k_used = k_vec
        o_used = o_vec
    elif apply_mismatch_per == "row":
        k_vec = np.ones((m, 1)) * k_nom
        if k_sigma_rel > 0.0:
            k_vec = k_vec * (1.0 + rng.normal(0.0, k_sigma_rel, size=(m, 1)))
        o_vec = np.zeros((m, 1))
        if o_sigma_abs > 0.0:
            o_vec = o_vec + rng.normal(0.0, o_sigma_abs, size=(m, 1))
        V_lin = Vmin + k_vec * (I_diff - Imin_eff) + o_vec
        k_used = k_vec
        o_used = o_vec
    else:
        V_lin = Vmin + k_nom * (I_diff - Imin_eff)
        k_used = k_nom
        o_used = 0.0

    # --- Optional soft nonlinearity ---
    if nonlin is not None and nonlin.lower() == "tanh":
        Vmid = 0.5 * (Vmin + Vmax)
        z = (V_lin - Vmid) / max(Vspan, eps)
        V_tia = Vmid + Vspan * 0.5 * np.tanh(nonlin_gain * z)
        V_tia = np.clip(V_tia, Vmin, Vmax)
    else:
        V_tia = V_lin

    # --- Add pre-ADC analog voltage noise ---
    if sigma_v > 0.0:
        V_tia = V_tia + rng.normal(0.0, sigma_v, size=V_tia.shape)

    # --- Headroom / full-scale selection ---
    if Vfs is None:
        Vfs_eff = (Vmax - 0.0) * headroom if clip_mode == "fs" else (Vmax - Vmin) * headroom
        if clip_mode == "range":
            Vfs_eff = (Vmax - Vmin) * headroom
    else:
        Vfs_eff = float(Vfs) * headroom

    # --- Clip to ADC input window ---
    if clip_mode == "fs":
        V_adc_in = np.clip(V_tia, 0.0, Vfs_eff)
        V_for_codes = V_adc_in
        norm = np.maximum(Vfs_eff, eps)
        x = V_for_codes / norm
    elif clip_mode == "range":
        V_adc_in = np.clip(V_tia, Vmin, Vmax)
        V_for_codes = V_adc_in
        norm = np.maximum(Vspan, eps)
        x = (V_for_codes - Vmin) / norm
    else:
        raise ValueError("clip_mode must be 'fs' or 'range'.")

    # --- Quantization ---
    L = (1 << Nbits) - 1
    x = np.clip(x, 0.0, 1.0)

    if stochastic_rounding:
        D = np.floor(x * L + rng.random(size=x.shape)).astype(np.int32)
    else:
        D = np.rint(x * L).astype(np.int32)
    D = np.clip(D, 0, L)

    # DAC view
    V_quant_unit = D.astype(np.float64) / L
    if clip_mode == "fs":
        V_quant = V_quant_unit * Vfs_eff
    else:
        V_quant = Vmin + V_quant_unit * Vfs_eff

    # Optional uniform ±0.5 LSB "qerror"
    if add_qerror:
        lsb = (Vfs_eff / L)
        V_quant = V_quant + rng.uniform(-0.5 * lsb, 0.5 * lsb, size=V_quant.shape)

    # --- Invert TIA to get current estimate ---
    if invert_with == "mismatch" and (apply_mismatch_per in ("col", "row")):
        if nonlin is None:
            I_hat = Imin_eff + (V_quant - Vmin - o_used) / np.maximum(k_used, eps)
        else:
            I_hat = Imin_eff + (V_quant - Vmin) / max(k_nom, eps)
    else:
        I_hat = Imin_eff + (V_quant - Vmin) / max(k_nom, eps)

    # --- Diagnostics ---
    used_codes = int(np.unique(D).size)
    rail_frac = float(np.mean((D == 0) | (D == L)))
    max_abs_I_err = float(np.max(np.abs(I_hat - (Imin_eff + (V_tia - Vmin)/max(k_nom, eps)))))

    meta = {
        "k_nom": float(k_nom),
        "Imin": float(Imin_eff),
        "Imax": float(Imax_eff),
        "Vfs": float(Vfs_eff),
        "codes_used": used_codes,
        "rail_frac": rail_frac,
        "max_abs_I_err_vs_nominal": max_abs_I_err,
        "Nbits": int(Nbits),
        "clip_mode": clip_mode,
        "headroom": float(headroom),
        "stochastic_rounding": bool(stochastic_rounding),
        "sigma_v": float(sigma_v),
        "k_sigma_rel": float(k_sigma_rel),
        "o_sigma_abs": float(o_sigma_abs),
        "apply_mismatch_per": apply_mismatch_per,
        "nonlin": nonlin,
        "nonlin_gain": float(nonlin_gain),
        "invert_with": invert_with,
    }

    if return_debug:
        meta["codes"] = D

    return V_quant, I_hat, meta


class TIACalibrator:
    def __init__(self, p_lo=0.1, p_hi=99.9):
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)
        self.table = {}  # tag -> {"Imin":..., "Imax":...}

    def update(self, tag: str, I_diff: np.ndarray):
        lo = float(np.percentile(I_diff, self.p_lo))
        hi = float(np.percentile(I_diff, self.p_hi))
        if tag in self.table:
            self.table[tag]["Imin"] = min(self.table[tag]["Imin"], lo)
            self.table[tag]["Imax"] = max(self.table[tag]["Imax"], hi)
        else:
            self.table[tag] = {"Imin": lo, "Imax": hi}

    def get(self, tag: str):
        if tag not in self.table:
            raise KeyError(f"No calibration for tag '{tag}'. Run calibrator.update(tag, I_diff) first or load from file.")
        return self.table[tag]["Imin"], self.table[tag]["Imax"]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.table, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            self.table = json.load(f)