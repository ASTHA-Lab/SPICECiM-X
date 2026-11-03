"""
BookSim2 backend for the inter-tile NoC (``[INTERCONNECT] noc = booksim``).

Needs a BookSim2 build with the trace-matrix patch
(``scripts/booksim/build_booksim.sh`` clones, patches and builds it); point
``[INTERCONNECT] booksim_binary`` at the resulting ``src/booksim``.

Every NoC phase of the inference (``arch.noc.build_transfers``: one input and
one output phase per matmul) becomes a traffic matrix between the global
buffer and the layer's edge tiles.  BookSim2 simulates the mesh with that
matrix near the saturation load of the buffer port (the phase is a burst, so
it runs at the highest stable load) and reports accepted throughput, average
packet latency and Orion power.  The phase time is

    flits / accepted_flits_per_cycle + average packet latency

and its dynamic energy is the data-dependent Orion power (buffers, crossbar,
allocators, output DFFs, channel wires; clock and leakage excluded, as for
the other peripherals) integrated over the phase time.  Orion energies come
from BookSim's 32 nm techfile and are C*V^2-scaled to the node.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from spicecimx.analog import tech_scale as _ts

# 2007 ITRS 32 nm techfile shipped with BookSim2 (src/power/techfile.txt).
TECH_REF = {"node_nm": 32.0, "Vdd": 0.9, "R": 606.321,
            "Cd": 0.267e-15, "Cg": 0.534e-15, "Cgdl": 0.1068e-15}


def techfile_clock_hz(t: Dict = TECH_REF) -> float:
    """Clock the Orion power numbers refer to: 1 / (20 FO4)."""
    fo4 = t["R"] * (3.0 * t["Cd"] + 12.0 * t["Cg"] + 12.0 * t["Cgdl"])
    return 1.0 / (20.0 * fo4)


_DYNAMIC = ("Channel Wire Power", "Channel Retiming Power", "Input Read Power",
            "Input Write Power", "Switch Power", "Switch Control Power",
            "Output DFF Power", "Output Control Power")
_ROUTER_DYN = ("Input Read Power", "Input Write Power", "Switch Power",
               "Switch Control Power", "Output DFF Power", "Output Control Power")


def parse_output(text: str) -> Dict:
    """Pull the summary statistics and the power breakdown from stdout."""
    out: Dict = {}
    pats = {
        "packet_latency": r"Packet latency average = ([\d.eE+-]+)",
        "network_latency": r"Network latency average = ([\d.eE+-]+)",
        "accepted_flit_rate": r"Accepted flit rate average = ([\d.eE+-]+)",
        "injected_flit_rate": r"Injected flit rate average = ([\d.eE+-]+)",
        "hops": r"Hops average = ([\d.eE+-]+)",
        "time_taken": r"Time taken is (\d+) cycles",
    }
    for k, p in pats.items():
        mm = re.findall(p, text)
        if mm:
            out[k] = float(mm[-1])
    power = {}
    for line in text.splitlines():
        m = re.match(r"-\s*([A-Za-z ]+Power):\s*([\d.eE+-]+)", line.strip())
        if m:
            power[m.group(1).strip()] = float(m.group(2))
        m = re.match(r"-\s*Completion Time:\s*([\d.eE+-]+)", line.strip())
        if m:
            out["completion_cycles"] = float(m.group(1))
    out["power_W"] = power
    out["unstable"] = ("unstable" in text.lower()) or ("Simulation unstable" in text)
    return out


def write_config(path: str, *, k: int, flit_bits: int, packet_flits: int,
                 load: float, matrix_file: str = "", traffic: str = "matrix",
                 tech_file: str = "", seed: int = 1) -> None:
    lines = [
        "topology = mesh;", f"k = {int(k)};", "n = 2;",
        "routing_function = dor;",
        "num_vcs = 4;", "vc_buf_size = 4;", "wait_for_tail_credit = 1;",
        "vc_allocator = islip;", "sw_allocator = islip;", "alloc_iters = 1;",
        "credit_delay = 2;", "routing_delay = 0;", "vc_alloc_delay = 1;",
        "sw_alloc_delay = 1;", "input_speedup = 1;", "output_speedup = 1;",
        "internal_speedup = 1.0;",
        f"packet_size = {int(packet_flits)};",
        "sim_type = latency;", "injection_rate_uses_flits = 1;",
        f"injection_rate = {load:.6f};",
        "warmup_periods = 3;", "sample_period = 2000;", "max_samples = 6;",
        f"seed = {int(seed)};",
        f"channel_width = {int(flit_bits)};",
    ]
    if traffic == "matrix":
        lines += [f"matrix_file = {matrix_file};", "traffic = matrix;",
                  "injection_process = matrix;"]
    else:
        lines += [f"traffic = {traffic};", "injection_process = bernoulli;"]
    if tech_file:
        lines += ["sim_power = 1;", f"tech_file = {tech_file};",
                  "power_output_file = pwr_tmp;"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _tech_file_for(binary: str) -> str:
    cand = os.path.join(os.path.dirname(os.path.abspath(binary)), "power", "techfile.txt")
    return cand if os.path.exists(cand) else ""


def run(binary: str, cfg_path: str, timeout_s: float = 600.0) -> Dict:
    p = subprocess.run([binary, cfg_path], cwd=os.path.dirname(cfg_path),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout_s)
    res = parse_output(p.stdout.decode(errors="replace"))
    res["rc"] = p.returncode
    return res


def dynamic_energy_J(res: Dict, cycles: float) -> Tuple[float, float]:
    """(total, router-only) data-dependent energy over ``cycles`` techfile
    clocks, from the Orion power breakdown."""
    pw = res.get("power_W", {})
    f = techfile_clock_hz()
    tot = sum(pw.get(k, 0.0) for k in _DYNAMIC) * cycles / f
    rtr = sum(pw.get(k, 0.0) for k in _ROUTER_DYN) * cycles / f
    return tot, rtr


def characterize_router(binary: str, workdir: str, *, k: int = 8,
                        flit_bits: int = 64, packet_flits: int = 4,
                        load: float = 0.1) -> Dict:
    """Router energy per bit (techfile node) from a uniform-traffic run."""
    os.makedirs(workdir, exist_ok=True)
    cfg = os.path.join(workdir, "router_char.cfg")
    write_config(cfg, k=k, flit_bits=flit_bits, packet_flits=packet_flits,
                 load=load, traffic="uniform", tech_file=_tech_file_for(binary))
    res = run(binary, cfg)
    T = res.get("completion_cycles", res.get("time_taken", 0.0))
    nodes = k * k
    flits = res.get("accepted_flit_rate", 0.0) * nodes * T
    hops = res.get("hops", 0.0)
    _, e_rtr = dynamic_energy_J(res, T)
    per_bit = e_rtr / (flits * (hops + 1.0) * flit_bits) if flits > 0 else float("nan")
    return {"router_pJ_per_bit_32nm": per_bit * 1e12, "hops": hops,
            "flits": flits, "cycles": T, "raw": res}


def _node_id(xy: Tuple[int, int], k: int) -> int:
    return int(xy[1]) * k + int(xy[0])


def run_booksim_phases(plan, transfers, ic, workdir: Optional[str] = None,
                       loads=(0.9, 0.7, 0.5, 0.3, 0.15)) -> Dict:
    """Simulate every NoC phase in BookSim2; returns totals + per-phase rows."""
    binary = ic.booksim_binary
    if not binary or not os.path.exists(binary):
        raise FileNotFoundError(
            f"[INTERCONNECT] noc = booksim but booksim_binary '{binary}' does not "
            f"exist; build it with scripts/booksim/build_booksim.sh")
    k = max(plan.width, plan.height)
    bxy = _node_id(plan.buffer_xy, k)
    wd = workdir or tempfile.mkdtemp(prefix="booksim_")
    os.makedirs(wd, exist_ok=True)
    tech = _tech_file_for(binary)
    phases: Dict[Tuple[int, str], List] = {}
    for t in transfers:
        phases.setdefault((t.layer_idx, t.phase), []).append(t)

    def one(key):
        li, ph = key
        ts = phases[key]
        d = os.path.join(wd, f"L{li:03d}_{ph}")
        os.makedirs(d, exist_ok=True)
        mf = os.path.join(d, "matrix.txt")
        n_lines = 0
        with open(mf, "w") as f:
            for t in ts:
                node = _node_id(t.node, k)
                s, dd = (bxy, node) if ph == "in" else (node, bxy)
                if s == dd:            # tile shares the buffer's router
                    continue
                f.write(f"{s} {dd} {t.bits}\n")
                n_lines += 1
        bits = sum(t.bits for t in ts)
        flits = int(math.ceil(bits / float(ic.noc_flit_bits)))
        if n_lines == 0:               # local to the buffer's router: no NoC
            return key, {"tag": ts[0].tag, "phase": ph, "bits": bits,
                         "flits": flits, "load": 0.0,
                         "accepted_flits_per_cycle": float("nan"),
                         "packet_latency": 0.0, "hops": 0.0, "cycles": 0.0,
                         "energy_32nm_J": 0.0}
        best = None
        for load in loads:
            cfg = os.path.join(d, "cfg")
            write_config(cfg, k=k, flit_bits=ic.noc_flit_bits,
                         packet_flits=ic.noc_packet_flits, load=load,
                         matrix_file=mf, tech_file=tech)
            r = run(binary, cfg)
            # Upstream main() returns -1 (rc 255) on SUCCESS; judge the run by
            # its statistics instead of the exit code.
            if not r["unstable"] and r.get("accepted_flit_rate", 0) > 0 \
                    and "packet_latency" in r:
                best = (load, r)
                break
        if best is None:
            return key, {"error": "no stable load"}
        load, r = best
        rate = r["accepted_flit_rate"] * k * k            # flits/cycle, network
        cyc = flits / rate + r.get("packet_latency", 0.0)
        T = r.get("completion_cycles", r.get("time_taken", 0.0))
        # Router energy only: BookSim's channel model assumes its techfile's
        # 2 mm links; the real link energy (tile pitch) is added analytically.
        _, e_rtr = dynamic_energy_J(r, T)
        delivered = rate * T
        e_flit = e_rtr / delivered if delivered > 0 else 0.0
        return key, {"tag": ts[0].tag, "phase": ph, "bits": bits, "flits": flits,
                     "load": load, "accepted_flits_per_cycle": rate,
                     "packet_latency": r.get("packet_latency", 0.0),
                     "hops": r.get("hops", 0.0), "cycles": cyc,
                     "energy_32nm_J": e_flit * flits}

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count() or 1)) as ex:
        rows = dict(ex.map(one, sorted(phases)))
    bad = [k_ for k_, v in rows.items() if "error" in v]
    if bad:
        raise RuntimeError(f"BookSim2 found no stable load for {len(bad)} phase(s): "
                           f"{bad[:5]}")
    scale = (_ts.cap_scale(ic_node(ic), TECH_REF["node_nm"])
             * (ic_vdd(ic) / TECH_REF["Vdd"]) ** 2)
    cycles = sum(v["cycles"] for v in rows.values())
    energy = sum(v["energy_32nm_J"] for v in rows.values()) * scale
    return {"cycles": cycles, "latency_s": cycles / ic.noc_clock_hz,
            "energy_router_J": energy, "booksim_energy_scale": scale,
            "booksim_phases": [dict(v, layer_idx=k_[0]) for k_, v in sorted(rows.items())]}


# The node / supply used for the C*V^2 scaling are attached to the
# InterconnectCfg by arch.noc.estimate_noc before calling the backend.
def ic_node(ic) -> float:
    return float(getattr(ic, "_node_nm", TECH_REF["node_nm"]))


def ic_vdd(ic) -> float:
    return float(getattr(ic, "_vdd", TECH_REF["Vdd"]))
