"""
Tile interconnect: shared-line loading between RSN subarrays + NoC.

The reference for the loading model is a brute-force modified nodal analysis
(MNA) of a whole layer — every subarray, its input-bus taps and column-line
taps, driver and TIA resistances — with linear cells.  For linear cells the
Norton port model is exact, so the boundary relaxation must reproduce the MNA
currents to numerical precision (pass 2), both in the numpy solver and
through the real transient backend driven by a deck-reading fake Spectre.
"""
import os
import sys

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
SUPPORTS = os.path.join(REPO_ROOT, "supports")
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.config import (HwCfg, SimCfg, PathsCfg, PeripheralsCfg,  # noqa: E402
                              InterconnectCfg)
from spicecimx.arch.floorplan import (                                   # noqa: E402
    resolve_interconnect, input_bus_segments, column_line_segments,
    place_layers, layer_grid, wire_r_per_um,
)
from spicecimx.arch.loading import Ladder, LayerLoading                  # noqa: E402
from spicecimx.arch.noc import build_transfers, estimate_noc            # noqa: E402

FAKE_DECK = os.path.join(PKG_ROOT, "tests", "fake_spectre_deck.py")


def _params(rs=4, cs=4, **kw):
    ic = InterconnectCfg(enable=True, loading=True, cell_pitch_x_um=1.0,
                         cell_pitch_y_um=1.0, subarray_gap_um=2.0, tile_gap_um=5.0,
                         wire_r_ohm_per_um=kw.pop("r_um", 20.0),
                         driver_r_out_ohm=kw.pop("r_drv", 150.0),
                         tia_r_in_ohm=kw.pop("r_tia", 80.0), **kw)
    hw = HwCfg(subarray_rows=rs, subarray_cols=cs, pe_rows=1, pe_cols=1,
               tile_rows=2, tile_cols=2)
    return ic, hw, resolve_interconnect(ic, hw, None, PeripheralsCfg(tech_node_nm=45))


# ---------------------------------------------------------------------------
# Ladder algebra
# ---------------------------------------------------------------------------
def test_ladder_matches_nodal_analysis():
    rng = np.random.default_rng(1)
    seg = rng.uniform(1, 20, 5)
    lad = Ladder(seg, r_src=7.0)
    I = rng.uniform(0, 1e-3, 5)
    # node chain: source(0V) -(r_src+seg0)- t0 -seg1- t1 ... ; taps draw I
    T = 5
    Gm = np.zeros((T, T))
    rlist = [7.0 + seg[0]] + list(seg[1:])
    for t in range(T):
        Gm[t, t] += 1 / rlist[t]
        if t > 0:
            Gm[t, t] += 0; Gm[t - 1, t - 1] += 1 / rlist[t]
            Gm[t, t - 1] -= 1 / rlist[t]; Gm[t - 1, t] -= 1 / rlist[t]
    v = np.linalg.solve(Gm, -I)            # tap currents leave the line
    assert np.allclose(lad.deviation(I), -v, rtol=1e-10)
    # power: sum of I^2 R over all resistors
    through = np.cumsum(I[::-1])[::-1]
    p_ref = np.sum(np.array(rlist) * through ** 2)
    assert np.isclose(lad.ir_power(I), p_ref)
    # self-consistent solve with tap conductances
    g = rng.uniform(1e-4, 1e-3, T)
    I0 = rng.uniform(0, 1e-3, T)
    d = lad.solve(I0, g)
    assert np.allclose(d, lad.deviation(I0 - g * d), rtol=1e-10)


# ---------------------------------------------------------------------------
# Coupled layer solve vs full MNA
# ---------------------------------------------------------------------------
def layer_mna(V, G, P, R, C, rs, cs):
    """Exact column currents {b: (m, R, n)} of the loaded layer (linear cells).
    With a buffered input bus every bus tap is held at its row voltage (the
    subarray's own driver); with a shared bus it is a resistive line."""
    if not P.input_shared:
        return _layer_mna_buffered(V, G, P, R, C, rs, cs)
    seg_in = np.array(input_bus_segments(C, P)) * P.r_um
    seg_out = np.array(column_line_segments(R, P)) * P.r_um
    k, n = R * rs, C * cs
    nb = k * 2 * C                         # bus taps
    nc = 2 * n * R                         # column taps
    bus = lambda i, t: i * 2 * C + t
    col = lambda b, j, p: nb + (0 if b == "pos" else n * R) + j * R + p
    out = {b: np.zeros((V.shape[0], R, n)) for b in ("pos", "neg")}
    for w in range(V.shape[0]):
        A = np.zeros((nb + nc, nb + nc)); rhs = np.zeros(nb + nc)

        def add(a, b_, g):
            A[a, a] += g
            if b_ is not None:
                A[b_, b_] += g; A[a, b_] -= g; A[b_, a] -= g

        for i in range(k):                 # input bus of row i
            add(bus(i, 0), None, 1 / (P.r_drv + seg_in[0]))
            rhs[bus(i, 0)] += V[w, i] / (P.r_drv + seg_in[0])
            for t in range(1, 2 * C):
                add(bus(i, t - 1), bus(i, t), 1 / seg_in[t])
        for b in ("pos", "neg"):           # column lines (tap 0 = row-block R-1)
            for j in range(n):
                order = list(range(R - 1, -1, -1))
                add(col(b, j, order[0]), None, 1 / (P.r_tia + seg_out[0]))
                for t in range(1, R):
                    add(col(b, j, order[t - 1]), col(b, j, order[t]), 1 / seg_out[t])
        for b in ("pos", "neg"):           # cells
            for i in range(k):
                p = i // rs
                for j in range(n):
                    q = j // cs
                    add(bus(i, 2 * q + (0 if b == "pos" else 1)), col(b, j, p), G[b][i, j])
        x = np.linalg.solve(A, rhs)
        for b in ("pos", "neg"):
            for i in range(k):
                p = i // rs
                for j in range(n):
                    q = j // cs
                    out[b][w, p, j] += G[b][i, j] * (
                        x[bus(i, 2 * q + (0 if b == "pos" else 1))] - x[col(b, j, p)])
    return out


def _layer_mna_buffered(V, G, P, R, C, rs, cs):
    seg_out = np.array(column_line_segments(R, P)) * P.r_um
    k, n = R * rs, C * cs
    out = {b: np.zeros((V.shape[0], R, n)) for b in ("pos", "neg")}
    order = list(range(R - 1, -1, -1))          # tap 0 = row-block R-1
    for w in range(V.shape[0]):
        for b in ("pos", "neg"):
            for j in range(n):
                A = np.zeros((R, R)); rhs = np.zeros(R)
                for t in range(R):
                    p = order[t]
                    gsum = G[b][p * rs:(p + 1) * rs, j].sum()
                    A[t, t] += gsum
                    rhs[t] += V[w, p * rs:(p + 1) * rs] @ G[b][p * rs:(p + 1) * rs, j]
                A[0, 0] += 1 / (P.r_tia + seg_out[0])
                for t in range(1, R):
                    g = 1 / seg_out[t]
                    A[t, t] += g; A[t - 1, t - 1] += g; A[t, t - 1] -= g; A[t - 1, t] -= g
                x = np.linalg.solve(A, rhs)
                for t in range(R):
                    p = order[t]
                    out[b][w, p, j] = (V[w, p * rs:(p + 1) * rs] - x[t]) @ \
                        G[b][p * rs:(p + 1) * rs, j]
    return out


def _random_layer(rng, R, C, rs, cs, m=3):
    k, n = R * rs, C * cs
    G = {b: rng.uniform(5e-6, 1e-4, (k, n)) for b in ("pos", "neg")}
    V = rng.uniform(0.05, 0.6, (m, k))
    return V, G


def _ideal(V, G, R, C, rs, cs):
    I0_col = {b: np.stack([V[:, p * rs:(p + 1) * rs] @ G[b][p * rs:(p + 1) * rs]
                           for p in range(R)], axis=1) for b in G}
    I0_row = np.empty((V.shape[0], R * rs, 2 * C))
    for q in range(C):
        for t, b in enumerate(("pos", "neg")):
            I0_row[:, :, 2 * q + t] = V * G[b][:, q * cs:(q + 1) * cs].sum(axis=1)
    return I0_row, I0_col


@pytest.mark.parametrize("bus", ["shared", "buffered"])
@pytest.mark.parametrize("R,C", [(1, 1), (3, 2), (2, 3)])
def test_layer_solve_matches_mna(R, C, bus):
    rs = cs = 4
    _, _, P = _params(rs, cs, input_bus=bus)
    V, G = _random_layer(np.random.default_rng(R * 10 + C), R, C, rs, cs)
    LL = LayerLoading(G, P)
    I0_row, I0_col = _ideal(V, G, R, C, rs, cs)
    sol = LL.solve(I0_row, I0_col, np.ones_like(V), tol_V=1e-12, max_iter=400)
    ref = layer_mna(V, G, P, R, C, rs, cs)
    for b in ("pos", "neg"):
        assert np.allclose(sol.I_col[b], ref[b], rtol=1e-6, atol=1e-12), b
    # the loading is real: currents dropped
    assert sum(ref[b].sum() for b in ref) < sum(I0_col[b].sum() for b in I0_col)


def test_boundary_check_is_consistent():
    rs = cs = 4
    R, C = 2, 2
    _, _, P = _params(rs, cs, input_bus="shared")
    V, G = _random_layer(np.random.default_rng(3), R, C, rs, cs)
    LL = LayerLoading(G, P)
    I0_row, I0_col = _ideal(V, G, R, C, rs, cs)
    sol = LL.solve(I0_row, I0_col, np.ones_like(V), tol_V=1e-13, max_iter=400)
    d_in, d_out = LL.boundary_from_currents(sol.I_row, sol.I_col)
    assert np.allclose(d_in, sol.d_in, atol=1e-9)
    for b in d_out:
        assert np.allclose(d_out[b], sol.d_out[b], atol=1e-9)


# ---------------------------------------------------------------------------
# Floorplan / NoC bookkeeping
# ---------------------------------------------------------------------------
def test_segments_and_tiles():
    _, _, P = _params(4, 4)
    s_in = input_bus_segments(3, P)
    assert len(s_in) == 6 and s_in[0] == P.subarray_gap_um
    # tile_sa_cols = 2 -> crossing from position 1 to 2 adds the tile gap
    assert s_in[4] == pytest.approx(P.subarray_w_um + P.subarray_gap_um + P.tile_gap_um)
    s_out = column_line_segments(4, P)
    assert len(s_out) == 4
    g = layer_grid(40, 20, P)
    assert (g.R, g.C, g.tiles_r, g.tiles_c) == (10, 5, 5, 3)
    assert wire_r_per_um(45) == pytest.approx(3.2)
    assert wire_r_per_um(180) == pytest.approx(0.24)


def test_placement_and_noc_volume_match_converter_counts():
    ic, hw, P = _params(16, 16)
    ic.noc = "analytic"
    ledger = [{"tag": "Q", "m": 5, "k": 192, "n": 32},
              {"tag": "W1", "m": 5, "k": 192, "n": 384},
              {"tag": "W2", "m": 5, "k": 384, "n": 192}]
    plan = place_layers([(L["tag"], L["k"], L["n"]) for L in ledger], P)
    occ = set()
    for pl in plan.layers:
        for r in range(pl.grid.tiles_r):
            for c in range(pl.grid.tiles_c):
                xy = (pl.x0 + c, pl.y0 + r)
                assert xy not in occ and xy[0] < plan.width
                occ.add(xy)
    tr = build_transfers(plan, ledger, P, dac_bits=8, adc_bits=8)
    bits_in = sum(t.bits for t in tr if t.phase == "in")
    bits_out = sum(t.bits for t in tr if t.phase == "out")
    assert bits_in == 8 * sum(L["m"] * L["k"] for L in ledger)      # = DAC convs
    assert bits_out == 8 * sum(L["m"] * L["n"] for L in ledger)     # = ADC convs
    res = estimate_noc(ledger, P, ic, dac_bits=8, adc_bits=8, vdd=1.0)
    assert res["energy_J"] > 0 and res["latency_s"] > 0
    assert res["bits"] == bits_in + bits_out


def test_noc_multicast_sends_shared_input_once():
    from spicecimx.arch.noc import multicast_groups
    ic, hw, P = _params(16, 16)
    ic.noc = "analytic"
    ledger = [{"tag": f"h{h}_{x}", "m": 5, "k": 192, "n": 32}
              for h in range(2) for x in "QKV"] + [{"tag": "WO", "m": 5, "k": 192, "n": 192}]
    assert multicast_groups(ledger) == [0] * 6 + [-1]
    ic.noc_multicast = True
    mc = estimate_noc(ledger, P, ic, dac_bits=8, adc_bits=8, vdd=1.0)
    uni = mc["unicast_analytic"]
    in_once = 8 * 5 * 192
    out_bits = 8 * 5 * (6 * 32 + 192)
    assert mc["bits"] == in_once + 8 * 5 * 192 + out_bits      # QKV once + WO + outputs
    assert uni["bits"] == 6 * in_once + 8 * 5 * 192 + out_bits
    assert mc["energy_J"] < uni["energy_J"] and mc["latency_s"] < uni["latency_s"]
    ic.noc_multicast = False
    off = estimate_noc(ledger, P, ic, dac_bits=8, adc_bits=8, vdd=1.0)
    assert off["energy_J"] == pytest.approx(uni["energy_J"])


def test_config_parses_interconnect(tmp_path):
    from spicecimx.config import load_cfg
    ini = tmp_path / "c.ini"
    ini.write_text(
        "[DATA]\ndataset = mnist\ndata_root = .\nimg_height = 28\nimg_width = 28\n"
        "channels = 1\nval_split = 0.1\nnum_workers = 0\naugment = false\n"
        "[MODEL]\npatch_size = 4\nd_model = 32\nn_heads = 2\nd_k = 0\nd_v = 0\n"
        "depth = 1\nmlp_ratio = 2\ndropout = 0\nnum_classes = 10\n"
        "use_class_token = true\npos_embed = learnable\n"
        "[TRAIN]\nepochs = 1\nbatch_size = 1\nlr = 1e-3\nweight_decay = 0\n"
        "betas = 0.9,0.999\nwarmup_epochs = 0\nlr_scheduler = none\nseed = 0\n"
        "amp = false\ndevice = cpu\nlog_interval = 1\n"
        "[SAVE]\nout_dir = .\nsave_weights = true\nsave_best = true\n"
        "[INTERCONNECT]\nenable = true\nloading_max_passes = 3\ntia_r_in_ohm = auto\n"
        "noc = booksim   # comment\nbooksim_binary = tools/booksim\nnoc_flit_bits = 128\n")
    cfg = load_cfg(str(ini))
    ic = cfg.interconnect
    assert ic.enable and ic.loading_max_passes == 3 and ic.tia_r_in_ohm < 0
    assert ic.noc == "booksim" and ic.noc_flit_bits == 128
    assert os.path.isabs(ic.booksim_binary)
    P = resolve_interconnect(ic, cfg.hw, cfg.pdk, cfg.peripherals)
    # auto TIA input resistance: (vmax_tia - vmin_tia) / i_fs / (1 + A)
    expect = (cfg.peripherals.vmax_tia - cfg.peripherals.vmin_tia) \
        / cfg.peripherals.adc_i_fullscale / (1 + ic.tia_open_loop_gain)
    assert P.r_tia == pytest.approx(expect)


def test_netlist_sl_termination_and_named_pwl():
    from spicecimx.hw.netlist import generate_subarray
    from spicecimx.sim.templates import render_pwl_sources_text
    std = os.path.join(SUPPORTS, "standard_cell.netlist")
    g = generate_subarray(4, 4, std, sl_termination="ground")
    s = generate_subarray(4, 4, std, sl_termination="source")
    assert "IPRB0 (net1i VSS) iprobe" in g and "SLT" not in g.split("IPRB0")[1]
    assert "IPRB3 (net7i SLT3) iprobe" in s
    with pytest.raises(ValueError):
        generate_subarray(4, 4, std, sl_termination="bogus")
    # decoder address inputs must never float (crowbar static current)
    for rows, nb in ((4, 2), (16, 4), (32, 5)):
        net = generate_subarray(rows, 4, std)
        for i in range(nb):
            assert f"VADDR{i} (A{i} VSS) vsource dc=0" in net
        assert f"VADDR{nb} " not in net
    txt, M, _ = render_pwl_sources_text(np.full((2, 3), 0.1), 1e-9, 1e-12, 1e-12, 3,
                                        name="VSL", node="SLT", ref="VSS")
    assert M == 2 and txt.startswith("VSL0 (SLT0 VSS) vsource")


def test_tran_maxstep_is_rendered(tmp_path):
    from spicecimx.sim.templates import render_testbench_transient
    kw = dict(template_path=os.path.join(SUPPORTS, "_golden_input.scs"), sim_time=1e-8,
              input_sources="", device_model_path="d.scs", transistor_model="m.scs",
              process_corner="tt", vdd=1.1, vss=0.0, temperature=25.0,
              iprobe_save_list=["IPRB0:in"])
    render_testbench_transient(output_path=str(tmp_path / "a.scs"), **kw)
    render_testbench_transient(output_path=str(tmp_path / "b.scs"), maxstep=2e-12, **kw)
    a, b = (tmp_path / "a.scs").read_text(), (tmp_path / "b.scs").read_text()
    assert "maxstep" not in a
    assert "tran tran stop=1e-08 maxstep=2e-12" in b


# ---------------------------------------------------------------------------
# Loaded RSN through the real transient backend
# ---------------------------------------------------------------------------
def _run_backend(tmp_path, monkeypatch, *, loading, passes=2, R=2, C=2,
                 bus="buffered"):
    from spicecimx.sim import transient_backend as tb
    from spicecimx.core.imc_matmul import derive_affine
    rs = cs = 4
    hw = HwCfg(subarray_rows=rs, subarray_cols=cs, pe_rows=1, pe_cols=1,
               tile_rows=2, tile_cols=2, vmin=0.0, vmax=0.6, R_access=0.0)
    sim = SimCfg(mode="transient", license_count=4, pulse_width=1e-9,
                 trise=1e-12, tfall=1e-12, postprocess_workers=2)
    paths = PathsCfg(project_dir=str(tmp_path), hwsimdb_root=str(tmp_path / "hw"),
                     tmp_root=str(tmp_path / "tmp"),
                     golden_input_transient=os.path.join(SUPPORTS, "_golden_input.scs"),
                     stdcell_netlist=os.path.join(SUPPORTS, "standard_cell.netlist"),
                     device_model_transient=os.path.join(SUPPORTS, "reram_cell.v"))
    ctx = tb.build_context(hw, sim, paths, batch_id="b")
    ic, _, P = _params(rs, cs, loading_max_passes=passes, loading_tol_V=1e-9,
                       input_bus=bus)
    ic.loading = loading
    ctx.interconnect, ctx.ic_params = ic, P
    alpha = hw.Ik1 / hw.Vref
    inc = 3 * sim.pulse_width + sim.trise + sim.tfall

    def fake_cmd(input_deck, raw_dir, log_path, extra_args=(), **kw):
        return [sys.executable, FAKE_DECK, os.path.abspath(input_deck),
                "-raw", os.path.abspath(raw_dir), f"ALPHA={alpha!r}",
                f"TOX={hw.Tox!r}", f"TREF={hw.Tref!r}", f"INC={inc!r}"]
    monkeypatch.setattr(tb, "build_spectre_command", fake_cmd)

    rng = np.random.default_rng(7)
    m, k, n = 3, R * rs, C * cs
    A = rng.uniform(0, 1, (m, k))
    B = rng.standard_normal((k, n))
    Gmax = float(alpha * np.exp(-(hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    Gmin = float(alpha * np.exp(-(hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    sc = derive_affine(A, B, vmin=hw.vmin, vmax=hw.vmax, Gmin=Gmin, Gmax=Gmax)
    I = tb._transient_imc(A=A, B=B, scales=sc, tag="L", bname="block_00",
                          dump_dir=None, ctx=ctx)
    # reference: the same (T-rounded) conductances as the decks
    from spicecimx.hw.device_emit import convert_to_conductance
    T_pos, T_neg = convert_to_conductance(B, sc.c, Gmin, hw.Tmin, hw.Tox, hw.Tref,
                                          alpha, R_access=0.0, Tmax=hw.Tmax)
    tg = lambda T: alpha * np.exp(-(hw.Tox - np.vectorize(lambda x: float(f"{x:.4e}"))(T))
                                  / (hw.Tox - hw.Tref))
    G = {"pos": tg(T_pos), "neg": tg(T_neg)}
    V = sc.a * A + sc.b
    return I, V, G, P, ctx, (R, C, rs, cs)


def test_backend_without_loading_is_ideal(tmp_path, monkeypatch):
    I, V, G, P, ctx, (R, C, rs, cs) = _run_backend(tmp_path, monkeypatch, loading=False)
    assert np.allclose(I, V @ G["pos"] - V @ G["neg"], rtol=1e-9, atol=1e-15)
    assert ctx.loading_records == []


@pytest.mark.parametrize("bus", ["buffered", "shared"])
def test_backend_loaded_rsn_matches_full_layer_mna(tmp_path, monkeypatch, bus):
    I, V, G, P, ctx, (R, C, rs, cs) = _run_backend(tmp_path, monkeypatch,
                                                   loading=True, bus=bus)
    ref = layer_mna(V, G, P, R, C, rs, cs)
    I_ref = ref["pos"].sum(axis=1) - ref["neg"].sum(axis=1)
    ideal = V @ G["pos"] - V @ G["neg"]
    assert np.max(np.abs(I - I_ref)) < 1e-6 * np.max(np.abs(I_ref)), \
        "loaded RSN (2 SPICE passes) must reproduce the full-layer nodal solution"
    assert np.max(np.abs(ideal - I_ref)) > 1e-3 * np.max(np.abs(I_ref)), \
        "test parameters must produce a visible loading effect"
    rec = ctx.loading_records[0]
    assert rec["passes"] == 2 and rec["accepted_pass"] == 2
    # buffered: only the column lines couple -> the check is exact.  shared:
    # the post-pass row split is a KCL-scaled model estimate (the sources'
    # terminal current is not used), so the check is approximate.
    assert rec["residual_V"] < (1e-6 if bus == "buffered" else 2e-3)
    assert rec["d_out_max_V"] > 0
    assert (rec["d_in_max_V"] > 0) == (bus == "shared")
    # accepted waveforms sit in the standard hwsimdb layout, pass 1 in tmp
    layer_dir = os.path.join(ctx.hwsimdb_dir, "block_00_L")
    trans = [r for r, _, f in os.walk(layer_dir) if "tran.tran.tran" in f]
    assert len(trans) == 2 * R * C
    assert os.path.exists(os.path.join(layer_dir, "loading.json"))
