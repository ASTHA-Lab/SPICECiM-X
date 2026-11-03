"""
Regression tests for the v1.1 release fixes (numpy-only; no Spectre, no torch
except for the clean-workspace test, which is skipped without torch).

a. derive_affine keeps one-signed B inside the conductance window
b. PolarityCache ignores non-finite entries / does not cache an all-NaN call
c. imc_matmul keys polarity + TIA calibration by "bname/tag"
d. TIACalibrator freeze_after; tia_adc_realistic NaN propagation and Vfs scaling
e. spectre.check_failed_jobs policies
f. templates.pad_rows_to_subarrays
g. subckt_registry: Virtuoso-escaped buses, (..) port lists, multi-netlist lookup
h. peripheral_chain: DAC / readout transfer parsing, decreasing ADC transfer
i. config.load_cfg: path resolution, inline comments, '%', validation
j. cli.infer.prepare_clean_workspace only touches <root>/<batch>
k. device_emit.convert_to_conductance clips T to [Tmin, Tmax]
"""

import os
import shutil
import sys
import types

import numpy as np
import pytest

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_TOP = os.path.dirname(PKG_ROOT)
SUPPORTS = os.path.join(REPO_TOP, "supports")
PERIPH_NETLIST = os.path.join(SUPPORTS, "peripherals", "peripherals_8b.scs")
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.core.imc_matmul import (                       # noqa: E402
    derive_affine, ideal_current, imc_matmul, PolarityCache,
)


# ---------------------------------------------------------------------------
# a. derive_affine
# ---------------------------------------------------------------------------
GMIN, GMAX = 1e-7, 5e-5


@pytest.mark.parametrize("sign", [+1, -1])
def test_derive_affine_one_signed_B_stays_in_window(sign):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 6))
    B = sign * rng.uniform(0.2, 1.5, size=(6, 5))
    sc = derive_affine(A, B, 0.0, 1.0, GMIN, GMAX)
    assert sc.c * np.max(np.abs(B)) + GMIN <= GMAX + 1e-15
    # the window is still fully used by the largest |B|
    assert sc.c * np.max(np.abs(B)) + GMIN == pytest.approx(GMAX, rel=1e-12)


def test_derive_affine_mixed_sign_unchanged():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 6))
    B = rng.standard_normal((6, 5))
    assert B.min() < 0 < B.max()
    sc = derive_affine(A, B, 0.0, 1.0, GMIN, GMAX)
    assert sc.c == pytest.approx((GMAX - GMIN) / (B.max() - B.min()), rel=1e-15)
    assert sc.d == pytest.approx(GMIN - sc.c * B.min(), rel=1e-15)
    assert sc.a == pytest.approx(1.0 / (A.max() - A.min()), rel=1e-15)
    assert sc.c * np.max(np.abs(B)) + GMIN <= GMAX + 1e-15


def test_derive_affine_zero_B_finite_scale():
    sc = derive_affine(np.ones((2, 3)), np.zeros((3, 2)), 0.0, 1.0, GMIN, GMAX)
    assert sc.c == pytest.approx(GMAX - GMIN) and sc.a == 1.0


# ---------------------------------------------------------------------------
# b. PolarityCache.decide
# ---------------------------------------------------------------------------
def test_polarity_all_nan_returns_plus_and_does_not_cache():
    pc = PolarityCache()
    ideal = np.arange(1.0, 7.0).reshape(2, 3)
    assert pc.decide("L", np.full((2, 3), np.nan), ideal) == +1
    assert pc.get("L") is None
    # a later healthy (sign-flipped) call still decides the orientation
    assert pc.decide("L", -ideal, ideal) == -1
    assert pc.get("L") == -1
    assert pc.decide("L", ideal, ideal) == -1          # cached now


def test_polarity_partial_nan_ignored():
    pc = PolarityCache()
    ideal = np.linspace(1.0, 2.0, 12).reshape(3, 4)
    I = -ideal.copy()
    I[0, :] = np.nan
    I[2, 3] = np.inf
    assert pc.decide("L", I, ideal) == -1
    assert pc.get("L") == -1
    pc2 = PolarityCache()
    I2 = ideal.copy()
    I2[1, 1] = np.nan
    assert pc2.decide("L", I2, ideal) == +1


# ---------------------------------------------------------------------------
# c. imc_matmul keys polarity and tia_apply by "bname/tag"
# ---------------------------------------------------------------------------
def test_imc_matmul_layer_key_includes_bname():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((3, 5))
    B = rng.standard_normal((5, 4))
    seen = []

    def tia(I, key):
        seen.append(key)
        return np.zeros_like(I), I, {"k_nom": 1.0}

    pc = PolarityCache()
    outs = []
    for bname, sign in (("blk0", +1.0), ("blk1", -1.0)):
        outs.append(imc_matmul(
            A, B, Gmax=GMAX, Gmin=GMIN,
            sim_backend=lambda sign=sign, **kw: sign * ideal_current(
                kw["A"], kw["B"], kw["scales"]),
            polarity_cache=pc, tia_apply=tia, tag="h0_Q", bname=bname))
    assert seen == ["blk0/h0_Q", "blk1/h0_Q"]
    # independent orientations per block, both recovered
    assert pc.get("blk0/h0_Q") == +1 and pc.get("blk1/h0_Q") == -1
    assert pc.get("h0_Q") is None
    for C in outs:
        np.testing.assert_allclose(C, A @ B, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# d. TIA / ADC
# ---------------------------------------------------------------------------
def test_tia_calibrator_freeze_after():
    from spicecimx.analog.tia_adc import TIACalibrator
    x1 = np.linspace(-1.0, 1.0, 101)
    x2 = np.linspace(-5.0, 5.0, 101)

    frozen = TIACalibrator(p_lo=0.0, p_hi=100.0, freeze_after=1)
    frozen.update("L", x1)
    r1 = frozen.get("L")
    assert frozen.is_frozen("L")
    frozen.update("L", x2)
    assert frozen.get("L") == r1 == (-1.0, 1.0)

    live = TIACalibrator(p_lo=0.0, p_hi=100.0, freeze_after=0)
    live.update("L", x1)
    live.update("L", x2)
    assert live.get("L") == (-5.0, 5.0)
    assert not live.is_frozen("L")

    # non-finite entries are ignored; all-NaN does not count as a call
    c = TIACalibrator(p_lo=0.0, p_hi=100.0, freeze_after=1)
    c.update("L", np.full(4, np.nan))
    with pytest.raises(KeyError):
        c.get("L")
    c.update("L", np.array([np.nan, 2.0, 3.0]))
    assert c.get("L") == (2.0, 3.0)


def test_tia_adc_realistic_nan_propagates():
    from spicecimx.analog.tia_adc import tia_adc_realistic
    I = np.array([[1e-6, np.nan, 3e-6], [np.inf, 5e-6, 2e-6]])
    Vq, Ih, meta = tia_adc_realistic(I, 0.0, 1.0, 8, rng=np.random.default_rng(0))
    bad = ~np.isfinite(I)
    assert np.all(np.isnan(Vq[bad])) and np.all(np.isnan(Ih[bad]))
    assert np.all(np.isfinite(Vq[~bad])) and np.all(np.isfinite(Ih[~bad]))
    # calibration range came from the finite entries only
    assert meta["Imin"] == 1e-6 and meta["Imax"] == 5e-6


def test_tia_adc_realistic_vfs_maps_imax_to_top_code():
    from spicecimx.analog.tia_adc import tia_adc_realistic
    Nbits, L = 8, 255
    Imin, Imax = -2e-5, 6e-5
    I = np.linspace(Imin, Imax, 51).reshape(3, 17)
    Vq, Ih, meta = tia_adc_realistic(
        I, 0.0, 0.5, Nbits, Imin=Imin, Imax=Imax, Vfs=0.5, headroom=1.0,
        return_debug=True, rng=np.random.default_rng(0))
    codes = meta["codes"]
    assert meta["Vfs"] == pytest.approx(0.5)           # not Vfs * Vmax
    assert codes.flat[-1] == L and codes.flat[0] == 0
    # linear, unclipped mapping across the whole range
    expected = np.rint((I - Imin) / (Imax - Imin) * L)
    assert np.max(np.abs(codes - expected)) <= 1      # rounding ties only
    lsb_I = (Imax - Imin) / L
    assert np.max(np.abs(Ih - I)) <= 0.5 * lsb_I + 1e-18


# ---------------------------------------------------------------------------
# e. check_failed_jobs
# ---------------------------------------------------------------------------
def _result(i, ok, post, error=None):
    from spicecimx.sim.spectre import SpectreJob, SimResult
    job = SpectreJob(job_id=f"L/Array_0_{i}/pos", cmd=[sys.executable], cwd=".",
                     raw_dir=".", branch="pos", subarray_row=0, subarray_col=i)
    return SimResult(job=job, ok=ok, post=post, error=error)


def test_check_failed_jobs_policies():
    from spicecimx.sim.spectre import check_failed_jobs, SpectreJobsFailed
    good = _result(0, True, {"currents": 0})
    bad = _result(1, False, None, error="spectre exited 1")
    nopost = _result(2, True, None)

    assert check_failed_jobs([good], tag="L", n_jobs=1, policy="raise") == []
    with pytest.raises(SpectreJobsFailed) as ei:
        check_failed_jobs([good, bad], tag="L", n_jobs=2, policy="raise")
    assert "L/Array_0_1/pos" in str(ei.value)
    assert issubclass(SpectreJobsFailed, RuntimeError)

    failed = check_failed_jobs([good, bad, nopost], tag="L", n_jobs=3, policy="nan")
    assert failed == [bad, nopost]
    with pytest.raises(SpectreJobsFailed):
        check_failed_jobs([good, nopost], tag="L", n_jobs=2, policy="raise")


# ---------------------------------------------------------------------------
# f. pad_rows_to_subarrays
# ---------------------------------------------------------------------------
def test_pad_rows_to_subarrays():
    from spicecimx.sim.templates import pad_rows_to_subarrays
    V = np.arange(1.0, 15.0).reshape(2, 7)
    P = pad_rows_to_subarrays(V, 4)
    assert P.shape == (2, 8) and P.dtype == V.dtype
    np.testing.assert_array_equal(P[:, :7], V)
    np.testing.assert_array_equal(P[:, 7:], 0.0)
    assert pad_rows_to_subarrays(V, 7) is V                 # already a multiple
    assert pad_rows_to_subarrays(V, 3).shape == (2, 9)
    assert pad_rows_to_subarrays(V, 16).shape == (2, 16)


# ---------------------------------------------------------------------------
# g. subckt_registry
# ---------------------------------------------------------------------------
def test_parse_one_subckt_virtuoso_escaped_bus():
    from spicecimx.hw.subckt_registry import parse_one_subckt
    line = ("subckt ADC_8b IN CLK " +
            " ".join(rf"OUT\<{i}\>" for i in range(7, -1, -1)) + " VDD VSS")
    d = parse_one_subckt(line)
    assert d.name == "ADC_8b"
    assert d.buses == {"OUT": [7, 6, 5, 4, 3, 2, 1, 0]}
    assert d.scalars == ["IN", "CLK", "VDD", "VSS"]
    assert d.bit_width == 8
    assert d.ports[2] == "OUT<7>" and d.ports[9] == "OUT<0>"


def test_parse_one_subckt_parenthesised_and_range():
    from spicecimx.hw.subckt_registry import parse_one_subckt
    d = parse_one_subckt("subckt X (A B C) // comment")
    assert d.name == "X" and d.ports == ["A", "B", "C"] and d.bit_width is None
    d2 = parse_one_subckt("subckt DAC (D<3:0> OUT VDD VSS) vfs=1")
    assert d2.buses == {"D": [3, 2, 1, 0]} and d2.scalars == ["OUT", "VDD", "VSS"]
    assert parse_one_subckt("inline subckt Y A B") is None


def _registry(netlists):
    from spicecimx.hw.subckt_registry import SubcktRegistry
    roles = {"dac": ("DAC_8b", "H"), "adc": ("ADC_8b", "H"),
             "tia": ("TIA_8b", "H"), "subtractor": ("SUBTRACTOR_8b", "H"),
             "multiplexer": ("MUX", "S")}
    return SubcktRegistry.from_config(roles, netlists)


def test_registry_from_netlist_list(tmp_path):
    assert os.path.exists(PERIPH_NETLIST), PERIPH_NETLIST
    missing = str(tmp_path / "does_not_exist.netlist")
    std = os.path.join(SUPPORTS, "standard_cell.netlist")
    reg = _registry([missing, std, PERIPH_NETLIST])
    assert reg.bits("dac") == 8 and reg.bits("adc") == 8
    assert reg.definition("dac").buses["INP"] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert reg.definition("adc").scalars == ["IN", "CLK", "VDD", "VSS"]
    assert reg.definition("tia").ports == ["IN", "OUT", "VDD", "VSS"]
    assert reg.definition("subtractor").ports == ["INP", "INN", "OUT", "VDD", "VSS"]
    assert reg.is_hardware("adc") and not reg.is_hardware("multiplexer")
    # a single path (str) still works
    assert _registry(PERIPH_NETLIST).bits("dac") == 8
    # an H role that is in none of the netlists fails loudly
    with pytest.raises(ValueError):
        _registry([missing])


# ---------------------------------------------------------------------------
# h. peripheral_chain
# ---------------------------------------------------------------------------
def _psf_stream(t, traces):
    """Flat psfascii-like (params, values): per time step '"time"' then each
    trace, as read_psfascii_tran returns them."""
    params, values = [], []
    for j, tj in enumerate(t):
        params.append('"time"'); values.append(tj)
        for name, arr in traces.items():
            params.append(f'"{name}"'); values.append(arr[j])
    return np.array(params), np.array(values, dtype=float)


def test_parse_dac_transfer_synthetic():
    from spicecimx.analog.peripheral_chain import emit_dac_char_deck, parse_dac_transfer
    dac_def = _registry(PERIPH_NETLIST).definition("dac")
    deck, meta = emit_dac_char_deck(
        dac_def, transistor_model="pdk.scs", process_corner="tt", vdd=1.1,
        vss=0.0, temperature=25.0, code_dwell=1e-9, max_codes=16)
    assert meta.bits == 8 and meta.n_codes == 16 and meta.inp_base == "INP"
    assert "XDAC (D7 D6 D5 D4 D3 D2 D1 D0 DAC_OUT DVDD 0) DAC_8b" in deck

    rng = np.random.default_rng(3)
    V_true = np.sort(rng.uniform(0.0, 0.6, 16))
    t = (np.arange(16 * 50) + 0.5) * (1e-9 / 50)    # samples strictly inside steps
    code_t = np.minimum((t // 1e-9).astype(int), 15)
    i_sup = np.full(t.size, 2e-6)
    params, values = _psf_stream(t, {"DAC_OUT": V_true[code_t], "VDACVDD:p": i_sup})
    V_dac, e = parse_dac_transfer(lambda p: (params, values), "unused", meta, vdd=1.1)
    np.testing.assert_allclose(V_dac, V_true, rtol=1e-12)
    # constant supply current -> energy = vdd * I * dwell per conversion
    assert e == pytest.approx(1.1 * 2e-6 * 1e-9, rel=1e-9)
    # with t_conv the settled current is charged for t_conv only
    _, e2 = parse_dac_transfer(lambda p: (params, values), "unused", meta,
                               vdd=1.1, t_conv=2e-10)
    assert e2 == pytest.approx(1.1 * 2e-6 * 2e-10, rel=1e-9)


def test_dac_energy_uses_per_step_settled_current():
    """Energy charges each step's SETTLED supply current (not the global
    minimum); a transient spike above it is added as dynamic charge."""
    from spicecimx.analog.peripheral_chain import emit_dac_char_deck, parse_dac_transfer
    dac_def = _registry(PERIPH_NETLIST).definition("dac")
    dwell, n = 1e-9, 4
    _, meta = emit_dac_char_deck(
        dac_def, transistor_model="pdk.scs", process_corner="tt", vdd=1.0,
        vss=0.0, temperature=25.0, code_dwell=dwell, max_codes=n)
    dt = dwell / 100
    t = np.arange(n * 100) * dt + dt / 2           # samples strictly inside steps
    code_t = (t // dwell).astype(int)
    i_set = 1e-6 * (1 + code_t)                    # 1, 2, 3, 4 uA settled
    i_sup = i_set.copy()
    spike = (t % dwell) < 10 * dt                  # +5 uA for the first 10 samples
    i_sup[spike] += 5e-6
    params, values = _psf_stream(t, {"DAC_OUT": 0.1 * code_t, "VDACVDD:p": i_sup})
    t_conv = 3e-10
    V_dac, e = parse_dac_transfer(lambda p: (params, values), "unused", meta,
                                  vdd=1.0, t_conv=t_conv)
    np.testing.assert_allclose(V_dac, 0.1 * np.arange(n), atol=1e-12)
    static = np.mean(1e-6 * np.arange(1, n + 1)) * t_conv
    assert e > static                              # spike adds dynamic charge
    # dynamic part ~ 5 uA x 10 samples (trapezoid / sample-edge effects ~10%)
    assert e - static == pytest.approx(5e-6 * 10 * dt, rel=0.15)
    # the old min-current floor would have charged 1 uA for every step
    assert static > 1e-6 * t_conv * 2


def test_mean_in_interpolates_empty_window():
    from spicecimx.analog.peripheral_chain import _mean_in
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 10.0, 20.0])
    assert _mean_in(t, v, 0.9, 1.1) == 10.0
    assert _mean_in(t, v, 1.2, 1.4) == pytest.approx(13.0)   # no samples -> interp
    assert _mean_in(t, v, 3.0, 4.0) is None                  # outside the trace


def test_parse_readout_transfer_synthetic():
    from spicecimx.analog.peripheral_chain import (
        emit_readout_char_deck, parse_readout_transfer)
    reg = _registry(PERIPH_NETLIST)
    i_sweep = np.linspace(-1e-4, 1e-4, 9)
    deck, meta = emit_readout_char_deck(
        reg.definition("tia"), reg.definition("subtractor"), reg.definition("adc"),
        i_sweep=i_sweep, transistor_model="pdk.scs", process_corner="tt",
        vdd=1.1, vss=0.0, temperature=25.0, step_dwell=4e-9)
    assert meta.adc_bits == 8 and meta.adc_out_base == "OUT"
    assert meta.clk_node == "ACLK"
    codes_true = np.array([250, 200, 160, 128, 100, 64, 30, 5, 0])   # decreasing
    dt = 4e-9 / 40
    t = (np.arange(9 * 40) + 0.5) * dt
    step = np.minimum((t // 4e-9).astype(int), 8)
    traces = {f"ADC_OUT_{b}": 1.1 * ((codes_true[step] >> b) & 1)
              for b in range(8)}
    traces.update({"VTIAVDD:p": np.full(t.size, 1e-5),
                   "VSUBVDD:p": np.full(t.size, 2e-5),
                   "VADCVDD:p": np.full(t.size, 3e-5)})
    params, values = _psf_stream(t, traces)
    i_vals, codes, e_tia, e_sub, e_adc = parse_readout_transfer(
        lambda p: (params, values), "unused", meta, vdd=1.1)
    np.testing.assert_allclose(i_vals, i_sweep)
    np.testing.assert_array_equal(codes, codes_true)
    assert e_tia == pytest.approx(1.1 * 1e-5 * 4e-9, rel=1e-9)
    assert e_sub == pytest.approx(2 * e_tia, rel=1e-9)
    assert e_adc == pytest.approx(3 * e_tia, rel=1e-9)


def test_tia_adc_apply_decreasing_transfer():
    from spicecimx.analog.peripheral_chain import PhysicalConverters
    i_fs = 1e-4
    xs = np.linspace(-i_fs, i_fs, 257)
    codes = np.clip(np.rint(127.5 - 1.1 * xs / i_fs * 127.5), 0, 255).astype(int)
    pc = PhysicalConverters(adc_hw=True, adc_bits=8, adc_i_values=xs,
                            adc_codes=codes, i_fullscale=i_fs)
    rng = np.random.default_rng(4)
    I = rng.uniform(-0.85 * i_fs, 0.85 * i_fs, size=(6, 7))
    c_out, vals = pc.tia_adc_apply(I)
    assert c_out.shape == I.shape
    # decreasing: larger current -> smaller code
    order = np.argsort(I.ravel())
    assert np.all(np.diff(c_out.ravel()[order]) <= 0)
    lsb = 2 * i_fs / (1.1 * 255)
    assert np.max(np.abs(vals - I)) <= lsb
    # saturates to the end codes outside the sweep
    c_sat, _ = pc.tia_adc_apply(np.array([[-2 * i_fs, 2 * i_fs]]))
    assert c_sat.tolist() == [[255, 0]]


# ---------------------------------------------------------------------------
# i. config.load_cfg
# ---------------------------------------------------------------------------
def _base_sections():
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str
    cp.read(os.path.join(REPO_TOP, "config_cifar10.ini"), encoding="utf-8-sig")
    out = []
    for sec in ("DATA", "MODEL", "TRAIN", "SAVE"):
        out.append(f"[{sec}]")
        out += [f"{k} = {v}" for k, v in cp.items(sec)]
        out.append("")
    return "\n".join(out)


PATH_LINES = """
[PATHS]
project_dir = .   # the folder holding this ini
images_dir = ./Test_Image
dump_root = imc_traces
hwsimdb_root = ./hwsimdb
tmp_root = tmp
golden_input_transient = ./supports/_golden_input.scs
golden_input_dc = ./supports/_golden_input_dc.scs
stdcell_netlist = supports/standard_cell.netlist
device_model_transient = ./supports/reram_cell.v
device_model_dc = ./supports/sky130_dc_memristor_dc.v
transistor_model = ./pdk/models.scs
char_table_path = ./supports/cell_table.npz
array_table_path = ./supports/array_table.npz
peripheral_netlist = ./supports/peripherals/peripherals_8b.scs
"""


def _write_ini(tmp_path, sim_lines):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    ini = proj / "cfg.ini"
    ini.write_text(_base_sections() + PATH_LINES + "\n[SIM]\n" + sim_lines +
                   "\n[ACCURACY]\nlabel_from_filename = ^(\\d+)_%s.*\\.png   ; regex\n",
                   encoding="utf-8")
    return proj, ini


def test_load_cfg_paths_comments_percent(tmp_path, monkeypatch):
    from spicecimx.config import load_cfg, _PATH_KEYS
    proj, ini = _write_ini(tmp_path, "mode = dc  # inline comment\n"
                                     "hybrid_transient_fraction = 0.25 # 25%\n"
                                     "on_job_failure = nan\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = load_cfg(str(ini))

    root = os.path.normcase(os.path.abspath(str(proj)))
    assert os.path.normcase(cfg.paths.project_dir) == root
    for k in _PATH_KEYS:
        v = getattr(cfg.paths, k)
        assert v and os.path.isabs(v), (k, v)
        assert os.path.normcase(v).startswith(root + os.sep), (k, v)
    assert cfg.paths.tmp_root == os.path.join(str(proj), "tmp")
    assert cfg.sim.mode == "dc"
    assert cfg.sim.on_job_failure == "nan"
    assert cfg.sim.hybrid_transient_fraction == 0.25
    assert cfg.accuracy.label_from_filename == r"^(\d+)_%s.*\.png"
    assert cfg.data.dataset == "cifar10" and cfg.model.patch_size == 4


def test_load_cfg_rejects_bad_on_job_failure(tmp_path):
    from spicecimx.config import load_cfg
    _, ini = _write_ini(tmp_path, "on_job_failure = explode\n")
    with pytest.raises(ValueError, match="on_job_failure"):
        load_cfg(str(ini))


# ---------------------------------------------------------------------------
# j. cli.infer.prepare_clean_workspace (needs torch: cli/infer imports it)
# ---------------------------------------------------------------------------
def _ws_cfg(project, images, hws, dump, tmp):
    paths = types.SimpleNamespace(
        project_dir=str(project), images_dir=str(images), hwsimdb_root=str(hws),
        dump_root=str(dump), tmp_root=str(tmp),
        stdcell_netlist=str(project / "supports" / "standard_cell.netlist"),
        array_table_path=str(project / "supports" / "array_table.npz"),
        char_table_path=str(project / "supports" / "cell_table.npz"))
    return types.SimpleNamespace(paths=paths,
                                 save=types.SimpleNamespace(out_dir="out"))


def test_prepare_clean_workspace_only_touches_batch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from spicecimx.cli import infer
    monkeypatch.setattr(infer, "_spawn_background_delete", lambda paths: None)

    project = tmp_path / "proj"
    images = project / "batchA"
    images.mkdir(parents=True)
    roots = [project / "hwsimdb", project / "imc_traces", project / "tmp"]
    for r in roots:
        for b in ("batchA", "batchB"):
            (r / b).mkdir(parents=True)
            (r / b / "stale.txt").write_text(b)
    infer.prepare_clean_workspace(_ws_cfg(project, images, *roots))
    for r in roots:
        assert (r / "batchA").is_dir() and not any((r / "batchA").iterdir())
        assert (r / "batchB" / "stale.txt").read_text() == "batchB"
    assert images.is_dir()
    trash = project / "trash"
    assert trash.is_dir() and len(list(trash.rglob("stale.txt"))) == 3


def test_prepare_clean_workspace_refuses_project_dir(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from spicecimx.cli import infer
    monkeypatch.setattr(infer, "_spawn_background_delete", lambda paths: None)
    # hwsimdb_root/<batch> would be an ancestor of the project dir
    images = tmp_path / "work"
    project = tmp_path / "root" / "work" / "proj"
    project.mkdir(parents=True)
    (project / "keep.txt").write_text("x")
    cfg = _ws_cfg(project, images, tmp_path / "root",
                  project / "imc_traces", project / "tmp")
    with pytest.raises(ValueError, match="refusing"):
        infer.prepare_clean_workspace(cfg)
    assert (project / "keep.txt").exists()


# ---------------------------------------------------------------------------
# k. convert_to_conductance clips T to [Tmin, Tmax]
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("R_access", [0.0, 2e3])
def test_convert_to_conductance_clips_thickness(R_access):
    from spicecimx.config import HwCfg
    from spicecimx.hw.device_emit import convert_to_conductance, effective_cell_window
    hw = HwCfg()
    alpha = hw.Ik1 / hw.Vref
    gR = lambda T: alpha * np.exp(-(hw.Tox - T) / (hw.Tox - hw.Tref))   # noqa: E731
    Gmax, Gmin = effective_cell_window(gR(hw.Tmax), gR(hw.Tmin), R_access)
    B = np.array([[1.0, -1.0, 0.5], [0.0, -0.25, 1.0]])
    # c slightly above the window: |B| = 1 would land above Tmax
    c = (Gmax - Gmin) * (1.0 + 1e-9)
    args = (B, c, Gmin, hw.Tmin, hw.Tox, hw.Tref, alpha)
    T_pos_u, _ = convert_to_conductance(*args, R_access=R_access)
    assert T_pos_u.max() > hw.Tmax                     # unclipped overshoots
    T_pos, T_neg = convert_to_conductance(*args, R_access=R_access, Tmax=hw.Tmax)
    for T in (T_pos, T_neg):
        assert np.all(T >= hw.Tmin) and np.all(T <= hw.Tmax)
    assert T_pos[0, 0] == hw.Tmax and T_neg[0, 1] == hw.Tmax
    assert T_pos[0, 1] == hw.Tmin and T_neg[0, 0] == hw.Tmin   # inactive branch
    assert hw.Tmin < T_pos[0, 2] < hw.Tmax                      # interior untouched
    np.testing.assert_allclose(T_pos[0, 2], T_pos_u[0, 2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
