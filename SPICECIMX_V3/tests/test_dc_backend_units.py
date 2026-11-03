"""
Unit tests for the DC-mode pieces of SPICECiM-X v3 (no Spectre needed).

1. render_dc_sources_text   -- one parameter-bound ``vsource dc=vi{c}`` per BL
2. render_dc_altergroups    -- ``paramset`` (one row per image) + ``sweep`` wrapper
3. render_testbench_dc      -- fills the top-level supports/_golden_input_dc.scs
4. PsfDcReader              -- parses the real binary PSF sample samples/dc.dc
5. _postprocess_dc_subarray -- reads per-iteration single-op PSF files
                               (``dc_sweep-NNN_dc_op.dc``) into an (M, ncolumn)
                               current matrix.  The files are synthesised here
                               in the exact binary layout PsfDcReader's
                               single-op path decodes.
6. make_backend dispatch
"""

import os
import shutil
import struct
import sys
import tempfile

import numpy as np

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_TOP = os.path.dirname(PKG_ROOT)
SUPPORTS = os.path.join(REPO_TOP, "supports")
SAMPLE_DC = os.path.join(PKG_ROOT, "samples", "dc.dc")
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from spicecimx.sim.templates import (                       # noqa: E402
    render_dc_sources_text, render_dc_altergroups, render_testbench_dc,
)
from spicecimx.sim.spectre import SpectreJob                # noqa: E402
from spicecimx.sim.dc_backend import _postprocess_dc_subarray  # noqa: E402
from spicecimx.io.parse_dc import PsfDcReader               # noqa: E402


# ---------------------------------------------------------------------------
# helper: write a minimal single-operating-point binary PSF
# ---------------------------------------------------------------------------
def write_single_op_psf(path, values):
    """Write ``{name: float}`` as a single-op DC PSF (sections 0, 1, 4 only).

    Section-4 record: 0x10 | record_id | name_len | name (4-byte padded) |
    type_ref | float64 (all big-endian).  Trailer: section index
    ``(sec_num, offset)*`` + ``Clarissa`` + u32 index offset.
    """
    u32 = struct.Struct(">I").pack
    sec0 = u32(0) + u32(0) + b"HEADER\x00\x00"
    sec1 = u32(1) + u32(0)
    sec4 = u32(4) + u32(0)
    for j, (name, val) in enumerate(values.items()):
        nb = name.encode("ascii")
        sec4 += u32(0x10) + u32(100 + j) + u32(len(nb)) + nb + b"\x00" * ((-len(nb)) & 3)
        sec4 += u32(0x63B) + struct.pack(">d", float(val))
    off0, off1 = 0, len(sec0)
    off4 = off1 + len(sec1)
    body = sec0 + sec1 + sec4
    index_start = len(body)
    index = u32(0) + u32(off0) + u32(1) + u32(off1) + u32(4) + u32(off4)
    with open(path, "wb") as f:
        f.write(body + index + b"Clarissa" + u32(index_start))


def _job(tmp, raw_dir, ncolumn, M):
    return SpectreJob(
        job_id="test/sa0/pos", cmd=[sys.executable, "-c", "pass"], cwd=tmp,
        raw_dir=raw_dir, branch="pos", subarray_row=1, subarray_col=2,
        extra={"ncolumn": ncolumn, "M": M},
    )


# ---------------------------------------------------------------------------
# 1. render_dc_sources_text
# ---------------------------------------------------------------------------
def test_render_dc_sources_text():
    V = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.7]])
    text, M = render_dc_sources_text(V, sweep_var="img_idx", scol=4)
    assert M == 3
    assert "parameters vi0=0 vi1=0" in text
    assert "VI0 (IN0 0) vsource dc=vi0 type=dc" in text
    assert "VI1 (IN1 0) vsource dc=vi1 type=dc" in text
    # per-image values live in the paramset, never in pwl()/table()
    assert "pwl(" not in text and "table(" not in text


# ---------------------------------------------------------------------------
# 2. render_dc_altergroups -> paramset + sweep
# ---------------------------------------------------------------------------
def test_render_dc_altergroups():
    V = np.array([[0.10, 0.50], [0.20, 0.60], [0.30, 0.70]])
    text = render_dc_altergroups(V)
    lines = [ln.strip() for ln in text.splitlines()]

    # paramset header, column-name row, then exactly one row per image in order
    i0 = lines.index("images paramset {")
    assert lines[i0 + 1].split() == ["vi0", "vi1"]
    rows = [[float(x) for x in lines[i0 + 2 + i].split()] for i in range(3)]
    np.testing.assert_allclose(rows, V)
    assert lines[i0 + 5] == "}"

    # a single sweep over the paramset wrapping one inner dc analysis
    assert "dc_sweep sweep paramset=images {" in lines
    inner = [ln for ln in lines if ln.startswith("dc_op dc")]
    assert len(inner) == 1 and "oppoint=rawfile" in inner[0]
    # the old per-image altergroup chain is gone
    assert "altergroup" not in text and "dc_img" not in text


def test_render_dc_altergroups_value_precision_and_shape_check():
    V = np.array([[1.234567891e-3, 0.0]])
    text = render_dc_altergroups(V)
    assert "0.0012345679 0" in text          # %.8g formatting
    try:
        render_dc_altergroups(np.zeros(3))
        raise AssertionError("1-D V must be rejected")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 3. render_testbench_dc with the top-level supports/ template
# ---------------------------------------------------------------------------
def test_render_testbench_dc(tmp_path):
    template = os.path.join(SUPPORTS, "_golden_input_dc.scs")
    assert os.path.exists(template), f"template missing: {template}"
    V = np.array([[0.1, 0.5], [0.2, 0.6]])
    src, _ = render_dc_sources_text(V, scol=2)
    alt = render_dc_altergroups(V)
    outpath = str(tmp_path / "input.scs")
    render_testbench_dc(
        template_path=template, output_path=outpath, n_images=2,
        input_sources=src, altergroups=alt,
        device_model_path="unused.scs", transistor_model="/pdk/transistor.scs",
        process_corner="tt", vdd=1.0, vss=0.0, temperature=25.0,
        iprobe_save_list=["IPRB0:in", "IPRB1:in"], tnom=31.0,
    )
    content = open(outpath).read()
    for tok in ("<input_source>", "<altergroups>", "<transistor_mod>",
                "<pcorner>", "<vdd>", "<vss>", "<temp>", "<csnames>"):
        assert tok not in content, f"unsubstituted placeholder {tok!r}"
    assert 'include "device_model.scs"' in content
    assert 'include "seed_netlist.scs"' in content
    assert 'include "/pdk/transistor.scs" section=tt' in content
    assert "VI0 (IN0 0) vsource dc=vi0 type=dc" in content
    assert "images paramset {" in content
    assert "dc_sweep sweep paramset=images {" in content
    assert "save IPRB0:in IPRB1:in" in content
    assert "tnom=31.0" in content and "tnom=27" not in content
    assert "dc=pwl(" not in content and "dc=table(" not in content


# ---------------------------------------------------------------------------
# 4. PsfDcReader on the real binary sample
# ---------------------------------------------------------------------------
def test_psf_dc_reader_real_sample():
    assert os.path.exists(SAMPLE_DC), SAMPLE_DC
    r = PsfDcReader(SAMPLE_DC)
    # samples/dc.dc is a swept DC (5-section) file: 17 points of 'tinit'
    assert r.sweep_name == "tinit"
    assert r.columns[0] == "tinit"
    assert r.matrix.shape == (17, 8) and r.n_points == 17 and r.n_traces == 7
    assert np.all(np.isfinite(r.matrix))
    np.testing.assert_allclose(r.column("tinit"),
                               np.linspace(3.3e-9, 4.9e-9, 17), rtol=1e-6)
    # BL is held at 0.1 V; the current through the cell grows with tinit
    np.testing.assert_allclose(r.column("BL"), 0.1, rtol=1e-6)
    i_cell = r.column("net2")
    assert np.all(np.diff(i_cell) > 0) and i_cell[0] > 0
    try:
        r.column("IPRB0:in")
        raise AssertionError("missing trace must raise KeyError")
    except KeyError:
        pass


def test_psf_dc_reader_single_op_roundtrip(tmp_path):
    p = tmp_path / "op.dc"
    want = {"IPRB0:in": 1.5e-6, "IPRB1:in": -2.25e-5, "BL": 0.3}
    write_single_op_psf(str(p), want)
    r = PsfDcReader(str(p))
    assert r.n_points == 1
    for k, v in want.items():
        assert r.column(k)[0] == v


# ---------------------------------------------------------------------------
# 5. _postprocess_dc_subarray reads per-iteration files
# ---------------------------------------------------------------------------
def test_postprocess_dc_subarray_parses_iteration_files(tmp_path):
    M, ncol = 4, 3
    raw_dir = tmp_path / "psf"
    raw_dir.mkdir()
    rng = np.random.default_rng(0)
    I = rng.uniform(-1e-5, 1e-5, size=(M, ncol))
    for i in range(M):
        if i == 3:
            continue                                      # missing iteration
        vals = {f"IPRB{c}:in": I[i, c] for c in range(ncol)}
        if i == 2:
            del vals["IPRB1:in"]                          # probe not saved
        vals["VDD:p"] = 1.0                               # unrelated trace
        # iteration 1 uses the non-zero-padded fallback name
        name = f"dc_sweep-{i}_dc_op.dc" if i == 1 else f"dc_sweep-{i:03d}_dc_op.dc"
        write_single_op_psf(str(raw_dir / name), vals)

    res = _postprocess_dc_subarray(_job(str(tmp_path), str(raw_dir), ncol, M))
    assert (res["sa_row"], res["sa_col"], res["branch"]) == (1, 2, "pos")
    cur = res["currents"]
    assert cur.shape == (M, ncol)
    # rows 0,1 fully parsed and finite, exactly the written values
    assert np.all(np.isfinite(cur[:2]))
    np.testing.assert_array_equal(cur[:2], I[:2])
    # row 2: only the missing probe is NaN
    np.testing.assert_array_equal(cur[2, [0, 2]], I[2, [0, 2]])
    assert np.isnan(cur[2, 1])
    # row 3: missing file -> NaN row
    assert np.all(np.isnan(cur[3]))


def test_postprocess_dc_subarray_swept_sample_and_garbage(tmp_path):
    """A swept file without IPRB traces and a corrupt file both degrade to NaN
    (no crash); an empty raw dir gives NaN of the requested shape."""
    raw_dir = tmp_path / "psf"
    raw_dir.mkdir()
    shutil.copy(SAMPLE_DC, raw_dir / "dc_sweep-000_dc_op.dc")
    (raw_dir / "dc_sweep-001_dc_op.dc").write_bytes(b"not a psf file")
    res = _postprocess_dc_subarray(_job(str(tmp_path), str(raw_dir), 4, 2))
    assert res["currents"].shape == (2, 4)
    assert np.all(np.isnan(res["currents"]))

    empty = tmp_path / "empty"
    empty.mkdir()
    res = _postprocess_dc_subarray(_job(str(tmp_path), str(empty), 16, 8))
    assert res["currents"].shape == (8, 16)
    assert np.all(np.isnan(res["currents"]))


# ---------------------------------------------------------------------------
# 6. make_backend dispatch
# ---------------------------------------------------------------------------
def test_make_backend_dispatch(tmp_path):
    from spicecimx.config import HwCfg, SimCfg, PathsCfg
    from spicecimx.sim import build_context, make_backend

    paths = PathsCfg(
        hwsimdb_root=str(tmp_path / "hwsimdb"),
        tmp_root=str(tmp_path / "tmp"),
        golden_input_transient=os.path.join(SUPPORTS, "_golden_input.scs"),
        golden_input_dc=os.path.join(SUPPORTS, "_golden_input_dc.scs"),
    )
    hw = HwCfg()
    for mode in ("transient", "dc"):
        ctx = build_context(hw, SimCfg(mode=mode, license_count=1), paths,
                            batch_id=f"t_{mode}")
        assert callable(make_backend(ctx))
    ctx_bad = build_context(hw, SimCfg(mode="nonsense", license_count=1), paths,
                            batch_id="t_bad")
    try:
        make_backend(ctx_bad)
        raise AssertionError("expected ValueError for bad mode")
    except ValueError as e:
        assert "nonsense" in str(e)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
