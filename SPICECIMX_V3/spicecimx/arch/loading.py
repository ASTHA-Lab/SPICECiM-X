"""
Shared-line loading between RSN subarrays ("loaded RSN").

The RSN decks drive every subarray from ideal sources and terminate every
column into an ideal ground.  In the tile the subarrays of one layer are
wired together with no peripherals in between (see ``arch.floorplan``):

    input bus     DAC/row driver --R_drv--o--R--o--R--o ...   (2C taps:
                  the BL driver of the +array and -array of each column-block)
    column line   TIA input --R_tia--o--R--o--R--o ...        (R taps: one
                  per row-block; the partial currents add by KCL)

Every tap current flows through the shared part of its line, so each RSN
sees a lowered input voltage ``V - d_in`` and a raised column termination
``d_out``.  Both depend on the currents of ALL subarrays of the layer, which
the RSN decks only produce one at a time — a classic domain-decomposition
coupling, resolved here by boundary relaxation:

  pass 1  RSN decks with ideal boundaries (the original flow)      -> I0
  solve   the linear line networks with a Norton model of every
          subarray port (cell small-signal conductance g = G h'(V)) -> d_in, d_out
  pass 2  RSN decks re-simulated with those boundary voltages       -> I
  check   d_check = Z I (the exact, linear line network) vs the applied d;
          if the residual exceeds tolerance, a secant update of each port
          conductance gives the next boundary for pass 3, ...

With the default buffered input bus (row drivers per subarray, Fig. 2) the
bus carries no DC current, d_in = 0, and only the column lines couple the
decks; ``input_bus = shared`` makes the bus resistive as well.

The capacitive part of the same lines is added in closed form: a
return-to-zero drive charges the wire capacitance once per vector
(E = C V^2 drawn from the supply), and the Elmore constant of each line
lengthens the MVM settle time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spicecimx.arch.floorplan import (
    InterconnectParams, LayerGrid, layer_grid,
    input_bus_segments, column_line_segments,
)


# ---------------------------------------------------------------------------
# One family of identical ladder lines
# ---------------------------------------------------------------------------
class Ladder:
    """A resistive line ``source --r_src-- (lead) tap0 --R1-- tap1 ...``.

    ``seg_ohm[0]`` is the lead from the source/sink end to tap 0 and
    ``seg_ohm[t]`` joins tap t-1 to tap t.  The deviation of every tap from
    the ideal (source) voltage is ``d = Z @ I`` with
    ``Z[t, u] = r_src + sum(seg_ohm[: min(t, u) + 1])`` (common-path
    resistance), where ``I`` are the tap currents flowing through the line.
    """

    def __init__(self, seg_ohm, r_src: float):
        self.seg = np.asarray(seg_ohm, dtype=np.float64)
        self.r_src = float(r_src)
        cum = np.cumsum(self.seg)
        T = self.seg.size
        idx = np.minimum.outer(np.arange(T), np.arange(T))
        self.Z = self.r_src + cum[idx]

    @property
    def taps(self) -> int:
        return int(self.seg.size)

    def deviation(self, I: np.ndarray) -> np.ndarray:
        """Tap deviations for tap currents ``I`` (..., T)."""
        return I @ self.Z.T

    def solve(self, I0: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Solve ``d = Z (I0 - g*d)`` for d, batched over leading axes."""
        T = self.taps
        A = np.eye(T) + self.Z[None, :, :] * g.reshape(-1, 1, T)
        rhs = I0.reshape(-1, T) @ self.Z.T
        d = np.linalg.solve(A, rhs[..., None])[..., 0]
        return d.reshape(I0.shape)

    def ir_power(self, I: np.ndarray) -> np.ndarray:
        """Power dissipated in the line + source resistance for tap currents
        ``I`` (..., T); returns (...,)."""
        through = np.cumsum(I[..., ::-1], axis=-1)[..., ::-1]   # current in seg t
        return (np.sum(through ** 2 * self.seg, axis=-1)
                + self.r_src * through[..., 0] ** 2)

    def elmore(self, c_wire_per_ohm_seg: np.ndarray, c_tap: float) -> float:
        """Elmore delay (s) from the source end to the far tap with the wire
        capacitance of each segment (F, lumped at its far end) and a tap
        capacitance at every tap."""
        c_node = np.asarray(c_wire_per_ohm_seg, dtype=np.float64) + float(c_tap)
        down = np.cumsum(c_node[::-1])[::-1]            # C beyond each segment
        return float(self.r_src * down[0] + np.sum(self.seg * down))


# ---------------------------------------------------------------------------
# A layer's loading problem
# ---------------------------------------------------------------------------
@dataclass
class LoadingResult:
    d_in:  np.ndarray                 # (m, k_pad, 2C)  input-bus drop per tap
    d_out: Dict[str, np.ndarray]      # branch -> (m, R, n_pad) column raise
    I_row: np.ndarray                 # (m, k_pad, 2C)  predicted row currents
    I_col: Dict[str, np.ndarray]      # branch -> (m, R, n_pad)
    iterations: int
    converged: bool


class LayerLoading:
    """Loading model of one weight-stationary layer.

    Parameters
    ----------
    G : dict branch -> (k_pad, n_pad) cell conductances (S) of the +/- arrays
        (the programmed linear conductances, padded to whole subarrays).
    params : resolved interconnect parameters.
    tap_c_F : capacitance of one bus tap (the BL driver input), for timing.
    """

    def __init__(self, G: Dict[str, np.ndarray], params: InterconnectParams,
                 tap_c_F: Optional[float] = None):
        self.G = {b: np.asarray(v, dtype=np.float64) for b, v in G.items()}
        self.p = params
        # buffered input bus: the DAC only charges the row drivers' inputs,
        # each subarray's driver feeds its own cells -> no input drop.
        self.shared = bool(getattr(params, "input_shared", False))
        tap_c_F = float(getattr(params, "tap_c_F", 2e-15) if tap_c_F is None else tap_c_F)
        self.tap_c_F = tap_c_F
        k_pad, n_pad = self.G["pos"].shape
        self.rs, self.cs = params.rs, params.cs
        self.R, self.C = k_pad // self.rs, n_pad // self.cs
        self.grid: LayerGrid = layer_grid(k_pad, n_pad, params)
        seg_in = np.array(input_bus_segments(self.C, params))
        seg_out = np.array(column_line_segments(self.R, params))
        self.len_in_um = float(seg_in.sum())
        self.len_out_um = float(seg_out.sum())
        self.bus = Ladder(seg_in * params.r_um, params.r_drv)
        self.col = Ladder(seg_out * params.r_um, params.r_tia)
        self.c_bus_F = self.len_in_um * params.c_um_F
        self.c_col_F = self.len_out_um * params.c_um_F
        self.tau_in = self.bus.elmore(seg_in * params.c_um_F, tap_c_F)
        self.tau_out = self.col.elmore(seg_out * params.c_um_F, 0.0)

    # -- block helpers ------------------------------------------------------
    def _rows(self, p):
        return slice(p * self.rs, (p + 1) * self.rs)

    def _cols(self, q):
        return slice(q * self.cs, (q + 1) * self.cs)

    @staticmethod
    def tap(q: int, b: str) -> int:
        return 2 * q + (0 if b == "pos" else 1)

    def _row_g(self, s):
        """(m, k, 2C) small-signal row conductance at every bus tap."""
        m, k = s.shape
        out = np.empty((m, k, 2 * self.C))
        for b in ("pos", "neg"):
            for q in range(self.C):
                out[:, :, self.tap(q, b)] = s * self.G[b][:, self._cols(q)].sum(axis=1)
        return out

    def _col_g(self, s, b):
        """(m, R, n) small-signal column conductance at every column tap."""
        m = s.shape[0]
        out = np.empty((m, self.R, self.G[b].shape[1]))
        for p in range(self.R):
            out[:, p, :] = s[:, self._rows(p)] @ self.G[b][self._rows(p), :]
        return out

    def _cross_in(self, s, d_out):
        """Row-current change caused by the column raises: (m, k, 2C)."""
        m, k = s.shape
        out = np.zeros((m, k, 2 * self.C))
        for b in ("pos", "neg"):
            for p in range(self.R):
                rows = self._rows(p)
                for q in range(self.C):
                    cols = self._cols(q)
                    out[:, rows, self.tap(q, b)] = s[:, rows] * (
                        d_out[b][:, p, cols] @ self.G[b][rows, cols].T)
        return out

    def _cross_out(self, s, d_in, b):
        """Column-current change caused by the bus drops: (m, R, n)."""
        m = s.shape[0]
        out = np.zeros((m, self.R, self.G[b].shape[1]))
        for p in range(self.R):
            rows = self._rows(p)
            for q in range(self.C):
                cols = self._cols(q)
                out[:, p, cols] = (s[:, rows] * d_in[:, rows, self.tap(q, b)]) \
                    @ self.G[b][rows, cols]
        return out

    # -- the coupled linear solve --------------------------------------------
    def solve(self, I0_row: np.ndarray, I0_col: Dict[str, np.ndarray],
              s: np.ndarray, *, g_row: Optional[np.ndarray] = None,
              g_col: Optional[Dict[str, np.ndarray]] = None,
              coupled: bool = True, max_iter: int = 60,
              tol_V: float = 1e-7) -> LoadingResult:
        """Boundary voltages for ideal-boundary currents ``I0``.

        ``s`` (m, k_pad) is h'(V) of every driven row, so the small-signal
        cell conductance is ``G * s``.  ``g_row``/``g_col`` override the
        lumped port conductances (secant update after a SPICE pass); then
        the cross terms are already inside them and ``coupled`` is False.
        """
        g_row = self._row_g(s) if g_row is None else g_row
        g_col = ({b: self._col_g(s, b) for b in ("pos", "neg")}
                 if g_col is None else g_col)
        m, k = s.shape
        d_in = np.zeros((m, k, 2 * self.C))
        d_out = {b: np.zeros_like(I0_col[b]) for b in ("pos", "neg")}
        it, conv = 0, False
        for it in range(1, max_iter + 1):
            if self.shared:
                I_row_eff = I0_row - (self._cross_in(s, d_out) if coupled else 0.0)
                d_in_new = self.bus.solve(I_row_eff, g_row)
            else:
                d_in_new = d_in
            d_out_new = {}
            for b in ("pos", "neg"):
                I_col_eff = I0_col[b] - (self._cross_out(s, d_in_new, b)
                                         if coupled else 0.0)
                # line order: tap 0 is the row-block nearest the TIA (R-1)
                ordered = I_col_eff[:, ::-1, :].transpose(0, 2, 1)
                g_ord = g_col[b][:, ::-1, :].transpose(0, 2, 1)
                d = self.col.solve(ordered, g_ord)
                d_out_new[b] = d.transpose(0, 2, 1)[:, ::-1, :]
            delta = max(float(np.max(np.abs(d_in_new - d_in), initial=0.0)),
                        max(float(np.max(np.abs(d_out_new[b] - d_out[b]), initial=0.0))
                            for b in ("pos", "neg")))
            d_in, d_out = d_in_new, d_out_new
            if delta < tol_V or not coupled:
                conv = True
                break
        I_row = I0_row - g_row * d_in - (self._cross_in(s, d_out) if coupled else 0.0)
        I_col = {b: I0_col[b] - g_col[b] * d_out[b]
                 - (self._cross_out(s, d_in, b) if coupled else 0.0)
                 for b in ("pos", "neg")}
        return LoadingResult(d_in=d_in, d_out=d_out, I_row=I_row, I_col=I_col,
                             iterations=it, converged=conv)

    # -- exact line response for measured currents ----------------------------
    def boundary_from_currents(self, I_row: np.ndarray,
                               I_col: Dict[str, np.ndarray]):
        """Exact line deviations produced by tap currents (the consistency
        check after a SPICE pass)."""
        d_in = self.bus.deviation(I_row) if self.shared else np.zeros_like(I_row)
        d_out = {}
        for b in ("pos", "neg"):
            ordered = I_col[b][:, ::-1, :].transpose(0, 2, 1)
            d = self.col.deviation(ordered)
            d_out[b] = d.transpose(0, 2, 1)[:, ::-1, :]
        return d_in, d_out

    def column_thevenin(self, d_out: Dict[str, np.ndarray],
                        I_col: Dict[str, np.ndarray]):
        """Thevenin termination of every column tap for a SPICE pass.

        Returns ``(R_self (R,), V_oc {b: (m, R, n)})``: the tap of row-block p
        sees the resistance ``Z[t, t]`` of its own path to the TIA in series
        with ``V_oc = d - Z[t, t] * I`` — the raise caused by the OTHER
        subarrays' currents ``I`` (predicted).  SPICE then resolves the
        subarray's own loading exactly at every time point (and the series
        resistance damps the trapezoidal ringing an ideal-source termination
        shows on its current)."""
        z_tt = np.diag(self.col.Z)[::-1]          # tap order -> row-block order
        V_oc = {b: d_out[b] - z_tt[None, :, None] * I_col[b] for b in d_out}
        return z_tt, V_oc

    def secant_conductances(self, I0_row, I0_col, I_row, I_col, d_in, d_out,
                            g_row_model, g_col_model):
        """Effective port conductances from two operating points (ideal
        boundary -> applied boundary).  Falls back to the model where the
        applied deviation is too small to resolve."""
        def _sec(I0, I, d, gm):
            with np.errstate(divide="ignore", invalid="ignore"):
                g = (I0 - I) / d
            ok = np.isfinite(g) & (np.abs(d) > 1e-6) & (g > 0)
            return np.where(ok, g, gm)
        g_row = _sec(I0_row, I_row, d_in, g_row_model)
        g_col = {b: _sec(I0_col[b], I_col[b], d_out[b], g_col_model[b])
                 for b in ("pos", "neg")}
        return g_row, g_col

    # -- energy / timing of the lines -----------------------------------------
    def line_energy(self, V_rows: np.ndarray, I_row: np.ndarray,
                    I_col: Dict[str, np.ndarray], d_out: Dict[str, np.ndarray],
                    t_on: float) -> Dict[str, float]:
        """Energy (J) spent in the shared lines for the whole batch.

        * ``ir_in``/``ir_out``: I^2 R in the bus (+driver) and column line
          (+TIA input) while the drive is on (``t_on`` per vector);
        * ``cv2_in``: wire capacitance of every input bus charged to its row
          voltage once per vector (return-to-zero pulse: E = C V^2);
        * ``cv2_out``: same for the column lines (swing = their raise).
        """
        ir_in = (float(np.sum(self.bus.ir_power(I_row))) * t_on) if self.shared else 0.0
        ir_out = 0.0
        cv2_out = 0.0
        for b in ("pos", "neg"):
            ordered = I_col[b][:, ::-1, :].transpose(0, 2, 1)
            ir_out += float(np.sum(self.col.ir_power(ordered))) * t_on
            cv2_out += self.c_col_F * float(np.sum(np.max(d_out[b], axis=1) ** 2))
        # buffered: the DAC charges the wire AND every driver input once per
        # vector; shared: the taps are the decks' own pass gates (in SPICE).
        c_in = self.c_bus_F + (0.0 if self.shared else 2 * self.C * self.tap_c_F)
        cv2_in = c_in * float(np.sum(V_rows ** 2))
        return {"ir_in_J": ir_in, "ir_out_J": ir_out,
                "cv2_in_J": cv2_in, "cv2_out_J": cv2_out,
                "total_J": ir_in + ir_out + cv2_in + cv2_out}


def hprime(linearizer, V: np.ndarray, dv: float = 1e-3) -> np.ndarray:
    """Slope h'(V) of the array/cell transfer (1 when no linearizer)."""
    if linearizer is None or not hasattr(linearizer, "forward"):
        return np.ones_like(V)
    up = linearizer.forward(V + dv)
    dn = linearizer.forward(np.maximum(V - dv, 0.0))
    width = (V + dv) - np.maximum(V - dv, 0.0)
    s = (up - dn) / np.where(width > 0, width, 1.0)
    return np.where(np.isfinite(s) & (s > 0), s, 1.0)


def summarize(name: str, ll: LayerLoading, res: LoadingResult,
              I0_col: Dict[str, np.ndarray], energy: Dict[str, float],
              *, passes: int, residual_V: float, m: int,
              E_deck_ideal_J: float = float("nan"),
              E_deck_loaded_J: float = float("nan")) -> Dict:
    """JSON-ready per-layer record."""
    tot0 = sum(float(np.sum(np.abs(I0_col[b]))) for b in ("pos", "neg"))
    tot1 = sum(float(np.sum(np.abs(res.I_col[b]))) for b in ("pos", "neg"))
    d_out_max = max(float(np.max(res.d_out[b], initial=0.0)) for b in ("pos", "neg"))
    return {
        "layer": name, "m": int(m), "R": ll.R, "C": ll.C,
        "tiles": ll.grid.n_tiles,
        "bus_length_um": ll.len_in_um, "col_length_um": ll.len_out_um,
        "bus_R_ohm": float(ll.bus.seg.sum()), "col_R_ohm": float(ll.col.seg.sum()),
        "d_in_max_V": float(np.max(res.d_in, initial=0.0)),
        "d_in_mean_V": float(np.mean(res.d_in)) if res.d_in.size else 0.0,
        "d_out_max_V": d_out_max,
        "current_change_rel": (tot1 - tot0) / tot0 if tot0 > 0 else 0.0,
        "tau_in_s": ll.tau_in, "tau_out_s": ll.tau_out,
        "line_energy_J": energy,
        "passes": int(passes), "residual_V": float(residual_V),
        "solver_iterations": int(res.iterations), "solver_converged": bool(res.converged),
        "E_deck_ideal_J": E_deck_ideal_J, "E_deck_loaded_J": E_deck_loaded_J,
    }
