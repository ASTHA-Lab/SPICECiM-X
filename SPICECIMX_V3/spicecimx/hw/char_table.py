"""
Cell I-V characterization table for the 1T1R fastdc backend.

The table holds the SPICE-measured current ``I_cell(T, V_BL)`` of a single
1T1R cell on a regular 2-D grid, indexed by programmed filament thickness
``T`` (in metres) and bit-line voltage ``V_BL`` (in volts).  Once built
(by ``spicecimx.sim.char_run``), it is the ONLY thing the fastdc backend
needs at inference time — every per-cell current is a bilinear-interp
lookup, every bit-line current is a NumPy sum.

This is much faster than DC mode (no Spectre per MAC) and *more* accurate
than the affine de-mapping at any V_BL, because the table captures the
true nonlinear I(V,T) of the cell (transistor in series with memristor)
rather than approximating it as ``I = G·V``.

CSV format
----------
The user's reference characterization CSV has the layout::

    tinit, /NM0/S, /NM0/S, ..., /NM0/S
    3.300E-9, -29.8E-9, -32.9E-9, ..., -644E-9
    3.310E-9, -31.0E-9, -34.1E-9, ..., -668E-9
    ...
    4.900E-9, -9.78E-6, -10.8E-6, ..., -193E-6

— a first column of ``tinit`` values, then one current column per V_BL
sweep point.  The V_BL grid isn't named in the CSV; it has to be passed
in alongside (``V_min``, ``V_max``, ``V_step``).  We assume a uniform grid.

For native storage we use ``.npz`` with explicit ``T_grid``, ``V_grid``,
and ``I`` arrays — round-trips cleanly across machines and avoids any
implicit-grid ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _read_csv_chartable(
    csv_path: Path,
    V_min: float,
    V_max: float,
    V_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a SPICE characterization CSV of the format described in the
    module docstring.  Returns ``(T_grid, V_grid, I)`` with I in amperes.

    The sign of the current is preserved as in the CSV.  For the user's
    setup all currents are negative (current flows BL→ReRAM→NMOS→0, and
    the probe ``NM0:s`` measures current into the source); the fastdc
    backend takes the absolute value when summing for differential.
    """
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    T_grid = raw[:, 0].copy()           # (nT,) thicknesses in metres
    I_2d   = raw[:, 1:].copy()          # (nT, nV) currents

    nV = I_2d.shape[1]
    V_grid = np.arange(nV, dtype=np.float64) * V_step + V_min
    expected_nV = int(round((V_max - V_min) / V_step)) + 1
    if nV != expected_nV:
        # The CSV column count doesn't match the configured V grid — emit
        # a clear error rather than silently mis-aligning the V axis.
        raise ValueError(
            f"CSV has {nV} V_BL columns but the configured grid "
            f"({V_min}..{V_max} step {V_step}) implies {expected_nV} "
            f"columns.  Adjust V_min/V_max/V_step or regenerate the CSV."
        )
    return T_grid, V_grid, I_2d


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------
@dataclass
class CellITable:
    """
    Bilinear lookup table for the 1T1R cell current.

    Attributes
    ----------
    T_grid : (nT,) thicknesses in metres, strictly ascending
    V_grid : (nV,) bit-line voltages in volts, strictly ascending
    I      : (nT, nV) cell currents (sign as recorded by SPICE)
    """
    T_grid: np.ndarray
    V_grid: np.ndarray
    I:      np.ndarray

    # ---- construction --------------------------------------------------
    def __post_init__(self):
        self.T_grid = np.ascontiguousarray(self.T_grid, dtype=np.float64)
        self.V_grid = np.ascontiguousarray(self.V_grid, dtype=np.float64)
        self.I      = np.ascontiguousarray(self.I,      dtype=np.float64)
        nT, nV = self.I.shape
        if self.T_grid.shape != (nT,):
            raise ValueError(f"T_grid shape {self.T_grid.shape} != ({nT},)")
        if self.V_grid.shape != (nV,):
            raise ValueError(f"V_grid shape {self.V_grid.shape} != ({nV},)")
        if not np.all(np.diff(self.T_grid) > 0):
            raise ValueError("T_grid must be strictly ascending")
        if not np.all(np.diff(self.V_grid) > 0):
            raise ValueError("V_grid must be strictly ascending")

        # Auto-extend the V axis down to 0 with I=0.  Without this, any
        # row whose driving voltage falls below the characterised V_min
        # gets *clamped* up to V_min in the bilinear lookup, returning
        # the V_min current instead of the physically-correct ~0.
        # Summed across an inner dim of 16-64 cells that systematic
        # over-current dwarfs the legitimate signal (the IMC pipeline
        # then collapses to the head bias, which is what the "all-7
        # predictions" symptom was).
        #
        # Physics: at V_BL = 0 the access transistor's V_DS is 0, the
        # ReRAM has no driving voltage, no current flows.  Prepending
        # (V=0, I=0) lets the lookup linearly interpolate between 0 and
        # the first characterised point — correct in the small-V limit
        # where the cell is approximately linear.
        if self.V_grid[0] > 1e-12:
            self.V_grid = np.concatenate(([0.0], self.V_grid))
            self.I      = np.concatenate(
                (np.zeros((nT, 1), dtype=self.I.dtype), self.I), axis=1,
            )

    @classmethod
    def from_csv(
        cls, csv_path: str | Path,
        *, V_min: float, V_max: float, V_step: float,
    ) -> "CellITable":
        """Load from the user's flat-CSV characterization format."""
        T_grid, V_grid, I = _read_csv_chartable(
            Path(csv_path), V_min, V_max, V_step,
        )
        return cls(T_grid=T_grid, V_grid=V_grid, I=I)

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "CellITable":
        """Load from a native .npz produced by :meth:`save_npz`."""
        with np.load(str(npz_path)) as z:
            return cls(T_grid=z["T_grid"], V_grid=z["V_grid"], I=z["I"])

    @classmethod
    def from_array_table(cls, array_tab, N: int) -> "CellITable":
        """Derive a per-cell I(T,V) table from an :class:`ArrayCharTable`.

        Lets fastdc run from ``array_table.npz`` alone (no separate cell CSV)
        when ``[SIM] use_cell_table = false``.

        Physics: the array characterisation measures the per-**column** current
        ``I_col(T, V)`` at uniform thickness for an ``N``-row array.  At uniform
        T the column is the sum of ``N`` identical cells, so the per-cell current
        is ``I_col / N``.  Two corrections make it a drop-in for the cell CSV:

        * **Sign** — the array IPRB sinks into VSS with the opposite sign to the
          single-cell BL probe, so we negate to match the cell I-V convention the
          fastdc lookup expects (cell currents are negative for positive V).
        * **Loading** — ``I_col / N`` is the *loaded* per-cell current (smaller
          than the ideal single cell, increasingly so at high V).  That loading
          is exactly what the array ``h(V)`` linerizer — built from the *same*
          ``I_col`` — inverts, so applying that linearizer to these derived
          currents linearises them without double-counting (it is in fact
          self-consistent: it linearises essentially exactly at ``T_ref``).
        """
        N = int(N)
        if N <= 0:
            raise ValueError(f"from_array_table needs N >= 1, got {N}")
        I_col = np.asarray(array_tab.I_at(N), dtype=np.float64)   # (nT, nV)
        I_cell = -I_col / float(N)
        return cls(
            T_grid=np.asarray(array_tab.T_grid, dtype=np.float64).copy(),
            V_grid=np.asarray(array_tab.V_grid, dtype=np.float64).copy(),
            I=I_cell,
        )

    @classmethod
    def load(cls, path: str | Path, **csv_kwargs) -> "CellITable":
        """
        Polymorphic load: chooses npz vs csv by extension.

        Parameters
        ----------
        path : path to .npz or .csv
        **csv_kwargs : forwarded to ``from_csv`` (V_min, V_max, V_step)
            when ``path`` ends in .csv.
        """
        p = Path(path)
        if p.suffix.lower() == ".npz":
            return cls.from_npz(p)
        if p.suffix.lower() == ".csv":
            return cls.from_csv(p, **csv_kwargs)
        raise ValueError(f"Unknown char-table extension: {p.suffix!r}")

    def save_npz(self, npz_path: str | Path) -> None:
        """Save in native compressed npz format."""
        Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(npz_path),
            T_grid=self.T_grid, V_grid=self.V_grid, I=self.I,
        )

    # ---- properties ----------------------------------------------------
    @property
    def nT(self) -> int: return int(self.I.shape[0])
    @property
    def nV(self) -> int: return int(self.I.shape[1])
    @property
    def T_min(self) -> float: return float(self.T_grid[0])
    @property
    def T_max(self) -> float: return float(self.T_grid[-1])
    @property
    def V_min(self) -> float: return float(self.V_grid[0])
    @property
    def V_max(self) -> float: return float(self.V_grid[-1])

    # ---- core interpolation -------------------------------------------
    def _index_and_frac(
        self, x: np.ndarray, grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each value in ``x``, return (lower-index, fractional position)
        for linear interpolation on the (possibly non-uniform) ``grid``.

        Out-of-range values are clamped to the grid edges with zero
        fractional position so the lookup degrades gracefully (instead of
        extrapolating).  This matters for cells programmed exactly at
        ``T_min`` (HRS) or for the rare V_BL=0 query.
        """
        x_clip = np.clip(x, grid[0], grid[-1])
        # searchsorted with 'right' gives the index of the first grid
        # point strictly greater than x; subtract one to get the lower.
        idx = np.searchsorted(grid, x_clip, side="right") - 1
        idx = np.clip(idx, 0, len(grid) - 2)
        x_lo  = grid[idx]
        x_hi  = grid[idx + 1]
        frac  = (x_clip - x_lo) / (x_hi - x_lo)
        return idx, frac

    def lookup_scalar(self, T: float, V: float) -> float:
        """
        Bilinear lookup of a single ``I(T, V)``.  Handy for tests and
        debugging; use :meth:`bitline_current` for crossbar computation.
        """
        iT, fT = self._index_and_frac(
            np.asarray([T]), self.T_grid,
        )
        iV, fV = self._index_and_frac(
            np.asarray([V]), self.V_grid,
        )
        iT = int(iT[0]); iV = int(iV[0])
        fT = float(fT[0]); fV = float(fV[0])
        I00 = self.I[iT,     iV]
        I10 = self.I[iT + 1, iV]
        I01 = self.I[iT,     iV + 1]
        I11 = self.I[iT + 1, iV + 1]
        return float(
            I00 * (1.0 - fT) * (1.0 - fV)
          + I10 * fT         * (1.0 - fV)
          + I01 * (1.0 - fT) * fV
          + I11 * fT         * fV
        )

    # ---- voltage linearization ----------------------------------------
    # The cell I-V is highly nonlinear in V_BL (G_eff doubles from V=0.1V to
    # V=1.0V for this ReRAM technology).  Without correction the affine
    # demap produces 10-20% per-element error per layer — fatal for multi-
    # block ViTs where errors compound through softmax/GELU nonlinearities.
    #
    # The fix: pre-warp the driving voltage so the cell "sees" a linearised
    # voltage.  Empirically the V-nonlinearity is approximately T-independent
    # (ratio G_eff@V_max / G_eff@V→0 varies by <10% across the T range), so
    # a single 1-D h(V) curve from the table is sufficient.  Residual error
    # after linearisation is ~5% (the T-dependence of h), down from ~100%
    # raw nonlinearity.
    #
    # Definition: h(V) := I(T_ref, V) / G_lin(T_ref)
    #   - G_lin = the V→0 slope (matches the bare-ReRAM Verilog-A model
    #     used by ``convert_to_conductance`` to compute per-cell T from B),
    #     estimated as I(T_ref, V_low) / V_low for the smallest V_low > 0
    #     in the table.
    #   - h(0) = 0, h is convex with h'(0) = 1 (for a super-linear cell)
    #   - h(V_max) > V_max because the cell is super-linear
    #   - h_inverse(V_target) ∈ [0, V_max] for V_target ∈ [0, V_max] because
    #     the V_max-of-A image is below V_max (the cell at h_inv(V_max)
    #     produces current G_lin · V_max which is below G_chord(V_max)·V_max
    #     = the cell's current at the actual V_max).
    #
    # Why this normalisation works: after linearisation, every cell of
    # thickness T produces current G_lin(T) · V_target.  Summing across
    # cells, the BL current is sum_i G_lin(T_i)·V_target_i, which is what
    # the affine demap is derived assuming.  c = (Gmax_lin - Gmin_lin) /
    # (Bmax - Bmin) (already what derive_affine uses), so the recovery
    # C = (I_diff - c·b·J@B) / (c·a) is exact (modulo the T-dependence
    # of h, ~5% residual).
    def build_v_linearization(
        self, T_ref: Optional[float] = None,
    ) -> "VLinearizer":
        """
        Build a 1-D voltage linearisation curve from this table.

        Parameters
        ----------
        T_ref : reference thickness in metres.  If None, uses the
                arithmetic midpoint of the T range.

        Returns
        -------
        VLinearizer with ``forward(V)`` = h(V) and ``inverse(V)`` = h⁻¹(V).
        """
        if T_ref is None:
            T_ref = 0.5 * (self.T_min + self.T_max)

        iT, fT = self._index_and_frac(np.asarray([T_ref]), self.T_grid)
        iT = int(iT[0]); fT = float(fT[0])
        I_T = self.I[iT] * (1.0 - fT) + self.I[iT + 1] * fT  # (nV,)

        # G_lin estimated from the smallest V>0 in the table (closest to
        # the V→0 slope; matches the Verilog-A linearised conductance).
        V_grid = self.V_grid
        if V_grid[0] == 0.0:
            V_low = V_grid[1]; I_low = abs(I_T[1])
        else:
            V_low = V_grid[0]; I_low = abs(I_T[0])
        G_lin = I_low / V_low if V_low > 0 else 0.0
        if G_lin == 0.0:
            # Pathological: T_ref is in deep HRS where I = 0 everywhere.
            # Fall back to identity — no linearisation applied.
            h = V_grid.copy()
        else:
            h = np.abs(I_T) / G_lin                  # h(V) ≥ V for super-linear cell
            h[0] = 0.0                                # force h(0)=0 exactly
            h = np.maximum.accumulate(h)              # guard monotonicity
        return VLinearizer(V_grid=V_grid, h=h)

    # ---- vectorised crossbar lookup -----------------------------------
    def bitline_current(
        self,
        T_cells: np.ndarray,        # (k, n)  programmed thicknesses
        V_rows:  np.ndarray,        # (m, k)  bit-line driving voltages
    ) -> np.ndarray:
        """
        Compute the bit-line currents for an entire subarray-branch by
        bilinear-interp-and-sum over the cell I-V table.

        For each cell (i, j) and each image (r):
            I_cell[r, i, j] = bilinear_interp(I, T_cells[i, j], V_rows[r, i])

        and then::

            I_BL[r, j] = sum_i I_cell[r, i, j]

        is returned.  The output preserves the sign of the table (so for
        the user's ``NM0:s`` probe convention currents are negative when
        V_BL > 0); the caller decides whether to take abs() based on
        crossbar polarity.

        Memory footprint: m*k*n floats temporarily (the per-cell current
        cube).  For (m=160, k=64, n=32) that's 2.5 MB — trivial.  For
        larger m we chunk along m to keep the working set bounded.
        """
        m, k = V_rows.shape
        k2, n = T_cells.shape
        if k != k2:
            raise ValueError(
                f"inner dim mismatch: V_rows has k={k}, T_cells has k={k2}"
            )

        iT, fT = self._index_and_frac(T_cells.ravel(), self.T_grid)
        iT = iT.reshape(k, n)
        fT = fT.reshape(k, n)

        bytes_per_elem = 8
        max_elems = (32 << 20) // bytes_per_elem
        chunk_m   = max(1, max_elems // (k * n))

        I_BL = np.empty((m, n), dtype=np.float64)
        for r0 in range(0, m, chunk_m):
            r1 = min(r0 + chunk_m, m)
            mb = r1 - r0
            iV, fV = self._index_and_frac(
                V_rows[r0:r1].ravel(), self.V_grid,
            )
            iV = iV.reshape(mb, k)
            fV = fV.reshape(mb, k)

            iT_b = iT[np.newaxis, :, :]
            fT_b = fT[np.newaxis, :, :]
            iV_b = iV[:, :, np.newaxis]
            fV_b = fV[:, :, np.newaxis]

            I00 = self.I[iT_b,     iV_b    ]
            I10 = self.I[iT_b + 1, iV_b    ]
            I01 = self.I[iT_b,     iV_b + 1]
            I11 = self.I[iT_b + 1, iV_b + 1]

            wT0 = 1.0 - fT_b; wT1 = fT_b
            wV0 = 1.0 - fV_b; wV1 = fV_b

            I_cell = (I00 * wT0 * wV0
                    + I10 * wT1 * wV0
                    + I01 * wT0 * wV1
                    + I11 * wT1 * wV1)
            I_BL[r0:r1] = I_cell.sum(axis=1)

        return I_BL


@dataclass
class VLinearizer:
    """
    1-D voltage pre-warp: applies ``V_phys = h⁻¹(V_target)`` so that the
    cell at the reference thickness produces a current proportional to
    ``V_target`` instead of to the nonlinear ``h(V_target)``.

    Constructed by :meth:`CellITable.build_v_linearization`.  Vectorised
    over any input shape.
    """
    V_grid: np.ndarray         # (nV,) original V grid
    h:      np.ndarray         # (nV,) values of h(V) at each V_grid point

    def __post_init__(self):
        self.V_grid = np.ascontiguousarray(self.V_grid, dtype=np.float64)
        self.h      = np.ascontiguousarray(self.h,      dtype=np.float64)

    def forward(self, V: np.ndarray) -> np.ndarray:
        """Compute h(V) via piecewise-linear interpolation."""
        return np.interp(V, self.V_grid, self.h)

    def inverse(self, V_target: np.ndarray) -> np.ndarray:
        """
        Compute h⁻¹(V_target) — the physical voltage to apply so that the
        cell produces current proportional to V_target.

        Uses numpy's interp with x=h, y=V_grid (valid because h is
        monotonic ascending).
        """
        return np.interp(V_target, self.h, self.V_grid)