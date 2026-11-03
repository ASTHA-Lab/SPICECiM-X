"""
Deck-reading Spectre stand-in for the loaded-RSN tests.

Unlike ``fake_spectre.py`` (which is told its operands through env vars),
this one reads what the transient backend actually wrote:

  * ``input.scs``          — the PWL plateau of every ``VI<i>`` (row drive)
                             and ``VSL<j>`` (column termination) per window;
  * ``device_model.scs``   — the programmed thickness ``initial_state`` of
                             cell ``R<r>C<c>``, turned into a LINEAR
                             conductance G = alpha * exp(-(Tox-T)/(Tox-Tref)).

Each cell conducts ``G * (V_row - V_sl)``; the dump holds the column currents
``IPRB<c>:in`` and the row-source terminal currents ``VI<r>:p`` (negative
when the source delivers current, as in Spectre).

argv:  <deck> -raw <dir> [ALPHA=.. TOX=.. TREF=.. INC=..]
"""
import os
import re
import sys

import numpy as np


def _plateaus(text, name, node):
    """{node index: plateau per window} — keyed by the driven node, because
    the original flow names row sources by their GLOBAL row (VI16 drives IN0
    of row-block 1)."""
    out = {}
    for m in re.finditer(rf"^{name}\d+ \({node}(\d+) \S+\) vsource wave=\[([^\]]*)\]",
                         text, re.M):
        vals = [float(x) for x in m.group(2).split()]
        pts = np.array(vals).reshape(-1, 2)
        out[int(m.group(1))] = pts[2::5, 1]          # value at t+pw+trise
    return out


def main():
    argv = sys.argv[1:]
    kv = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("-"))
    deck = argv[0]
    raw = argv[argv.index("-raw") + 1]
    alpha, tox, tref = float(kv["ALPHA"]), float(kv["TOX"]), float(kv["TREF"])
    inc = float(kv["INC"])
    os.makedirs(raw, exist_ok=True)

    text = open(deck).read()
    vi = _plateaus(text, "VI", "IN")
    vsl = _plateaus(text, "VSL", "SLT") or _plateaus(text, "VSL", "SLX")
    rsl = {int(a): float(r) for a, r in
           re.findall(r"^RSL(\d+) \(SLT\d+ SLX\d+\) resistor r=(\S+)", text, re.M)}
    dev = open(os.path.join(os.path.dirname(deck), "device_model.scs")).read()
    cells = re.findall(r"subckt \S+_R(\d+)C(\d+) BL SL WL.*?initial_state=(\S+)", dev, re.S)
    rs = max(int(r) for r, _, _ in cells) + 1
    cs = max(int(c) for _, c, _ in cells) + 1
    G = np.zeros((rs, cs))
    for r, c, T in cells:
        G[int(r), int(c)] = alpha * np.exp(-(tox - float(T)) / (tox - tref))
    M = len(vi[0])
    V = np.stack([vi[i] for i in range(rs)], axis=1)                 # (M, rs)
    S = (np.stack([vsl[j] for j in range(cs)], axis=1) if vsl
         else np.zeros((M, cs)))                                       # (M, cs)
    if rsl:   # Thevenin termination: tap = V_oc + R * I_col (solve per column)
        Rv = np.array([rsl[j] for j in range(cs)])
        S = (S + Rv * (V @ G)) / (1.0 + Rv * G.sum(axis=0))
    cell = G[None] * (V[:, :, None] - S[:, None, :])                   # (M, rs, cs)
    I_col = cell.sum(axis=1)
    I_row = cell.sum(axis=2)

    with open(os.path.join(raw, "tran.tran.tran"), "w") as f:
        f.write("// fake psfascii\nVALUE\n")
        for w in range(M):
            for frac in (0.40, 0.52, 0.65):
                f.write(f'( "time" {(w + frac) * inc:.9e} )\n')
                for c in range(cs):
                    f.write(f'( "IPRB{c}:in" {I_col[w, c]:.12e} )\n')
                for r in range(rs):
                    f.write(f'( "VI{r}:p" {-I_row[w, r]:.12e} )\n')
    sys.exit(0)


if __name__ == "__main__":
    main()
