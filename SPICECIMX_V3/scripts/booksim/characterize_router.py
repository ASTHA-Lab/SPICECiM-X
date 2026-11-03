#!/usr/bin/env python3
"""Router energy per bit from BookSim2's Orion model (32 nm techfile).

    python characterize_router.py <booksim binary> [workdir]

Prints the value used as ``arch.noc.ROUTER_PJ_PER_BIT_32NM`` (8x8 mesh,
uniform traffic, 64-bit flits, 4 VCs x 4 flits, 4-flit packets) at a few
loads, so the load sensitivity is visible too.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from spicecimx.arch.booksim import characterize_router  # noqa: E402

binary = sys.argv[1]
wd = sys.argv[2] if len(sys.argv) > 2 else "router_char"
rows = []
for load in (0.05, 0.1, 0.2, 0.3):
    r = characterize_router(binary, os.path.join(wd, f"load_{load}"), load=load)
    r.pop("raw", None)
    r["load"] = load
    rows.append(r)
    print(json.dumps(r))
