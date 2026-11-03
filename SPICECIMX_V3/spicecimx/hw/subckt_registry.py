r"""
Subcircuit registry for SPICECiM-X v3.

The framework treats each peripheral/array component (DAC, ADC, TIA,
SUBTRACTOR, MULTIPLEXER, CELL, WL_DECODER_DRIVER, ...) as either:

  * **software-based (S)** — modelled numerically (the current framework
    behaviour: ideal DAC/TIA/ADC math, energy from analog.pwr_energy), or
  * **hardware-based (H)** — instantiated from a real ``subckt`` defined in
    ``supports/standard_cell.netlist`` and simulated in SPICE (transient).

Which role is S vs H, and *which* subckt name implements each role, is
declared in the ``[SUBCIRCUITS]`` section of ``config.ini`` as
``role = subckt_name, mode``.  Everything else about the subckt — its exact
pin list, the order of pins, and the bus/bit width — is parsed FROM the
netlist here, never hard-coded, so the user can drop in a different cell
library (different pin names, different bit width) without touching code.

Example ``config.ini``::

    [SUBCIRCUITS]
    multiplexer       = MUX,         S
    dac               = DAC_8b,      S
    adc               = ADC_8b,      S
    tia               = TIA_SUB_8b,  S
    subtractor        = TIA_SUB_8b,  S
    cell              = rram_cell_1T1R_180N_STD, H
    wl_decoder_driver = WL_DRIVER,   H

and a matching netlist header (Spectre, ``\`` line continuation)::

    subckt DAC_8b INP<7> INP<6> INP<5> INP<4> INP<3> INP<2> INP<1> \
            INP<0> OUT VDD VSS

is parsed into a :class:`SubcktDef` with ``buses={'INP': [7,6,5,4,3,2,1,0]}``,
``scalars=['OUT','VDD','VSS']`` and ``bit_width=8``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsed subckt definition
# ---------------------------------------------------------------------------
# A bus pin in expanded form: NAME<index>.  A range form NAME<hi:lo> is also
# accepted and expanded.  Plain tokens are scalar pins.
_BUS_BIT_RE   = re.compile(r"^([A-Za-z_]\w*)<(\d+)>$")
_BUS_RANGE_RE = re.compile(r"^([A-Za-z_]\w*)<(\d+):(\d+)>$")


@dataclass
class SubcktDef:
    """A subckt parsed out of standard_cell.netlist."""
    name: str
    ports: List[str]                       # full pin list in declaration order
    buses: Dict[str, List[int]] = field(default_factory=dict)   # base -> indices (declaration order)
    scalars: List[str] = field(default_factory=list)            # non-bus pins, in order

    @property
    def bit_width(self) -> Optional[int]:
        """Widest bus = the converter resolution (e.g. DAC_8b -> 8); None
        for a subckt without bus pins."""
        return max((len(v) for v in self.buses.values()), default=None)

    def bus_width(self, base: str) -> int:
        return len(self.buses.get(base, []))

    def bus_pins(self, base: str) -> List[str]:
        """Expanded pin names for a bus, in declaration order."""
        return [f"{base}<{i}>" for i in self.buses.get(base, [])]

    def __repr__(self) -> str:
        bus = ", ".join(f"{b}<{len(idx)}b>" for b, idx in self.buses.items())
        return (f"SubcktDef(name={self.name!r}, bits={self.bit_width}, "
                f"buses=[{bus}], scalars={self.scalars}, "
                f"n_ports={len(self.ports)})")


# ---------------------------------------------------------------------------
# Netlist parsing
# ---------------------------------------------------------------------------
def _join_continuations(text: str) -> List[str]:
    r"""Join Spectre ``\`` line continuations into single logical lines."""
    out: List[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        # strip trailing comments only outside of continuation handling
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
        else:
            buf += stripped
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def _expand_port(tok: str) -> List[Tuple[str, Optional[int]]]:
    """Return [(base, index_or_None)] for one token, expanding NAME<hi:lo>."""
    m = _BUS_RANGE_RE.match(tok)
    if m:
        base, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
        step = -1 if hi >= lo else 1
        return [(base, i) for i in range(hi, lo + step, step)]
    m = _BUS_BIT_RE.match(tok)
    if m:
        return [(m.group(1), int(m.group(2)))]
    return [(tok, None)]


def parse_one_subckt(logical_line: str) -> Optional[SubcktDef]:
    r"""Parse a single ``subckt NAME port port ...`` logical line.

    Accepts the Spectre forms ``subckt NAME (p1 p2 ...)`` and Virtuoso's
    escaped bus pins ``INP\<7\>`` (read as ``INP<7>``).
    """
    line = logical_line.split("//", 1)[0].replace("\\", "")
    line = line.replace("(", " ").replace(")", " ")
    toks = line.split()
    if len(toks) < 2 or toks[0].lower() != "subckt":
        return None
    name = toks[1]
    ports: List[str] = []
    buses: Dict[str, List[int]] = {}
    scalars: List[str] = []
    for tok in toks[2:]:
        # ignore inline parameter assignments (e.g. name=value) if any appear
        if "=" in tok:
            break
        for base, idx in _expand_port(tok):
            if idx is None:
                ports.append(base)
                scalars.append(base)
            else:
                ports.append(f"{base}<{idx}>")
                buses.setdefault(base, []).append(idx)
    return SubcktDef(name=name, ports=ports, buses=buses, scalars=scalars)


def parse_subckts(netlist_path: str,
                  names: Optional[List[str]] = None) -> Dict[str, SubcktDef]:
    """
    Parse all (or the named) ``subckt`` headers from a netlist file.

    Parameters
    ----------
    netlist_path : str
        Path to standard_cell.netlist (or any Spectre netlist).
    names : list of str, optional
        If given, only these subckt names are returned.  Matching is
        case-sensitive on the declared name.

    Returns
    -------
    dict  name -> SubcktDef
    """
    if not os.path.exists(netlist_path):
        raise FileNotFoundError(f"netlist not found: {netlist_path}")
    with open(netlist_path, "r", errors="ignore") as fh:
        text = fh.read()

    want = set(names) if names else None
    out: Dict[str, SubcktDef] = {}
    for line in _join_continuations(text):
        s = line.lstrip()
        if not s.lower().startswith("subckt "):
            continue
        d = parse_one_subckt(s)
        if d is None:
            continue
        if want is None or d.name in want:
            out[d.name] = d
    return out


# ---------------------------------------------------------------------------
# Role registry (config [SUBCIRCUITS] -> resolved subckt defs)
# ---------------------------------------------------------------------------
@dataclass
class SubcktEntry:
    role: str                  # logical role: 'dac' | 'adc' | 'tia' | ...
    subckt_name: str           # subckt name as declared in standard_cell.netlist
    mode: str                  # 'S' (software model) | 'H' (hardware netlist)
    definition: Optional[SubcktDef] = None   # parsed when mode == 'H' (or found)

    @property
    def is_hardware(self) -> bool:
        return str(self.mode).strip().upper() == "H"

    @property
    def is_software(self) -> bool:
        return not self.is_hardware

    @property
    def bit_width(self) -> Optional[int]:
        return self.definition.bit_width if self.definition is not None else None


class SubcktRegistry:
    """Resolved map of role -> SubcktEntry, with netlist defs attached.

    Hardware (H) roles MUST resolve to a subckt present in the netlist; a
    missing one raises so a misconfigured physical run fails loudly rather
    than silently falling back to the software model.  Software (S) roles
    attach their def opportunistically (if the name exists) so the energy /
    bit-width can still be read from the netlist when available.
    """

    def __init__(self, entries: Dict[str, SubcktEntry]):
        self.entries = entries

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, subcircuits: Dict[str, Tuple[str, str]],
                    netlist_path) -> "SubcktRegistry":
        """Build from a ``{role: (subckt_name, mode)}`` mapping + the netlist
        (a path, or a list of paths searched in order — e.g. the std-cell
        library and the peripheral netlist)."""
        paths = [netlist_path] if isinstance(netlist_path, str) else list(netlist_path)
        netlist_path = ", ".join(paths)
        wanted_names = [v[0] for v in subcircuits.values() if v and v[0]]
        defs: Dict[str, SubcktDef] = {}
        for p in paths:
            if p and os.path.exists(p) and wanted_names:
                for k, d in parse_subckts(p, names=wanted_names).items():
                    defs.setdefault(k, d)

        entries: Dict[str, SubcktEntry] = {}
        missing_hw: List[str] = []
        for role, (sub_name, mode) in subcircuits.items():
            d = defs.get(sub_name)
            entry = SubcktEntry(role=role, subckt_name=sub_name,
                                mode=mode, definition=d)
            if entry.is_hardware and d is None:
                missing_hw.append(f"{role} -> '{sub_name}'")
            entries[role] = entry

        if missing_hw:
            raise ValueError(
                "[SUBCIRCUITS] the following hardware (H) roles name a subckt "
                f"not found in {netlist_path}: {', '.join(missing_hw)}.  "
                "Add the subckt to [PATHS] peripheral_netlist (see "
                "supports/peripherals/) or fix the name/mode.")
        return cls(entries)

    def warnings(self, cell_prefix: str = "") -> List[str]:
        """Notes about roles whose S/H setting or name does not change what
        is simulated, so a config that looks hardware-accurate isn't."""
        out: List[str] = []
        mux = self.entries.get("multiplexer")
        if mux and mux.is_hardware:
            out.append("multiplexer = H: the MUX is not simulated in SPICE (no "
                       "deck instantiates it); its energy stays the [PERIPHERALS] "
                       "model.")
        cell = self.entries.get("cell")
        if cell and cell_prefix and cell.subckt_name != cell_prefix:
            out.append(f"cell = {cell.subckt_name}: the crossbar cells are emitted "
                       f"from [PDK] cell_prefix = {cell_prefix}; this entry is "
                       "informational only.")
        if cell and cell.is_software:
            out.append("cell = S has no effect: the 1T1R cell is always simulated "
                       "(SPICE, or its SPICE-built table in fastdc).")
        drv = self.entries.get("bl_driver")
        if drv and (drv.is_software or drv.subckt_name != "BL_DRIVER"):
            out.append(f"bl_driver = {drv.subckt_name}, {drv.mode}: the subarray "
                       "always instantiates the std-cell BL_DRIVER.")
        adc_hw = self.is_hardware("adc")
        for role in ("tia", "subtractor"):
            e = self.entries.get(role)
            if not e:
                continue
            if e.is_hardware and not adc_hw:
                out.append(f"{role} = H only takes effect with adc = H (the "
                           "TIA->SUB->ADC chain is characterised as one).")
            elif adc_hw and e.is_software:
                out.append(f"adc = H uses the {e.subckt_name} subckt for {role} "
                           f"even though {role} = S (the chain needs a real "
                           "input stage).")
        return out

    # -- accessors ---------------------------------------------------------
    def get(self, role: str) -> Optional[SubcktEntry]:
        return self.entries.get(role)

    def mode(self, role: str, default: str = "S") -> str:
        e = self.entries.get(role)
        return e.mode if e else default

    def is_hardware(self, role: str) -> bool:
        e = self.entries.get(role)
        return bool(e and e.is_hardware)

    def name(self, role: str, default: str = "") -> str:
        e = self.entries.get(role)
        return e.subckt_name if e else default

    def bits(self, role: str, default: Optional[int] = None) -> Optional[int]:
        """Resolution (bus width) for a role, read from the netlist subckt.
        Falls back to ``default`` if the role isn't declared or has no def."""
        e = self.entries.get(role)
        if e and e.bit_width is not None:
            return e.bit_width
        return default

    def definition(self, role: str) -> Optional[SubcktDef]:
        e = self.entries.get(role)
        return e.definition if e else None

    def any_hardware(self) -> bool:
        return any(e.is_hardware for e in self.entries.values())

    def summary(self) -> str:
        lines = ["[SUBCIRCUITS] resolved registry:"]
        for role, e in self.entries.items():
            tag = "HW" if e.is_hardware else "sw"
            bits = f", {e.bit_width}b" if e.bit_width is not None else ""
            found = "" if e.definition is not None else "  (no netlist def)"
            lines.append(f"  {role:18s} -> {e.subckt_name:24s} [{tag}{bits}]{found}")
        return "\n".join(lines)


__all__ = [
    "SubcktDef", "SubcktEntry", "SubcktRegistry",
    "parse_subckts", "parse_one_subckt",
]