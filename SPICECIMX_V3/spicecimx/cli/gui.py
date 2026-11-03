# -*- coding: utf-8 -*-
"""
================================================================================
  gui.py — SPICECiM-X configuration & run GUI
================================================================================

Launched with ``python -m spicecimx.cli.infer --gui`` (or
``python -m spicecimx.cli.gui``).  Features:

  * tabbed editor for **every** section/key in ``config.ini`` (dropdowns for
    enums, **checkboxes** for true/false, folder/file pickers for paths),
    comments preserved on save;
  * **night** (default, dark-gray) and **light** themes, toggled live;
  * larger, consistent fonts and an adaptive, resizable, scrollable layout;
  * real PNG **icons** on the buttons (loaded from ``assets/icons`` next to this
    file — no Unicode glyphs, no PIL needed at run time; graceful text fallback);
  * an "extra CLI args" box for per-run overrides (e.g. ``--mode auto --limit 5``);
  * **Run** launches inference in a subprocess and streams its output live into a
    status pane (with **Stop**); on success the **report window** pops up.

The config read/edit/save helpers are pure functions (no Tk) so they're testable
headless; the Tk UI is only built inside ``launch_gui``.
================================================================================
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from collections import OrderedDict
from configparser import ConfigParser
from typing import Dict, List, Optional, Tuple


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Config read / edit / save  (pure, Tk-free, testable)                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def parse_config(path: str) -> "OrderedDict[str, OrderedDict[str, str]]":
    """Return {section: {key: value}} preserving order + case.  Same parser
    settings as ``spicecimx.config.make_parser``: inline ``#``/``;`` comments
    are stripped and ``%`` is literal."""
    cp = ConfigParser(inline_comment_prefixes=("#", ";"), interpolation=None)
    cp.optionxform = str
    cp.read(path)
    out: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
    for sec in cp.sections():
        out[sec] = OrderedDict((k, v) for k, v in cp.items(sec))
    return out


def set_config_value(raw: str, section: str, key: str, value: str) -> str:
    """Replace ``key = value`` within ``[section]`` in the raw .ini text,
    preserving comments/layout.  Appends the key at the end of the section if
    it isn't present.  Returns the new text."""
    lines = raw.splitlines(keepends=True)
    sec_start = sec_end = None
    sec_hdr = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*$")
    for i, ln in enumerate(lines):
        m = sec_hdr.match(ln)
        if m:
            if m.group("name") == section and sec_start is None:
                sec_start = i
            elif sec_start is not None and sec_end is None:
                sec_end = i
                break
    if sec_start is None:                       # section missing -> append it
        nl = "" if raw.endswith("\n") or raw == "" else "\n"
        return raw + f"{nl}\n[{section}]\n{key} = {value}\n"
    if sec_end is None:
        sec_end = len(lines)

    key_re = re.compile(rf"^(\s*){re.escape(key)}(\s*)=(.*)$")
    for i in range(sec_start + 1, sec_end):
        m = key_re.match(lines[i].rstrip("\r\n"))
        if m:
            indent = m.group(1)
            # keep an inline comment (parse_config strips it from the value)
            c = re.search(r"\s[#;].*$", m.group(3))
            comment = c.group(0) if c else ""
            eol = "\n" if lines[i].endswith("\n") else ""
            lines[i] = f"{indent}{key} = {value}{comment}{eol}"
            return "".join(lines)
    insert_at = sec_end
    while insert_at - 1 > sec_start and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, f"{key} = {value}\n")
    return "".join(lines)


def apply_edits(raw: str, edits: Dict[Tuple[str, str], str]) -> str:
    for (sec, key), val in edits.items():
        raw = set_config_value(raw, sec, key, val)
    return raw


def project_dir_for(cfg: "OrderedDict[str, OrderedDict[str, str]]",
                    config_dir: str = ".") -> str:
    """Absolute project_dir, resolved like ``config.resolve_paths``: a relative
    project_dir is relative to the config file's directory."""
    pd = os.path.expanduser(cfg.get("PATHS", {}).get("project_dir", ".") or ".")
    if not os.path.isabs(pd):
        pd = os.path.join(config_dir, pd)
    return os.path.normpath(os.path.abspath(pd))


def dump_dir_for(cfg: "OrderedDict[str, OrderedDict[str, str]]",
                 config_dir: str = ".",
                 extra_args: Optional[List[str]] = None) -> str:
    """Compute the dump_root/batch_id dir the run will write to (where infer
    puts Final_report.txt, accuracy_report.json and the energy report).

    Mirrors ``config.resolve_paths``: project_dir is relative to the config
    file's directory, dump_root/images_dir relative to project_dir.  The
    ``--project/--images_dir/--dump_root`` overrides in ``extra_args`` are
    relative to the run's cwd, which the GUI sets to ``config_dir``."""
    paths = cfg.get("PATHS", {})
    dump_root = os.path.expanduser(paths.get("dump_root", "imc_traces") or "imc_traces")
    images_dir = os.path.expanduser(paths.get("images_dir", "./Test_Image") or "./Test_Image")
    ov: Dict[str, str] = {}
    toks = list(extra_args or [])
    for i, tok in enumerate(toks):
        for k in ("--project", "--images_dir", "--dump_root"):
            if tok == k and i + 1 < len(toks):
                ov[k] = toks[i + 1]
            elif tok.startswith(k + "="):
                ov[k] = tok.split("=", 1)[1]
    if "--project" in ov:
        project = os.path.join(config_dir, os.path.expanduser(ov["--project"]))
    else:
        project = project_dir_for(cfg, config_dir)
    if "--dump_root" in ov:
        dump_root = os.path.join(config_dir, os.path.expanduser(ov["--dump_root"]))
    elif not os.path.isabs(dump_root):
        dump_root = os.path.join(project, dump_root)
    if "--images_dir" in ov:
        images_dir = ov["--images_dir"]
    batch = os.path.basename(os.path.normpath(images_dir)) or "default"
    return os.path.normpath(os.path.abspath(os.path.join(dump_root, batch)))


# ── per-key UI hints (enums / booleans / file pickers) ──────────────────────
_ENUMS = {
    ("SIM", "mode"): ["dc", "fastdc", "transient", "hybrid", "auto"],
    # parse_transient._reduce supports median|mean|last (raises on anything else)
    ("SIM", "settling_reducer"): ["median", "mean", "last"],
    ("SIM", "dc_input_encoding"): ["sweep", "per_image"],
    ("PERIPHERALS", "adc_energy_model"): ["walden", "accelergy"],
    ("LATENCY", "adc_type"): ["sar", "flash", "pipeline", "integrating", "sigma_delta"],
    # only 'classification' is registered in core.metrics.TASK_REGISTRY today
    ("ACCURACY", "task"): ["classification"],
    ("DATA", "channels"): ["auto", "1", "3"],
}
_BOOLS = {"true", "false"}
_DIR_KEYS = {"images_dir", "data_root", "out_dir", "dump_root", "hwsimdb_root",
             "tmp_root", "project_dir"}
_FILE_KEYS = {"golden_input_transient", "golden_input_dc", "stdcell_netlist",
              "device_model_transient", "device_model_dc", "transistor_model",
              "char_table_path", "array_table_path", "labels_file"}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Themes                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝
THEMES = {
    "night": dict(
        bg="#23272e", panel="#2b3039", fg="#e7ebf2", muted="#9aa4b3",
        accent="#4f8cff", accent_fg="#ffffff", border="#3a4150",
        entry_bg="#1c2026", entry_fg="#e7ebf2", select="#34507a",
        tab_bg="#2b3039", tab_active="#3a4150", status_bg="#14171c",
        status_fg="#cfd8e3", heading="#ffffff",
    ),
    "light": dict(
        bg="#eef0f4", panel="#ffffff", fg="#1f2430", muted="#5b6472",
        accent="#2563eb", accent_fg="#ffffff", border="#cfd4dc",
        entry_bg="#ffffff", entry_fg="#1f2430", select="#cfe0ff",
        tab_bg="#e4e7ec", tab_active="#ffffff", status_bg="#0f1318",
        status_fg="#cfd8e3", heading="#11151c",
    ),
}


def _icon_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icons")


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Tk GUI                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def launch_gui(config_path: str) -> int:
    try:
        import tkinter as tk
        from tkinter import ttk, font as tkfont, filedialog, scrolledtext, messagebox
    except Exception as e:                       # pragma: no cover
        print(f"[gui] tkinter unavailable ({e}). Install python3-tk or run "
              f"without --gui.", file=sys.stderr)
        return 2

    config_path = os.path.abspath(config_path)
    # infer resolves project_dir against the config file's own directory, so
    # runs are launched from there and the report paths are resolved from it.
    config_dir = os.path.dirname(config_path)
    raw_holder = {"raw": open(config_path).read() if os.path.exists(config_path) else ""}
    cfg = parse_config(config_path)

    root = tk.Tk()
    root.title(f"SPICECiM-X  -  {os.path.basename(config_path)}")
    root.geometry("1100x820")
    root.minsize(820, 560)
    try:
        root.tk.call("tk", "scaling", float(os.environ.get("SPICECIM_GUI_SCALE", "1.3")))
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")                  # clam honours custom colours
    except Exception:
        pass

    # ── fonts (larger than Tk defaults) ──
    BASE = int(os.environ.get("SPICECIM_GUI_FONT", "12"))
    f_base = tkfont.Font(size=BASE)
    f_bold = tkfont.Font(size=BASE, weight="bold")
    f_head = tkfont.Font(size=BASE + 6, weight="bold")
    f_mono = tkfont.Font(size=BASE - 1)
    f_tab = tkfont.Font(size=BASE - 1)
    fams = set(tkfont.families())
    for fam in ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Helvetica", "Arial"):
        if fam in fams:
            f_base.config(family=fam); f_bold.config(family=fam)
            f_head.config(family=fam); f_tab.config(family=fam); break
    for fam in ("Consolas", "DejaVu Sans Mono", "Menlo", "Courier New"):
        if fam in fams:
            f_mono.config(family=fam); break

    # ── icons (PNG via stdlib PhotoImage; text fallback if missing) ──
    icons: Dict[str, Optional["tk.PhotoImage"]] = {}
    for name in ("run", "stop", "save", "help", "report", "clear",
                 "folder", "folder_lg", "sun", "moon", "app"):
        p = os.path.join(_icon_dir(), f"{name}.png")
        try:
            icons[name] = tk.PhotoImage(file=p) if os.path.exists(p) else None
        except Exception:
            icons[name] = None
    if icons.get("app"):
        try:
            root.iconphoto(True, icons["app"])
        except Exception:
            pass

    state = {"theme": "light" if os.environ.get("SPICECIM_GUI_THEME", "").lower() == "light" else "night"}
    themed_text: List["tk.Text"] = []
    themed_canvas: List["tk.Canvas"] = []

    # ── theme application ──
    def apply_theme():
        t = THEMES[state["theme"]]
        root.configure(bg=t["bg"])
        style.configure(".", background=t["bg"], foreground=t["fg"],
                        fieldbackground=t["entry_bg"], font=f_base)
        style.configure("TFrame", background=t["bg"])
        style.configure("Panel.TFrame", background=t["panel"])
        style.configure("TLabel", background=t["bg"], foreground=t["fg"], font=f_base)
        style.configure("Head.TLabel", background=t["bg"], foreground=t["heading"], font=f_head)
        style.configure("Muted.TLabel", background=t["bg"], foreground=t["muted"], font=f_base)
        style.configure("Field.TLabel", background=t["panel"], foreground=t["fg"], font=f_base)
        style.configure("TButton", background=t["panel"], foreground=t["fg"],
                        font=f_base, padding=(10, 6), borderwidth=1)
        style.map("TButton", background=[("active", t["tab_active"]), ("pressed", t["border"])])
        style.configure("Accent.TButton", background=t["accent"], foreground=t["accent_fg"],
                        font=f_bold, padding=(12, 6))
        style.map("Accent.TButton", background=[("active", t["accent"]), ("pressed", t["accent"])])
        style.configure("Icon.TButton", background=t["bg"], padding=4, borderwidth=0)
        style.map("Icon.TButton", background=[("active", t["tab_active"])])
        style.configure("TEntry", fieldbackground=t["entry_bg"], foreground=t["entry_fg"],
                        insertcolor=t["fg"], bordercolor=t["border"], padding=4)
        style.configure("TCombobox", fieldbackground=t["entry_bg"], foreground=t["entry_fg"],
                        background=t["panel"], arrowcolor=t["fg"], padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", t["entry_bg"])],
                  foreground=[("readonly", t["entry_fg"])])
        root.option_add("*TCombobox*Listbox.background", t["entry_bg"])
        root.option_add("*TCombobox*Listbox.foreground", t["entry_fg"])
        root.option_add("*TCombobox*Listbox.selectBackground", t["select"])
        root.option_add("*TCombobox*Listbox.font", f_base)
        style.configure("TCheckbutton", background=t["panel"], foreground=t["fg"], font=f_base)
        style.map("TCheckbutton", background=[("active", t["panel"])])
        style.configure("Seg.Toolbutton", background=t["entry_bg"], foreground=t["muted"],
                        font=f_bold, padding=(10, 5), borderwidth=1, relief="flat", anchor="center")
        style.map("Seg.Toolbutton",
                  background=[("selected", t["accent"]), ("active", t["tab_active"])],
                  foreground=[("selected", t["accent_fg"]), ("active", t["fg"])])
        style.configure("TNotebook", background=t["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=t["tab_bg"], foreground=t["muted"],
                        font=f_tab, padding=(8, 7))
        style.map("TNotebook.Tab", background=[("selected", t["panel"])],
                  foreground=[("selected", t["accent"])])
        style.configure("Vertical.TScrollbar", background=t["panel"],
                        troughcolor=t["bg"], arrowcolor=t["fg"], bordercolor=t["bg"])
        for txt in themed_text:
            txt.configure(bg=t["status_bg"], fg=t["status_fg"], insertbackground=t["status_fg"])
        for cv in themed_canvas:
            cv.configure(bg=t["panel"])
        if icons.get("sun") or icons.get("moon"):
            theme_btn.configure(image=icons.get("sun") if state["theme"] == "night"
                                else icons.get("moon"))
        else:
            theme_btn.configure(text=("Light mode" if state["theme"] == "night" else "Night mode"))

    def toggle_theme():
        state["theme"] = "light" if state["theme"] == "night" else "night"
        apply_theme()

    # ── header ──
    header = ttk.Frame(root, padding=(14, 10)); header.pack(fill="x")
    if icons.get("app"):
        ttk.Label(header, image=icons["app"]).pack(side="left", padx=(0, 10))
    titlebox = ttk.Frame(header); titlebox.pack(side="left")
    ttk.Label(titlebox, text="SPICECiM-X", style="Head.TLabel").pack(anchor="w")
    ttk.Label(titlebox, text=config_path, style="Muted.TLabel").pack(anchor="w")
    theme_btn = ttk.Button(header, style="Icon.TButton", command=toggle_theme,
                           image=icons.get("sun"),
                           text=("" if icons.get("sun") else "Light mode"), compound="left")
    theme_btn.pack(side="right")
    ttk.Label(header, text="extra CLI args:", style="Muted.TLabel").pack(side="right", padx=(0, 6))
    extra_var = tk.StringVar(value="")
    ttk.Entry(header, textvariable=extra_var, width=28, font=f_base).pack(side="right", padx=(0, 10))

    # ── main paned area: notebook (top) + status (bottom) ──
    paned = ttk.Panedwindow(root, orient="vertical")
    paned.pack(fill="both", expand=True, padx=12, pady=(0, 6))
    nb = ttk.Notebook(paned); paned.add(nb, weight=3)

    vars_map: Dict[Tuple[str, str], "tk.Variable"] = {}
    bool_keys: set = set()
    subckt_map: Dict[Tuple[str, str], Tuple["tk.StringVar", "tk.StringVar"]] = {}

    def _add_section_tab(sec: str, items: "OrderedDict[str, str]"):
        outer = ttk.Frame(nb, style="Panel.TFrame"); nb.add(outer, text=sec)
        canvas = tk.Canvas(outer, highlightthickness=0, bd=0)
        themed_canvas.append(canvas)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="Panel.TFrame")
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True); sb.pack(side="right", fill="y")

        def _wheel(e):
            delta = (-1 * (e.delta // 120)) if getattr(e, "delta", 0) else (-1 if e.num == 4 else 1)
            canvas.yview_scroll(delta, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            inner.bind(seq, _wheel); canvas.bind(seq, _wheel)

        inner.columnconfigure(1, weight=1)
        for r, (key, val) in enumerate(items.items()):
            ttk.Label(inner, text=key, style="Field.TLabel", anchor="e", width=26)\
                .grid(row=r, column=0, sticky="e", padx=(10, 8), pady=4)
            enum = _ENUMS.get((sec, key))
            is_bool = (enum is None) and (val.strip().lower() in _BOOLS)
            # [SUBCIRCUITS]: "<subckt_name>, <S|H>" -> name box + S/H toggle
            m_sub = (sec == "SUBCIRCUITS") and re.match(r"^\s*([^,]+?)\s*,\s*([SHsh])\s*$", val)
            if m_sub:
                name0, mode0 = m_sub.group(1), m_sub.group(2).upper()
                cell = ttk.Frame(inner, style="Panel.TFrame")
                cell.grid(row=r, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
                cell.columnconfigure(0, weight=1)
                name_var = tk.StringVar(value=name0)
                ttk.Entry(cell, textvariable=name_var, font=f_base)\
                    .grid(row=0, column=0, sticky="ew", padx=(0, 10))
                mode_var = tk.StringVar(value=mode0)
                seg = ttk.Frame(cell, style="Panel.TFrame"); seg.grid(row=0, column=1, sticky="e")
                ttk.Radiobutton(seg, text="S", value="S", variable=mode_var,
                                style="Seg.Toolbutton", width=3)\
                    .grid(row=0, column=0)
                ttk.Radiobutton(seg, text="H", value="H", variable=mode_var,
                                style="Seg.Toolbutton", width=3)\
                    .grid(row=0, column=1)
                ttk.Label(seg, text="  (S=software · H=hardware)", style="Muted.TLabel")\
                    .grid(row=0, column=2, padx=(8, 0))
                subckt_map[(sec, key)] = (name_var, mode_var)
                continue
            if is_bool:
                bv = tk.BooleanVar(value=(val.strip().lower() == "true"))
                vars_map[(sec, key)] = bv; bool_keys.add((sec, key))
                ttk.Checkbutton(inner, variable=bv, style="TCheckbutton")\
                    .grid(row=r, column=1, sticky="w", padx=4, pady=4)
            elif enum is not None:
                sv = tk.StringVar(value=val); vars_map[(sec, key)] = sv
                ttk.Combobox(inner, textvariable=sv, values=enum, state="readonly",
                             font=f_base, width=28)\
                    .grid(row=r, column=1, sticky="w", padx=4, pady=4)
            else:
                sv = tk.StringVar(value=val); vars_map[(sec, key)] = sv
                ttk.Entry(inner, textvariable=sv, font=f_base)\
                    .grid(row=r, column=1, sticky="ew", padx=4, pady=4)
                if key in _DIR_KEYS or key in _FILE_KEYS:
                    def _browse(v=sv, k=key):
                        p = (filedialog.askdirectory() if k in _DIR_KEYS
                             else filedialog.askopenfilename())
                        if p:
                            v.set(p)
                    ttk.Button(inner, style="Icon.TButton", image=icons.get("folder"),
                               text=("" if icons.get("folder") else "Browse"), command=_browse)\
                        .grid(row=r, column=2, sticky="w", padx=(2, 8), pady=4)

    for sec, items in cfg.items():
        _add_section_tab(sec, items)

    # ── status pane ──
    status_box = ttk.Frame(paned, style="Panel.TFrame"); paned.add(status_box, weight=1)
    ttk.Label(status_box, text="status", style="Muted.TLabel").pack(anchor="w", padx=4, pady=(4, 0))
    status = scrolledtext.ScrolledText(status_box, height=10, font=f_mono, relief="flat", bd=0)
    themed_text.append(status)
    status.pack(fill="both", expand=True, padx=4, pady=(2, 4))

    def log(msg: str):
        # Tk main thread only; worker threads go through log_q.
        status.insert("end", msg); status.see("end")

    # Tk is not thread-safe: the subprocess reader thread only puts text (or a
    # callable to run on the Tk thread) on this queue; drain_log() empties it
    # from the main loop via root.after polling.
    log_q: "queue.Queue" = queue.Queue()

    def drain_log():
        try:
            while True:
                item = log_q.get_nowait()
                if callable(item):
                    item()
                else:
                    log(item)
        except queue.Empty:
            pass
        root.after(50, drain_log)

    # ── actions ──
    proc_holder: Dict[str, Optional[subprocess.Popen]] = {"p": None}

    def collect_edits() -> Dict[Tuple[str, str], str]:
        edits = {}
        for (sec, key), var in vars_map.items():
            new = ("true" if var.get() else "false") if (sec, key) in bool_keys else var.get()
            if new != cfg.get(sec, {}).get(key, None):
                edits[(sec, key)] = new
        for (sec, key), (name_var, mode_var) in subckt_map.items():
            new = f"{name_var.get().strip()}, {mode_var.get()}"
            if new != cfg.get(sec, {}).get(key, None):
                edits[(sec, key)] = new
        return edits

    def do_save() -> bool:
        edits = collect_edits()
        raw = apply_edits(raw_holder["raw"], edits)
        try:
            with open(config_path, "w") as f:
                f.write(raw)
            raw_holder["raw"] = raw
            for (sec, key), v in edits.items():
                cfg.setdefault(sec, OrderedDict())[key] = v
            log(f"[gui] saved {len(edits)} change(s) to {config_path}\n")
            return True
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return False

    def _extra_args() -> List[str]:
        return shlex.split(extra_var.get(), posix=(os.name != "nt"))

    def show_report():
        try:
            extra = _extra_args()
        except ValueError:
            extra = []
        d = dump_dir_for(cfg, config_dir, extra)
        t = THEMES[state["theme"]]
        win = tk.Toplevel(root); win.title("Report"); win.geometry("900x680"); win.configure(bg=t["bg"])
        txt = scrolledtext.ScrolledText(win, font=f_mono, bg=t["status_bg"], fg=t["status_fg"],
                                        insertbackground=t["status_fg"], relief="flat", bd=0)
        txt.pack(fill="both", expand=True, padx=8, pady=8)

        def section(title, body):
            txt.insert("end", f"\n{'='*70}\n  {title}\n{'='*70}\n{body}\n")
        found = False
        fr = os.path.join(d, "Final_report.txt")
        if os.path.exists(fr):
            section("PREDICTIONS (Final_report.txt)", open(fr).read()); found = True
        ar = os.path.join(d, "accuracy_report.json")
        if os.path.exists(ar):
            try:
                j = json.load(open(ar))
                section("ACCURACY", "\n".join(f"  {k}: {v}" for k, v in j.items() if k != "samples"))
                found = True
            except Exception:
                pass
        # analog.pwr_energy.generate_report writes these into the same dir
        for en in ("power_energy_latency_report.csv", "power_energy_latency_report.json"):
            ep = os.path.join(d, en)
            if os.path.exists(ep):
                section(f"ENERGY / LATENCY ({en})", open(ep).read()); found = True; break
        if not found:
            section("No report files found", f"Looked under:\n  {d}")
        txt.configure(state="disabled")

    def reader(p: subprocess.Popen):
        # worker thread: never touch Tk here, only log_q
        for line in iter(p.stdout.readline, ""):
            log_q.put(line)
        p.stdout.close(); p.wait()
        log_q.put(f"\n[gui] run finished (exit {p.returncode}).\n")

        def _done():
            proc_holder["p"] = None
            if p.returncode == 0:
                show_report()
        log_q.put(_done)

    def do_run():
        if proc_holder["p"] is not None:
            messagebox.showinfo("Busy", "A run is already in progress."); return
        try:
            extra = _extra_args()
        except ValueError as e:
            messagebox.showerror("Bad CLI args", str(e)); return
        if not do_save():
            return
        cmd = [sys.executable, "-m", "spicecimx.cli.infer", "--config", config_path]
        cmd += extra
        log(f"\n[gui] $ {subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd)}\n"
            f"[gui]   (cwd: {config_dir})\n")
        # cwd is the config dir, so make sure the child can still import the
        # spicecimx package this GUI was loaded from.
        env = dict(os.environ)
        pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = os.pathsep.join(
            [pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, cwd=config_dir, env=env)
        except Exception as e:
            messagebox.showerror("Launch failed", str(e)); return
        proc_holder["p"] = p
        threading.Thread(target=reader, args=(p,), daemon=True).start()

    def do_stop():
        p = proc_holder["p"]
        if p is not None:
            p.terminate(); log("\n[gui] stop requested.\n")

    def do_clear():
        status.delete("1.0", "end")

    def show_help():
        proj = project_dir_for(cfg, config_dir)
        # Prefer the rich HTML help opened in the default browser.
        here = os.path.dirname(os.path.abspath(__file__))
        html_candidates = [
            os.path.join(here, "assets", "tutorial.html"),   # shipped next to the GUI
            os.path.join(proj, "tutorial.html"),
            os.path.join(proj, "SPICECIMX_V3", "tutorial.html"),
            os.path.join(here, "..", "..", "tutorial.html"),
        ]
        html_path = next((c for c in html_candidates if os.path.exists(c)), None)
        if html_path:
            try:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(html_path))
                log(f"[gui] opened help: {os.path.abspath(html_path)}\n")
                return
            except Exception as e:
                log(f"[gui] could not open browser ({e}); showing text help\n")
        # Fallback: text window (TUTORIAL.md or the built-in blurb).
        t = THEMES[state["theme"]]
        win = tk.Toplevel(root); win.title("Help / Tutorial"); win.geometry("900x680"); win.configure(bg=t["bg"])
        box = scrolledtext.ScrolledText(win, font=f_mono, bg=t["status_bg"], fg=t["status_fg"],
                                        relief="flat", bd=0)
        box.pack(fill="both", expand=True, padx=8, pady=8)
        tut = None
        for cand in ("TUTORIAL.md", os.path.join(proj, "TUTORIAL.md"),
                     os.path.join(proj, "SPICECIMX_V3", "TUTORIAL.md")):
            if os.path.exists(cand):
                tut = open(cand).read(); break
        box.insert("end", tut or _BUILTIN_HELP); box.configure(state="disabled")

    # ── toolbar ──
    bar = ttk.Frame(root, padding=(12, 8)); bar.pack(fill="x")

    def tbtn(text, icon, cmd, accent=False):
        return ttk.Button(bar, text="  " + text, image=icons.get(icon), compound="left",
                          command=cmd, style="Accent.TButton" if accent else "TButton")
    tbtn("Run", "run", do_run, accent=True).pack(side="left", padx=4)
    tbtn("Stop", "stop", do_stop).pack(side="left", padx=4)
    tbtn("Save config", "save", lambda: do_save()).pack(side="left", padx=4)
    tbtn("Report", "report", show_report).pack(side="left", padx=4)
    tbtn("Help / Tutorial", "help", show_help).pack(side="left", padx=4)
    tbtn("Clear log", "clear", do_clear).pack(side="left", padx=4)
    ttk.Button(bar, text="Quit", command=root.destroy).pack(side="right", padx=4)

    apply_theme()
    drain_log()
    log("[gui] ready. Edit parameters in the tabs, then Save + Run.\n"
        "[gui] tip: set SIM > mode = auto for transient energy + fastdc accuracy.\n")
    root.mainloop()
    return 0


_BUILTIN_HELP = """\
SPICECiM-X - quick help

1. Edit any parameter in the section tabs (DATA, MODEL, HW, SIM, PATHS, ...).
   - SIM > mode:  dc | fastdc | transient | hybrid | auto
       auto = transient pass (power/energy/latency on 1 image) + fastdc pass
              (accuracy on all images); reports both.
   - PATHS values: use the folder buttons to browse.
2. 'Save config' writes your edits back to config.ini (comments preserved).
3. 'extra CLI args' overrides per-run, e.g.:  --mode auto --limit 5
4. 'Run' launches inference and streams the log; on success the report window
   pops up. 'Stop' terminates the run. Toggle night/light with the sun/moon.

Pipeline reminder:
   characterize (cell)  ->  characterize_array  ->  infer
The first two are one-time; infer auto-invokes the array characterisation if
the cache is missing/stale.
"""


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="SPICECiM-X config/run GUI")
    ap.add_argument("--config", default="config.ini")
    a = ap.parse_args(argv)
    return launch_gui(a.config)


if __name__ == "__main__":
    raise SystemExit(main())