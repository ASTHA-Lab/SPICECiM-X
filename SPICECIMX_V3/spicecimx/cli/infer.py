"""
SPICECiM-X v3 — ViT IMC inference CLI.

Equivalent of v2's ``reram_imc_infer.py``: loads a trained ViT_ReRAM
checkpoint and runs SPICE-accurate IMC inference on a directory of
images.  Drop-in replacement command:

    # v2
    python reram_imc_infer.py --batch_all

    # v3
    python -m spicecimx.cli.infer --batch_all

All hardware / simulator / path knobs are read from ``config.ini`` (the
same file your trainer uses), in the new ``[HW]``, ``[SIM]``, and
``[PATHS]`` sections.  Command-line flags override config values.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from spicecimx.config import load_cfg, make_parser, resolve_paths, _PATH_KEYS
from spicecimx.core.imc_matmul import imc_matmul as _imc_kernel, PolarityCache
from spicecimx.core.vit_forward import (
    vit_forward_imc, vit_forward_imc_batched,
)
from spicecimx.analog.tia_adc import TIACalibrator, tia_adc_realistic
from spicecimx.io.torch_bridge import (
    load_project_modules, find_checkpoint, build_preprocess,
    extract_vit_weights, load_state_dict_file,
)
from spicecimx.sim import make_backend
from spicecimx.sim.transient_backend import build_context


# ---------------------------------------------------------------------------
# Derived utilities
# ---------------------------------------------------------------------------
def _derive_batch_dirs(paths_cfg, images_dir: str):
    """Same convention as v2: scope hwsimdb/tmp by the leaf folder name."""
    batch_id = os.path.basename(os.path.normpath(images_dir)) or "default"
    hwsimdb = os.path.join(paths_cfg.hwsimdb_root, batch_id)
    tmp     = os.path.join(paths_cfg.tmp_root, batch_id)
    return batch_id, hwsimdb, tmp


def _hw_conductance_window(hw) -> Tuple[float, float]:
    """
    Return the (Gmax, Gmin) conductance window that vit_forward should
    derive its affine map ``c, d`` from.  This is the *cell-level* window
    (1T1R series combination of access transistor + ReRAM), not the bare
    ReRAM window, so that the affine de-mapping in ``imc_matmul``
    corresponds to the conductance actually realised at the bit-line.

    With R_access == 0 this reduces to the bare-ReRAM window (v2 behavior).
    """
    from spicecimx.hw.device_emit import effective_cell_window
    alpha = hw.Ik1 / hw.Vref
    G_R_max = alpha * np.exp(-((hw.Tox - hw.Tmax) / (hw.Tox - hw.Tref)))
    G_R_min = alpha * np.exp(-((hw.Tox - hw.Tmin) / (hw.Tox - hw.Tref)))
    G_cell_max, G_cell_min = effective_cell_window(
        G_R_max=float(G_R_max), G_R_min=float(G_R_min),
        R_access=hw.R_access,
    )
    return G_cell_max, G_cell_min


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        "SPICECiM-X v3 — ViT ReRAM-IMC inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="config.ini",
                   help="Path to config.ini (DATA/MODEL/TRAIN/SAVE/HW/SIM/PATHS).")
    p.add_argument("--project", type=str, default=None,
                   help="Project root with config.py, main.py, vit_reram.py "
                        "(overrides paths.project_dir).")
    p.add_argument("--images_dir", type=str, default=None,
                   help="Directory of images to run (overrides paths.images_dir).")
    p.add_argument("--checkpoint", type=str, default="",
                   help="Explicit checkpoint path (else best.pt/last.pt in [SAVE] "
                        "out_dir, then <project>/outputs).")
    p.add_argument("--dump_root", type=str, default=None,
                   help="Per-image CSV dump root (overrides paths.dump_root).")
    p.add_argument("--mode", type=str, default=None,
                   choices=["transient", "dc", "hybrid", "fastdc", "auto"],
                   help="Simulator mode (overrides sim.mode). 'auto' = transient "
                        "for power/energy/latency (1 image) + fastdc for accuracy.")
    p.add_argument("--licenses", type=int, default=None,
                   help="Number of Spectre licenses (overrides sim.license_count).")
    p.add_argument("--output_classes", type=int, default=None,
                   help="Trim final logits to first K columns.")
    p.add_argument("--batch_all", action="store_true",
                   help="Time-multiplex all images through each IMC matmul.")
    p.add_argument("--no_intermediates", action="store_true",
                   help="Skip per-image CSV dumps for speed.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (0 = all).")
    p.add_argument("--gui", action="store_true",
                   help="Launch the configuration/run GUI instead of running.")
    # internal flags used by 'auto' mode to split the two passes
    p.add_argument("--only_image", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--no_accuracy", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_energy", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_clean", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Physical converter chain helpers (feature 5)
# ---------------------------------------------------------------------------
def _materialize_netlist(src: str, dst: str) -> None:
    """Copy a netlist next to a deck, rewriting relative ``include`` /
    ``ahdl_include`` paths to absolute ones (relative to ``src``'s folder) so
    the copy works from any directory."""
    import re as _re
    base = os.path.dirname(os.path.abspath(src))
    with open(src, "r") as fh:
        text = fh.read()

    def _abs(m):
        p = m.group(2)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(base, p))
        return f'{m.group(1)}"{p}"'
    text = _re.sub(r'^(\s*(?:ahdl_include|include)\s+)"([^"]+)"', _abs, text,
                   flags=_re.M)
    with open(dst, "w") as fh:
        fh.write(text)


def _char_includes(paths) -> List[str]:
    """Netlist files the converter characterisation decks include."""
    inc = ["standard_cell.scs"]
    if getattr(paths, "peripheral_netlist", ""):
        inc.append("peripherals.scs")
    return inc


def _run_char_decks(decks, ctx, workdir: str):
    """Run each (name, deck_text) characterisation deck through Spectre and
    return ``{name: tran_path}``.  Uses the same scheduler/primitives as the
    crossbar flow.  The decks include ``standard_cell.scs`` (and
    ``peripherals.scs`` when [PATHS] peripheral_netlist is set): both are
    copied into each scratch dir with their include paths made absolute."""
    from spicecimx.sim.spectre import SpectreJob, build_spectre_command

    jobs = []
    for name, deck_text in decks:
        d = os.path.join(workdir, f"_char_{name}")
        os.makedirs(d, exist_ok=True)
        deck_path = os.path.join(d, "input.scs")
        with open(deck_path, "w") as fh:
            fh.write(deck_text)
        _materialize_netlist(ctx.paths.stdcell_netlist,
                             os.path.join(d, "standard_cell.scs"))
        if getattr(ctx.paths, "peripheral_netlist", ""):
            _materialize_netlist(ctx.paths.peripheral_netlist,
                                 os.path.join(d, "peripherals.scs"))
        raw = os.path.join(d, "psf")
        import shutil
        shutil.rmtree(raw, ignore_errors=True)
        os.makedirs(raw, exist_ok=True)
        log = os.path.join(raw, "spectre.out")
        cmd = build_spectre_command(deck_path, raw, log,
                                    extra_args=ctx.sim.spectre_args)
        jobs.append(SpectreJob(
            job_id=f"char/{name}", cmd=cmd, cwd=d, raw_dir=raw,
            branch="char", subarray_row=0, subarray_col=0,
            extra={"name": name}))

    sched = getattr(ctx, "scheduler", None)
    if sched is None:                    # fastdc has no Spectre scheduler
        from spicecimx.sim.spectre import JobScheduler, LicensePool
        sched = JobScheduler(LicensePool(max(1, min(len(jobs), ctx.sim.license_count))),
                             post_workers=1, pipeline_postprocess=False)
    results = sched.run_layer(
        jobs, postprocess_fn=_char_deck_post, progress_cb=lambda *a, **k: None)
    out = {}
    for r in results:
        tran = (r.post or {}).get("tran") if getattr(r, "ok", False) else None
        if tran and os.path.exists(tran):
            out[r.post["name"]] = tran
        else:
            print(f"[hw] characterisation job failed: {r.job.job_id} "
                  f"({r.error or 'no tran.tran.tran'}); see "
                  f"{os.path.join(r.job.raw_dir, 'spectre.out')}")
    return out


def _char_deck_post(job):
    """Post-processor for the converter decks (module level: the scheduler
    runs it in a spawn-context process pool, which must pickle it)."""
    return {"name": job.extra["name"],
            "tran": os.path.join(job.raw_dir, "tran.tran.tran")}


def _build_physical_converters(cfg, ctx, registry, workdir: str):
    """Characterise the hardware-mode converters with SPICE and return a
    PhysicalConverters.  Raises on failure: H roles were requested
    explicitly, so silently using the software models would mislabel the
    results."""
    from spicecimx.analog.peripheral_chain import characterize_converters
    from spicecimx.io.parse_transient import read_psfascii_tran
    pcv = characterize_converters(
        registry, cfg,
        run_decks=lambda decks: _run_char_decks(decks, ctx, workdir),
        read_fn=read_psfascii_tran,
        workdir=workdir,
        includes=_char_includes(cfg.paths),
    )
    print(f"[hw] physical converters ready — "
          f"DAC={'spice' if pcv.dac_hw else 'sw'}, "
          f"TIA={'spice' if pcv.tia_hw else 'sw'}, "
          f"SUB={'spice' if pcv.sub_hw else 'sw'}, "
          f"ADC={'spice' if pcv.adc_hw else 'sw'}; measured energy "
          f"(J/conv): DAC={pcv.e_dac_J:.3e} TIA={pcv.e_tia_J:.3e} "
          f"SUB={pcv.e_sub_J:.3e} ADC={pcv.e_adc_J:.3e}")
    if pcv.dac_hw and pcv.dac_table is not None:
        print(f"[hw]    DAC transfer: code 0 -> {pcv.dac_table[0]:.4f} V, "
              f"code {len(pcv.dac_table) - 1} -> {pcv.dac_table[-1]:.4f} V")
    if pcv.adc_hw and pcv.adc_codes is not None:
        ok = pcv.adc_codes >= 0
        print(f"[hw]    readout transfer: I in ±{pcv.i_fullscale:.3e} A -> codes "
              f"{int(pcv.adc_codes[ok].min())}..{int(pcv.adc_codes[ok].max())} "
              f"({len(np.unique(pcv.adc_codes[ok]))} distinct)")
    return pcv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    if getattr(args, "gui", False):
        from spicecimx.cli.gui import launch_gui
        return launch_gui(args.config)
    mode = (args.mode or load_cfg(args.config).sim.mode).lower()
    if mode == "auto":
        return _run_auto(args)
    return _run_pipeline(args)


def _spawn_background_delete(paths) -> None:
    """Delete the given paths in a detached process so the run never waits
    for a slow recursive delete (portable: no reliance on ``rm``)."""
    import subprocess
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return
    code = ("import shutil, sys\n"
            "for p in sys.argv[1:]:\n"
            "    shutil.rmtree(p, ignore_errors=True)\n")
    kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED | NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True                   # survives this process
    try:
        subprocess.Popen([sys.executable, "-c", code, *paths], **kw)
    except OSError as e:
        print(f"[clean] background delete not started ({e}); "
              f"remove {paths} by hand")


def _is_within(child: str, parent: str) -> bool:
    child, parent = os.path.abspath(child), os.path.abspath(parent)
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:                                   # different drives
        return False


def prepare_clean_workspace(cfg) -> None:
    """Give this batch a clean slate.

    Only the batch's own working/output dirs are cleared —
    ``hwsimdb_root/<batch>``, ``dump_root/<batch>`` and ``tmp_root/<batch>``
    where ``<batch>`` is the leaf name of ``images_dir`` — so reports of other
    image batches and runs of other batches in the same project survive.  Each
    dir is moved into ``<project>/trash/<run_id>`` (an instant rename on the
    same filesystem) and deleted in the background.

    A target that is (or contains) the project dir, the images, the support
    files or the trainer output is refused instead of being moved.
    """
    import time
    import shutil
    batch_id = os.path.basename(os.path.normpath(cfg.paths.images_dir)) or "default"
    targets = []
    for root in (cfg.paths.hwsimdb_root, cfg.paths.dump_root, cfg.paths.tmp_root):
        if not root:
            continue
        t = os.path.join(root, batch_id)
        if t not in targets:
            targets.append(t)

    protected = [cfg.paths.project_dir, cfg.paths.images_dir,
                 os.path.dirname(cfg.paths.stdcell_netlist or "") or None,
                 cfg.paths.array_table_path, cfg.paths.char_table_path]
    out_dir = getattr(cfg.save, "out_dir", "")
    if out_dir:
        protected.append(out_dir if os.path.isabs(out_dir)
                         else os.path.join(cfg.paths.project_dir, out_dir))
    for t in targets:
        for p in protected:
            if p and (_is_within(p, t) or os.path.abspath(p) == os.path.abspath(t)):
                raise ValueError(
                    f"[clean] refusing to clear {t}: it contains {p}. Point "
                    f"hwsimdb_root / dump_root / tmp_root at dedicated folders "
                    f"or set [SAVE] clean_workspace = false.")

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    trash_run = os.path.join(cfg.paths.project_dir, "trash", run_id)
    moved = []
    for t in targets:
        if os.path.isdir(t) and os.listdir(t):
            os.makedirs(trash_run, exist_ok=True)
            dest = os.path.join(trash_run, f"{len(moved)}_" +
                                os.path.basename(os.path.dirname(os.path.normpath(t))))
            try:
                os.rename(t, dest)                  # instant on same filesystem
            except OSError:
                shutil.move(t, dest)                # cross-filesystem fallback
            moved.append(dest)
        os.makedirs(t, exist_ok=True)               # fresh empty dir for this run
    if moved:
        print(f"[clean] cleared {len(moved)} stale dir(s) of batch '{batch_id}' -> "
              f"{trash_run} (deleting in background)")
        _spawn_background_delete([trash_run])


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _discover_images(images_dir, limit=0):
    """All images under ``images_dir`` (recursive), sorted, extension match
    case-insensitive, each file once."""
    files = []
    for root, _dirs, names in os.walk(images_dir):
        for n in names:
            if n.lower().endswith(_IMG_EXTS):
                files.append(os.path.join(root, n))
    files.sort()
    if limit and limit > 0:
        files = files[:limit]
    return files


def _run_auto(args):
    """auto mode: one transient pass (power/energy/latency, no accuracy) on a
    single image, then a fastdc pass (accuracy, no energy) on all images."""
    import copy, random as _random, shutil
    cfg = _load_with_overrides(args)
    img_files = _discover_images(cfg.paths.images_dir, args.limit)
    if not img_files:
        raise FileNotFoundError(f"No images found under {cfg.paths.images_dir}")

    # ── Transient image selection ─────────────────────────────────────────
    # Priority: named image ([SIM] auto_transient_image) > legacy fixed index
    # (auto_transient_image_index >= 0) > N distinct random images
    # (auto_transient_num_images, seeded by auto_transient_seed).  With more
    # than one image, the transient pass runs per image and the reported
    # crossbar energy/power is the AVERAGE across them.
    name = (getattr(cfg.sim, "auto_transient_image", "") or "").strip()
    idx = cfg.sim.auto_transient_image_index
    if name:
        matches = [f for f in img_files if os.path.basename(f) == name]
        if not matches:
            raise FileNotFoundError(
                f"[auto] [SIM] auto_transient_image='{name}' is not among the "
                f"{len(img_files)} image(s) under {cfg.paths.images_dir}"
                + (f" (--limit {args.limit} may have excluded it)" if args.limit else "")
                + ". Fix the name or leave it empty for random selection.")
        picks = matches[:1]
        how = f"fixed image '{name}'"
    elif idx is not None and idx >= 0:
        if idx >= len(img_files):
            raise IndexError(f"[auto] auto_transient_image_index={idx} but only "
                             f"{len(img_files)} image(s) were found")
        picks = [img_files[idx]]
        how = f"fixed index {idx}"
    else:
        n = max(1, int(getattr(cfg.sim, "auto_transient_num_images", 1) or 1))
        n = min(n, len(img_files))
        picks = _random.Random(cfg.sim.auto_transient_seed).sample(img_files, n)
        how = f"{n} random image(s), seed={cfg.sim.auto_transient_seed}"

    # Clean ONCE for the whole auto run (after the inputs are validated); the
    # two passes share the same dirs, so the sub-passes must not re-clean.
    if getattr(cfg.save, "clean_workspace", True) and not getattr(args, "no_clean", False):
        prepare_clean_workspace(cfg)

    print("=" * 74)
    print("  AUTO MODE — transient (power/energy/latency) + fastdc (accuracy)")
    print("=" * 74)
    print(f"[auto] transient power/energy/latency pass: {how}")
    for k, img in enumerate(picks):
        print(f"        [{k}] {os.path.basename(img)}")

    subdirs, img_map = [], {}
    multi = len(picks) > 1
    for k, img in enumerate(picks):
        stem = os.path.splitext(os.path.basename(img))[0]
        sub = f"timg{k:02d}_{stem}" if multi else ""
        a1 = copy.copy(args)
        a1.mode = "transient"; a1.only_image = img
        a1.no_accuracy = True; a1.no_energy = False; a1.no_clean = True
        a1.hwsimdb_subdir = sub
        _run_pipeline(a1)
        if sub:
            subdirs.append(sub)
            img_map[sub] = os.path.basename(img)

    # Multi-image: write the marker so generate_report averages the images,
    # then emit the consolidated (averaged) report at the batch level.
    if multi:
        _, hwsimdb_batch, _ = _derive_batch_dirs(cfg.paths, cfg.paths.images_dir)
        os.makedirs(hwsimdb_batch, exist_ok=True)
        with open(os.path.join(hwsimdb_batch, "transient_images.json"), "w") as f:
            json.dump({"subdirs": subdirs, "images": img_map}, f, indent=2)
        dr = os.path.join(cfg.paths.dump_root, os.path.basename(
            os.path.normpath(cfg.paths.images_dir)) or "default")
        # The operation manifest is count-based and identical for every
        # transient image; each sub-pass wrote it under dump/<batch>/<sub>/.
        # Lift the first one (and any measured converter energy) to the batch
        # level, where the consolidated report reads it.
        for fn in ("peripheral_ops.json", "measured_converter_energy.json",
                   "interconnect_report.json"):
            src = os.path.join(dr, subdirs[0], fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dr, fn))
        try:
            from spicecimx.analog.pwr_energy import generate_report
            print(f"[auto] consolidated report: crossbar = average of "
                  f"{len(subdirs)} transient image(s)")
            generate_report(dr, args.config, hwsimdb_dir=hwsimdb_batch)
        except Exception as e:
            print(f"[auto] consolidated energy report failed: "
                  f"{type(e).__name__}: {e}")

    print(f"\n[auto] fastdc accuracy pass on {len(img_files)} image(s)")
    a2 = copy.copy(args)
    a2.mode = "fastdc"; a2.only_image = None
    a2.no_accuracy = False; a2.no_energy = True; a2.no_clean = True
    _run_pipeline(a2)

    batch_id = os.path.basename(os.path.normpath(cfg.paths.images_dir)) or "default"
    dump_root = os.path.join(cfg.paths.dump_root, batch_id)
    print("\n" + "=" * 74)
    print("  AUTO MODE COMPLETE")
    _names = ", ".join(os.path.basename(x) for x in picks)
    print(f"    power/energy/latency  <- transient "
          f"({'avg of ' if len(picks) > 1 else 'image: '}{_names})")
    print(f"    accuracy              <- fastdc ({len(img_files)} images)")
    print(f"    reports under: {dump_root}")
    print("      power_energy_latency_report.{json,csv} (transient)   "
          "accuracy_report.json + Final_report.txt (fastdc)")
    print("=" * 74)


def _load_with_overrides(args):
    """load_cfg + the CLI overrides that change paths.  Paths given on the
    command line are relative to the current directory; everything is
    absolute afterwards."""
    cfg = load_cfg(args.config)
    if getattr(args, "project", None):
        # re-root the config's relative paths under the new project dir
        raw = make_parser()
        raw.read(args.config)
        cfg.paths.project_dir = os.path.abspath(args.project)
        for k in _PATH_KEYS:
            if raw.has_option("PATHS", k):
                setattr(cfg.paths, k, raw.get("PATHS", k).strip())
        resolve_paths(cfg.paths, os.getcwd())
    if getattr(args, "images_dir", None):
        cfg.paths.images_dir = os.path.abspath(args.images_dir)
    if getattr(args, "dump_root", None):
        cfg.paths.dump_root = os.path.abspath(args.dump_root)
    return cfg


def _run_pipeline(args):
    cfg = _load_with_overrides(args)

    # Apply CLI overrides
    if args.mode:        cfg.sim.mode          = args.mode
    if args.licenses is not None: cfg.sim.license_count = args.licenses

    # Fresh slate: move stale hwsimdb/imc_traces/tmp to ./trash and delete in
    # the background (skipped for auto's 2nd pass, which shares the 1st's dirs).
    if getattr(cfg.save, "clean_workspace", True) and not getattr(args, "no_clean", False):
        prepare_clean_workspace(cfg)

    # Resolve batch-scoped dirs.  In multi-image auto mode each transient
    # image gets its own hwsimdb/tmp subdir (args.hwsimdb_subdir) so the
    # per-image waveforms don't overwrite each other.
    batch_id, hwsimdb_dir, tmp_dir = _derive_batch_dirs(cfg.paths, cfg.paths.images_dir)
    sub = getattr(args, "hwsimdb_subdir", "") or ""
    if sub:
        batch_id = os.path.join(batch_id, sub)
        hwsimdb_dir = os.path.join(hwsimdb_dir, sub)
        tmp_dir = os.path.join(tmp_dir, sub)
    os.makedirs(hwsimdb_dir, exist_ok=True)
    os.makedirs(tmp_dir,     exist_ok=True)
    print(f"[batch] ID={batch_id}  hwsimdb={hwsimdb_dir}  tmp={tmp_dir}")
    print(f"[sim]   mode={cfg.sim.mode}  licenses={cfg.sim.license_count}")

    # Load user's trainer modules and trained checkpoint
    load_cfg_fn, build_model, ViT_ReRAM = load_project_modules(cfg.paths.project_dir)
    # User's trainer config (DATA/MODEL/TRAIN/SAVE) — we re-read with their loader
    user_cfg = load_cfg_fn(args.config)
    # Build model on CPU (SPICE post-processing happens on numpy)
    device = torch.device("cpu")
    ckpt_path = find_checkpoint(cfg.paths.project_dir, args.checkpoint,
                                out_dir=getattr(getattr(user_cfg, "save", None),
                                                "out_dir", "") or "")
    if not ckpt_path:
        raise FileNotFoundError(
            f"No checkpoint found: looked for best.pt/last.pt in [SAVE] out_dir "
            f"and {os.path.join(cfg.paths.project_dir, 'outputs')}. "
            f"Pass --checkpoint <path>.")
    print(f"[load] checkpoint: {ckpt_path}")
    sd = load_state_dict_file(ckpt_path)
    # Input channels: taken from the checkpoint's patch projection when present
    # (exact), else from [DATA] channels (auto -> 1 for *mnist*, 3 otherwise).
    if "patch_proj.weight" in sd:
        C = int(sd["patch_proj.weight"].shape[1])
    elif str(user_cfg.data.channels) != "auto":
        C = int(user_cfg.data.channels)
    else:
        C = 1 if "mnist" in user_cfg.data.dataset.lower() else 3
    model = build_model(user_cfg, in_channels=C, num_classes=user_cfg.model.num_classes).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {ckpt_path} does not match the model built from "
            f"[MODEL]: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}. "
            f"Check [MODEL]/[DATA] against the training config.")
    model.eval()

    # Pre-extract weights once
    weights = extract_vit_weights(model)

    # Discover images
    img_files: List[str] = _discover_images(cfg.paths.images_dir, args.limit)
    if getattr(args, "only_image", None):
        _only = os.path.abspath(args.only_image)
        img_files = [q for q in img_files if os.path.abspath(q) == _only] or [args.only_image]
    if not img_files:
        raise FileNotFoundError(f"No images found under {cfg.paths.images_dir}")
    print(f"[input] {len(img_files)} image(s)")

    # ── Build (or load) the array characterisation table FIRST ────────────
    # It is the SPICE ground truth for the cell + access transistor + BL
    # driver, so both the V-linearizer AND the effective series resistance
    # R_access are derived from it.
    _array_tab = None
    if getattr(cfg.paths, "array_table_path", None):
        from spicecimx.cli.characterize_array import build_or_load as _build_array
        _array_tab = _build_array(cfg, force=False)

    # ── Resolve R_access ('auto' -> derive from the array table) ──────────
    if cfg.hw.R_access is None or cfg.hw.R_access < 0:
        if _array_tab is not None:
            from spicecimx.hw.array_table import estimate_r_access
            est = estimate_r_access(_array_tab, Ik1=cfg.hw.Ik1, Vref=cfg.hw.Vref,
                                    Tox=cfg.hw.Tox, Tref=cfg.hw.Tref)
            if est:
                cfg.hw.R_access = est["R_access"]
                print(f"[hw]    R_access={cfg.hw.R_access:.3e} ohm  [auto: fitted "
                      f"from array table, median@LRS={est['R_access_lrs']:.3e}, "
                      f"{est['n_points']} pts]")
            else:
                cfg.hw.R_access = 0.0
                print("[hw]    R_access=0 (auto failed: array table looks ideal)")
        else:
            cfg.hw.R_access = 0.0
            print("[hw]    R_access=0 (auto requested but no array_table_path; "
                  "set [PATHS] array_table_path or give a number in [HW])")
    else:
        print(f"[hw]    R_access={cfg.hw.R_access:.3e} ohm  [from config]")

    # Build the IMC plumbing (one set for the whole run)
    Gmax, Gmin = _hw_conductance_window(cfg.hw)
    alpha = cfg.hw.Ik1 / cfg.hw.Vref
    G_R_max = float(alpha * np.exp(-((cfg.hw.Tox - cfg.hw.Tmax) / (cfg.hw.Tox - cfg.hw.Tref))))
    G_R_min = float(alpha * np.exp(-((cfg.hw.Tox - cfg.hw.Tmin) / (cfg.hw.Tox - cfg.hw.Tref))))
    print(f"[hw]    bare ReRAM:   Gmin={G_R_min:.3e}  Gmax={G_R_max:.3e}")
    print(f"[hw]    cell (1T1R):  Gmin={Gmin:.3e}  Gmax={Gmax:.3e}  "
          f"(LRS compressed by {(1.0 - Gmax/G_R_max)*100:.1f}%)")

    # ── Voltage pre-warp (h_array⁻¹) for all backends ─────────────────────
    # Build (or load from cache) the array-level I-V characterisation table
    # ONCE per run and use it for V-linearization in every backend.  The
    # array char captures cell I-V nonlinearity PLUS BL_DRIVER IR drop;
    # driving each VI source at V_phys = h_array⁻¹(V_target) makes the
    # column current track G0_array·V_target to within ~0.5% per layer.
    #
    # The same h_array curve serves all three backends:
    #   * fastdc  — pre-warps V before per-cell table lookup;
    #   * dc      — pre-warps V before Spectre deck emission;
    #   * transient — pre-warps V before PWL waveform emission.
    array_linearizer = None
    if _array_tab is not None:
        array_tab = _array_tab          # built above (also used for R_access)
        # The actual subarray size at inference is hw.subarray_rows (square
        # subarrays — typically 16 in this framework).  build_v_linearization
        # interpolates the table to this N if it's between the two anchors.
        N_inf = int(cfg.hw.subarray_rows)
        array_linearizer = array_tab.build_v_linearization(N=N_inf)
        print(f"[v-lin] using array h(V) @ N={N_inf}  "
              f"(super-linearity {array_linearizer.super_linearity_factor:.3f}, "
              f"T_ref={array_linearizer.T_ref*1e9:.2f} nm, "
              f"G0_array={array_linearizer.G0_array*1e6:.2f} μS)")

    if cfg.sim.mode.lower() == "fastdc":
        # Load or build the cell I-V characterisation table once per run.
        from spicecimx.hw.char_table import CellITable
        from spicecimx.sim.fastdc_backend import FastDcBackendContext

        char_path = getattr(cfg.paths, "char_table_path", None)
        if not char_path:
            raise ValueError(
                "fastdc mode requires paths.char_table_path in config.ini "
                "(point to a .npz or .csv).  Run "
                "`python -m spicecimx.cli.characterize` to build one."
            )
        if not os.path.isabs(char_path):
            char_path = os.path.join(cfg.paths.project_dir, char_path)
        print(f"[fastdc] loading cell table: {char_path}")
        if char_path.lower().endswith(".csv"):
            cell_table = CellITable.from_csv(
                char_path,
                V_min=getattr(cfg.sim, "fastdc_V_min", 0.1),
                V_max=getattr(cfg.sim, "fastdc_V_max", 1.0),
                V_step=getattr(cfg.sim, "fastdc_V_step", 0.01),
            )
        else:
            cell_table = CellITable.from_npz(char_path)
        print(f"[fastdc] table: nT={cell_table.nT} ({cell_table.T_min*1e9:.2f}"
              f"..{cell_table.T_max*1e9:.2f} nm), nV={cell_table.nV} "
              f"({cell_table.V_min:.2f}..{cell_table.V_max:.2f} V)")

        ctx = FastDcBackendContext(
            hw=cfg.hw, sim=cfg.sim, paths=cfg.paths,
            hwsimdb_dir=hwsimdb_dir, tmp_dir=tmp_dir,
            cell_table=cell_table, batch_id=batch_id,
        )
        # Prefer the array linearizer (captures loading) over the per-cell
        # one (single-cell h only) if both are available.  Falls back to
        # cell_table.build_v_linearization() when no array table is configured.
        if array_linearizer is not None:
            ctx.v_linearizer = array_linearizer
        else:
            ctx.v_linearizer = cell_table.build_v_linearization()
            print("[fastdc] V-linearization on (single-cell h(V); for array-loading "
                  "compensation set paths.array_table_path and re-run)")
    else:
        ctx = build_context(
            hw=cfg.hw, sim=cfg.sim, paths=cfg.paths,
            batch_id=batch_id,
            pdk=cfg.pdk, parasitics=cfg.parasitics,
        )
        # DC and transient modes: same array linearizer, applied to the
        # voltages they emit to Spectre.  When no array table is available,
        # fall back to single-cell h (the pre-array-char behaviour) so
        # existing configs continue to work — they'll just see the
        # array-loading residual as the dominant per-layer error term.
        if array_linearizer is not None:
            ctx.v_linearizer = array_linearizer
        elif cfg.sim.mode.lower() == "dc":
            char_path = getattr(cfg.paths, "char_table_path", None)
            if char_path:
                from spicecimx.hw.char_table import CellITable
                if not os.path.isabs(char_path):
                    char_path = os.path.join(cfg.paths.project_dir, char_path)
                print(f"[dc] (fallback) loading single-cell table for V-linearization: {char_path}")
                if char_path.lower().endswith(".csv"):
                    cell_table = CellITable.from_csv(
                        char_path,
                        V_min=getattr(cfg.sim, "fastdc_V_min", 0.1),
                        V_max=getattr(cfg.sim, "fastdc_V_max", 1.0),
                        V_step=getattr(cfg.sim, "fastdc_V_step", 0.01),
                    )
                else:
                    cell_table = CellITable.from_npz(char_path)
                ctx.v_linearizer = cell_table.build_v_linearization()
                print("[dc] V-linearization on (single-cell h only; ~5% layer residual expected). "
                      "For best accuracy set paths.array_table_path and re-run.")
            else:
                print("[dc] V-linearization OFF — neither array_table_path nor "
                      "char_table_path is set.  Per-layer rmse ~10-20% expected.")
    # ── Tile interconnect ([INTERCONNECT]): shared-line loading + NoC ─────
    if cfg.interconnect.enable:
        from spicecimx.arch.floorplan import resolve_interconnect
        ctx.interconnect = cfg.interconnect
        ctx.ic_params = resolve_interconnect(cfg.interconnect, cfg.hw, cfg.pdk,
                                             cfg.peripherals)
        P = ctx.ic_params
        print(f"[ic]    tile interconnect ON: loading={cfg.interconnect.loading} "
              f"(passes={cfg.interconnect.loading_max_passes}), noc={cfg.interconnect.noc}; "
              f"cell pitch {P.pitch_x_um:.2f} x {P.pitch_y_um:.2f} um, wire "
              f"{P.r_um:.2f} ohm/um {P.c_um_F*1e15:.2f} fF/um, R_drv {P.r_drv:g} ohm, "
              f"R_tia {P.r_tia:.2f} ohm, tile {P.tile_w_um:.0f} x {P.tile_h_um:.0f} um")
        if cfg.interconnect.loading and cfg.sim.mode.lower() in ("dc", "hybrid"):
            print(f"[ic]    NOTE: shared-line loading is modelled in the transient and "
                  f"fastdc backends only; mode={cfg.sim.mode} runs ideal RSN boundaries.")
    sim_backend = make_backend(ctx)
    pcache = PolarityCache()

    # ── Operation ledger ─────────────────────────────────────────────────
    # Wrap the imc_matmul kernel so every (A,B) matmul records its (m,k,n)
    # shape for the energy/latency report, without touching imc_matmul or
    # vit_forward.  Fully transparent when the report is disabled.
    from spicecimx.analog.pwr_energy import OpLedger
    op_ledger = OpLedger(n_bit_slices=cfg.peripherals.n_bit_slices)

    def imc_kernel(A, B, **kw):
        op_ledger.record(tag=kw.get("tag", ""),
                         m=A.shape[0], k=A.shape[1], n=B.shape[1])
        # Drive range of the activation encoding V = a·A + b comes from [HW].
        kw.setdefault("vmin", cfg.hw.vmin)
        kw.setdefault("vmax", cfg.hw.vmax)
        return _imc_kernel(A, B, **kw)

    cal = TIACalibrator(p_lo=cfg.sim.tia_calib_p_lo, p_hi=cfg.sim.tia_calib_p_hi,
                        freeze_after=cfg.peripherals.tia_calib_images)
    print(f"[tia]   per-layer calibration, "
          + (f"frozen after {cfg.peripherals.tia_calib_images} call(s)"
             if cfg.peripherals.tia_calib_images > 0 else "never frozen")
          + f"; ADC full scale {cfg.peripherals.vmax_adc:g} V, "
            f"TIA window [{cfg.peripherals.vmin_tia:g}, {cfg.peripherals.vmax_tia:g}] V")

    # Holder for physical converters (feature 5).  Populated below after the
    # registry is built and the hardware-mode converters are characterised;
    # stays None when all converters are software (the default), so tia_apply
    # keeps its exact previous behaviour.
    _pcv_holder: Dict[str, Any] = {"pcv": None}

    def tia_apply(I, tag):
        """Per-call wrapper: update batch calibration, then quantize + invert.

        When the ADC role is hardware (declared H in [SUBCIRCUITS]) and the
        TIA+ADC chain has been SPICE-characterised, route I through the measured
        TIA+ADC transfer (peripheral_chain) instead of the software model — the
        ADC's real digital code becomes the output, matching the same
        (V_quant, I_hat, meta) contract the software path returns."""
        pcv = _pcv_holder["pcv"]
        cal.update(tag, I)
        Imin, Imax = cal.get(tag)
        if pcv is not None and getattr(pcv, "adc_hw", False) \
                and getattr(pcv, "adc_i_values", None) is not None:
            # Per-layer programmable TIA gain: map this layer's calibrated
            # range [Imin, Imax] onto the characterised sweep ±i_fs, push it
            # through the measured TIA->SUB->ADC transfer, and scale back.
            mid, half = 0.5 * (Imin + Imax), max(0.5 * (Imax - Imin), 1e-30)
            g = pcv.i_fullscale / half
            codes, values = pcv.tia_adc_apply((I - mid) * g)
            I_hat = values / g + mid
            nb = int(pcv.adc_bits)
            span = max(1, (2 ** nb) - 1)
            V_quant = cfg.peripherals.vmin_tia + (codes / span) * (cfg.peripherals.vmax_tia - cfg.peripherals.vmin_tia)
            return V_quant, I_hat, {"codes": codes, "Nbits": nb,
                                    "source": "spice_tia_adc"}
        return tia_adc_realistic(
            I,
            Vmin=cfg.peripherals.vmin_tia, Vmax=cfg.peripherals.vmax_tia,
            Nbits=cfg.peripherals.adc_bits,
            Imin=Imin, Imax=Imax,
            Vfs=cfg.peripherals.vmax_adc,    # ADC full-scale voltage
            add_qerror=cfg.peripherals.enable_qerror,
            use_batch_calib=False,           # we provide explicit Imin/Imax
        )

    # Per-image dump root
    dump_root = os.path.join(cfg.paths.dump_root, batch_id)
    os.makedirs(dump_root, exist_ok=True)

    # ── Physical converter chain (feature 5) ──────────────────────────────
    # Build the subckt registry from [SUBCIRCUITS] (validates that every
    # hardware-mode subckt exists in standard_cell.netlist).  If any of
    # dac/adc/tia/subtractor is hardware AND mode is transient, characterise
    # the real subckts once with SPICE and attach the measured transfers +
    # energies to the backend context.  Everything is gated: with the default
    # all-software converters this block is a no-op and the existing path is
    # untouched.  Any failure degrades gracefully to the software converters.
    from spicecimx.hw.subckt_registry import SubcktRegistry
    registry = SubcktRegistry.from_config(
        cfg.subcircuits.entries,
        [p for p in (cfg.paths.stdcell_netlist,
                     getattr(cfg.paths, "peripheral_netlist", "")) if p])
    print(registry.summary())
    for note in registry.warnings(cell_prefix=cfg.pdk.cell_prefix):
        print(f"[hw] NOTE: {note}")

    if any(registry.is_hardware(r) for r in ("dac", "adc", "tia", "subtractor")):
        # The converters are static transfers: characterised once with SPICE
        # (transient decks), then applied in every simulation mode, so the
        # accuracy pass of 'auto' (fastdc) sees the same converters as the
        # energy pass (transient).
        pcv = _build_physical_converters(cfg, ctx, registry, dump_root)
        ctx.physical_converters = pcv      # DAC apply in the backends
        _pcv_holder["pcv"] = pcv           # TIA+SUB+ADC apply in tia_apply
        pcv.write_measured_energy(dump_root)  # measured energy -> report

    # Preprocess pipeline
    from PIL import Image
    preprocess = build_preprocess(user_cfg, in_channels=C)

    # Load all images
    imgs: List[torch.Tensor] = []
    names: List[str] = []
    for fp in img_files:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception as e:
            print(f"  skip {fp}: {e}"); continue
        imgs.append(preprocess(img).unsqueeze(0))
        names.append(os.path.basename(fp))

    # ── Run inference ─────────────────────────────────────────────────────
    report_path = os.path.join(dump_root, "Final_report.txt")
    report_lines: List[str] = []

    out_classes = args.output_classes or user_cfg.model.num_classes
    if out_classes > user_cfg.model.num_classes:
        raise ValueError(f"--output_classes {out_classes} > [MODEL] num_classes "
                         f"{user_cfg.model.num_classes}")

    # ── Accuracy report (Torch-vs-GT / IMC-vs-GT / SW-HW) — extensible task ──
    acc = None
    gt_labels: Dict[str, Any] = {}
    if cfg.accuracy.generate_accuracy_report and not getattr(args, "no_accuracy", False):
        try:
            from spicecimx.core.metrics import (
                get_task, AccuracyAccumulator, load_labels,
            )
            task_name = cfg.accuracy.task.strip().lower()
            task = (get_task(task_name, topk=cfg.accuracy.topk)
                    if task_name == "classification"
                    else get_task(task_name))
            acc = AccuracyAccumulator(task)
            lf = cfg.accuracy.labels_file
            if lf and not os.path.isabs(lf):
                lf = os.path.join(cfg.paths.project_dir, lf)
            gt_labels = load_labels(
                names, labels_file=(lf or None),
                label_from_filename=(cfg.accuracy.label_from_filename or None))
            print(f"[acc] task={cfg.accuracy.task}; GT labels resolved for "
                  f"{len(gt_labels)}/{len(names)} image(s)")
        except Exception as e:
            print(f"[acc] accuracy report disabled ({type(e).__name__}: {e})")
            acc = None

    t0 = time.time()
    if args.batch_all:
        print("[mode] batched (time-multiplexed)")
        logits_list, extras = vit_forward_imc_batched(
            imgs, model,
            Gmax=Gmax, Gmin=Gmin,
            imc_kernel=imc_kernel,
            sim_backend=sim_backend,
            polarity_cache=pcache,
            tia_apply=tia_apply,
            dumps_dir=dump_root,
            output_classes=out_classes,
            save_intermediates=not args.no_intermediates,
        )
        # compare Torch and IMC over the same class set
        torch_logits = extras["torch_logits"][:, :out_classes]
        for i, name in enumerate(names):
            base = os.path.splitext(name)[0]
            dump_dir = os.path.join(dump_root, base)
            os.makedirs(dump_dir, exist_ok=True)
            _emit_per_image_csvs(dump_dir, torch_logits[i:i+1], logits_list[i])
            line = _format_prediction(name, torch_logits[i:i+1], logits_list[i])
            print(line); report_lines.append(line)
            if acc is not None:
                acc.add(torch_logits[i:i+1], logits_list[i],
                        gt=gt_labels.get(name), name=name)
    else:
        print("[mode] per-image")
        for i, (img, name) in enumerate(zip(imgs, names)):
            base = os.path.splitext(name)[0]
            dump_dir = os.path.join(dump_root, base)
            os.makedirs(dump_dir, exist_ok=True)

            imc_logits, extras = vit_forward_imc(
                img, model,
                Gmax=Gmax, Gmin=Gmin,
                imc_kernel=imc_kernel,
                sim_backend=sim_backend,
                polarity_cache=pcache,
                tia_apply=tia_apply,
                dumps_dir=dump_dir,
                output_classes=out_classes,
                save_intermediates=not args.no_intermediates,
                weights_dict=weights,
            )
            torch_logits = extras["torch_logits"][:, :out_classes]
            _emit_per_image_csvs(dump_dir, torch_logits, imc_logits)
            line = _format_prediction(name, torch_logits, imc_logits)
            print(line); report_lines.append(line)
            if acc is not None:
                acc.add(torch_logits, imc_logits,
                        gt=gt_labels.get(name), name=name)

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    elapsed = time.time() - t0
    print(f"\n[done] {len(names)} image(s) in {elapsed:.1f}s "
          f"({elapsed / max(1, len(names)):.1f}s/img)")
    print(f"[done] report: {report_path}")

    # ── Accuracy report (Torch-vs-GT / IMC-vs-GT / SW-HW agreement) ────────
    if acc is not None and acc.n > 0:
        print("\n" + acc.format())
        acc_path = os.path.join(dump_root, "accuracy_report.json")
        try:
            acc.write_json(acc_path)
            print(f"[acc] accuracy report: {acc_path}")
        except Exception as e:
            print(f"[acc] could not write accuracy json: {type(e).__name__}: {e}")

    # ── Power / energy / latency report (feature 2 + 3) ───────────────────
    if cfg.peripherals.generate_energy_report and not getattr(args, "no_energy", False):
        from spicecimx.analog.pwr_energy import (
            reduce_crossbar_csvs, generate_report,
        )
        # 1) Write the operation manifest — normalised to ONE inference, so
        #    peripheral counts and the transient crossbar energy (one image in
        #    per-image mode, the whole batch with --batch_all) agree.
        one_inf = op_ledger.per_inference(len(names), bool(args.batch_all))
        one_inf.write_manifest(
            dump_root, dac_bits=cfg.peripherals.dac_bits,
            adc_bits=cfg.peripherals.adc_bits,
            crossbar_inferences=(len(names) if args.batch_all else 1),
        )
        # 1b) Tile interconnect: shared-line loading records + NoC.
        if cfg.interconnect.enable:
            _write_interconnect_report(cfg, ctx, one_inf, dump_root,
                                       crossbar_inferences=(len(names)
                                                            if args.batch_all else 1))
        # 2) Crossbar SPICE energy is transient-only (DC/FastDC have no time
        #    domain).  Integrate the :pwr waveforms into per-layer CSVs.
        if cfg.sim.mode.lower() == "transient":
            print("[pwr] reducing crossbar power/energy CSVs...")
            # Mirror each layer's _pwr/_eng CSV into the dump, organised per
            # block (dump/crossbar_power/[<image>/]block_XX/<tag>_{pwr,eng}.csv)
            _sub = getattr(args, "hwsimdb_subdir", "") or ""
            dump_copy = os.path.join(dump_root, _sub) if _sub else dump_root
            reduce_crossbar_csvs(hwsimdb_dir, dump_dir=dump_copy)
            print(f"[pwr] per-block power/energy CSVs copied under: "
                  f"{os.path.join(dump_copy, 'crossbar_power')}")
        else:
            print(f"[pwr] mode={cfg.sim.mode}: no crossbar SPICE energy "
                  f"(time-domain unavailable); reporting peripheral + latency only.")
        # 3) Consolidated report.
        try:
            generate_report(dump_root, args.config, hwsimdb_dir=hwsimdb_dir)
        except Exception as e:
            print(f"[pwr] energy report failed: {type(e).__name__}: {e}")


def _write_interconnect_report(cfg, ctx, ledger, dump_dir: str, *,
                               crossbar_inferences: int = 1) -> str:
    """``<dump>/interconnect_report.json``: per-layer shared-line loading
    (transient: SPICE boundary relaxation; fastdc: line solve) and the NoC
    of ONE inference.  Read by analog.pwr_energy.generate_report."""
    from spicecimx.analog import tech_scale as _ts
    ic, P = cfg.interconnect, ctx.ic_params
    recs = list(getattr(ctx, "loading_records", []) or [])
    tot = lambda key: float(sum(r.get("line_energy_J", {}).get(key, 0.0) for r in recs))
    fin = lambda key: float(sum(r[key] for r in recs
                                if isinstance(r.get(key), float) and r[key] == r[key]))
    loading = {
        "enabled": bool(ic.loading), "mode": cfg.sim.mode,
        "n_layers": len(recs),
        "line_energy_J": {k: tot(k) for k in ("ir_in_J", "ir_out_J", "cv2_in_J",
                                              "cv2_out_J", "total_J")},
        "E_deck_ideal_J": fin("E_deck_ideal_J"),
        "E_deck_loaded_J": fin("E_deck_loaded_J"),
        "tau_line_max_s": max([max(r["tau_in_s"], r["tau_out_s"]) for r in recs],
                              default=0.0),
        "d_in_max_V": max([r["d_in_max_V"] for r in recs], default=0.0),
        "d_out_max_V": max([r["d_out_max_V"] for r in recs], default=0.0),
        "residual_max_V": max([r["residual_V"] for r in recs
                               if r.get("residual_V") == r.get("residual_V")],
                              default=float("nan")),
        "layers": recs,
    }
    noc = {"model": "none"}
    if ic.noc != "none" and ledger.layers:
        vdd = _ts.corner_vdd(P.node_nm, cfg.sim.process_corner,
                             vdd_override=cfg.sim.vdd)
        from spicecimx.arch.noc import estimate_noc
        noc = estimate_noc(ledger.layers, P, ic, dac_bits=cfg.peripherals.dac_bits,
                           adc_bits=cfg.peripherals.adc_bits, vdd=vdd,
                           workdir=os.path.join(dump_dir, "booksim"))
    out = {"params": P.as_dict(), "crossbar_inferences": int(crossbar_inferences),
           "noc_overlap": bool(ic.noc_overlap), "loading": loading, "noc": noc}
    path = os.path.join(dump_dir, "interconnect_report.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[ic]    interconnect report: {path}  (lines "
          f"{loading['line_energy_J']['total_J']*1e9:.3f} nJ over {len(recs)} layer(s); "
          f"NoC {noc.get('energy_J', 0.0)*1e9:.3f} nJ, {noc.get('latency_s', 0.0)*1e6:.2f} us)")
    return path


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _emit_per_image_csvs(dump_dir: str, torch_logits: np.ndarray,
                         imc_logits: np.ndarray):
    np.savetxt(os.path.join(dump_dir, "softmax_torch.csv"),
               _softmax(torch_logits), delimiter=",", fmt="%.6g")
    np.savetxt(os.path.join(dump_dir, "softmax_imc.csv"),
               _softmax(imc_logits), delimiter=",", fmt="%.6g")
    np.savetxt(os.path.join(dump_dir, "logits_imc.csv"),
               imc_logits, delimiter=",", fmt="%.6g")
    np.savetxt(os.path.join(dump_dir, "logits_torch.csv"),
               torch_logits, delimiter=",", fmt="%.6g")


def _format_prediction(name: str, torch_logits: np.ndarray,
                       imc_logits: np.ndarray) -> str:
    pt = _softmax(torch_logits); pi = _softmax(imc_logits)
    pred_t = int(np.argmax(pt, axis=1)[0]); p_t = float(np.max(pt))
    pred_i = int(np.argmax(pi, axis=1)[0]); p_i = float(np.max(pi))
    return (f"{name} | torch: {pred_t} (p={p_t:.3f}) | "
            f"imc: {pred_i} (p={p_i:.3f}) | match={pred_t == pred_i}")


if __name__ == "__main__":
    main()