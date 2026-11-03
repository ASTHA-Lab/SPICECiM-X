"""
================================================================================
  metrics.py — extensible accuracy reporting for SPICECiM-X
================================================================================

Produces the three-way accuracy comparison the inference flow needs:

  * **Torch vs GT**  — accuracy of the FP/software reference model.
  * **IMC vs GT**    — accuracy of the SPICE/hardware-simulated model.
  * **SW vs HW**     — agreement between the two (how often the hardware makes
                       the same decision as the software), independent of GT.

The report is built around a small **Task** abstraction so the *same* pipeline
serves different problem types.  Today the only task is image classification;
to support a new problem type later (e.g. an LLM doing next-token prediction)
you implement a `Task` subclass and register it — no change to infer.py or the
accumulator:

    classification : output = class logits  -> predict = argmax
                     GT = class index;  extra = top-k accuracy, mean confidence
    language_modeling (example sketch below):
                     output = per-position vocab logits -> predict = argmax/token
                     GT = target token ids;  extra = perplexity, token top-k

A Task only has to answer four questions about a model *output*:
  predict(output)        -> the discrete prediction(s)
  is_correct(output, gt) -> correctness vs ground truth (array of bools)
  agree(a, b)            -> do two outputs predict the same thing
  *_extra(...)           -> optional task-specific metrics (perplexity, top-k…)

The `AccuracyAccumulator` consumes (torch_output, imc_output, gt) tuples one at
a time and emits a JSON-serialisable report + a human summary.
================================================================================
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Task abstraction                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝
class Task(ABC):
    """A problem type.  Subclass + register to support new model outputs."""
    name: str = "task"

    @abstractmethod
    def predict(self, output: np.ndarray) -> np.ndarray:
        """Map a model output to its discrete prediction(s)."""

    @abstractmethod
    def is_correct(self, output: np.ndarray, gt: Any) -> np.ndarray:
        """Boolean array: is the prediction correct vs ground truth?"""

    def agree(self, out_a: np.ndarray, out_b: np.ndarray) -> np.ndarray:
        """Boolean array: do two outputs make the same prediction?"""
        return np.asarray(self.predict(out_a) == self.predict(out_b))

    def sample_extra(self, output: np.ndarray, gt: Any) -> Dict[str, float]:
        """Optional per-sample metrics, accumulated by mean."""
        return {}

    def has_gt(self, gt: Any) -> bool:
        return gt is not None


class ClassificationTask(Task):
    """Single-label classification: output = (…, n_classes) logits."""
    name = "classification"

    def __init__(self, topk: Sequence[int] = (1,)):
        self.topk = tuple(sorted(set(int(k) for k in topk)))

    def _logits2d(self, output: np.ndarray) -> np.ndarray:
        a = np.asarray(output, dtype=np.float64)
        return a.reshape(1, -1) if a.ndim == 1 else a

    def predict(self, output: np.ndarray) -> np.ndarray:
        return np.argmax(self._logits2d(output), axis=-1)

    def is_correct(self, output: np.ndarray, gt: Any) -> np.ndarray:
        pred = self.predict(output)
        gt = np.atleast_1d(np.asarray(gt))
        return pred == gt

    def sample_extra(self, output: np.ndarray, gt: Any) -> Dict[str, float]:
        z = self._logits2d(output)
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
        out: Dict[str, float] = {"confidence": float(np.max(p, axis=-1).mean())}
        if gt is not None:
            gt = np.atleast_1d(np.asarray(gt))
            order = np.argsort(-z, axis=-1)
            for k in self.topk:
                if k <= 1:
                    continue
                hit = np.any(order[:, :k] == gt[:, None], axis=-1)
                out[f"top{k}"] = float(hit.mean())
        return out


# ── future-task sketch (not registered by default) ─────────────────────────
class LanguageModelingTask(Task):
    """Next-token prediction skeleton, to show the abstraction scales beyond
    classification.  output = (…, seq, vocab) logits; gt = target token ids.
    Register with ``register_task('language_modeling', LanguageModelingTask)``
    once an LLM inference path exists."""
    name = "language_modeling"

    def predict(self, output: np.ndarray) -> np.ndarray:
        return np.argmax(np.asarray(output), axis=-1)        # per-position token

    def is_correct(self, output: np.ndarray, gt: Any) -> np.ndarray:
        return np.asarray(self.predict(output) == np.asarray(gt)).ravel()

    def sample_extra(self, output: np.ndarray, gt: Any) -> Dict[str, float]:
        if gt is None:
            return {}
        z = np.asarray(output, dtype=np.float64)
        z = z - z.max(axis=-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
        gt = np.asarray(gt)
        tok_logp = np.take_along_axis(logp, gt[..., None], axis=-1).squeeze(-1)
        return {"perplexity": float(np.exp(-tok_logp.mean()))}


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Task registry                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════╝
TASK_REGISTRY: Dict[str, type] = {
    "classification": ClassificationTask,
    # "language_modeling": LanguageModelingTask,   # enable when LLM path lands
}


def register_task(name: str, cls: type) -> None:
    TASK_REGISTRY[name.strip().lower()] = cls


def get_task(name: str, **kwargs) -> Task:
    key = (name or "classification").strip().lower()
    if key not in TASK_REGISTRY:
        raise KeyError(
            f"unknown accuracy task '{name}'. Registered: "
            f"{sorted(TASK_REGISTRY)}. Add one with metrics.register_task(...).")
    return TASK_REGISTRY[key](**kwargs)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Accumulator + report                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════╝
class AccuracyAccumulator:
    """Accumulate (torch_output, imc_output, gt) samples → accuracy report.

    GT is optional: when absent the Torch-vs-GT / IMC-vs-GT figures are skipped
    and only the SW-vs-HW agreement is reported.
    """

    def __init__(self, task: Task):
        self.task = task
        self.n = 0
        self.n_gt = 0
        self.torch_correct = 0
        self.imc_correct = 0
        self.agree = 0
        self._extra_sum: Dict[str, float] = {}
        self._extra_cnt: Dict[str, int] = {}
        self.samples: List[Dict[str, Any]] = []

    def _accum_extra(self, prefix: str, d: Dict[str, float]) -> None:
        for k, v in d.items():
            key = f"{prefix}_{k}"
            self._extra_sum[key] = self._extra_sum.get(key, 0.0) + float(v)
            self._extra_cnt[key] = self._extra_cnt.get(key, 0) + 1

    def add(self, torch_output: np.ndarray, imc_output: np.ndarray,
            gt: Any = None, name: Optional[str] = None) -> None:
        self.n += 1
        t_pred = self.task.predict(torch_output)
        i_pred = self.task.predict(imc_output)
        agree = bool(np.all(self.task.agree(torch_output, imc_output)))
        self.agree += int(agree)

        t_ok = i_ok = None
        if self.task.has_gt(gt):
            self.n_gt += 1
            t_ok = bool(np.all(self.task.is_correct(torch_output, gt)))
            i_ok = bool(np.all(self.task.is_correct(imc_output, gt)))
            self.torch_correct += int(t_ok)
            self.imc_correct += int(i_ok)
            self._accum_extra("torch", self.task.sample_extra(torch_output, gt))
            self._accum_extra("imc", self.task.sample_extra(imc_output, gt))
        else:
            self._accum_extra("torch", self.task.sample_extra(torch_output, None))
            self._accum_extra("imc", self.task.sample_extra(imc_output, None))

        self.samples.append({
            "name": name,
            "torch_pred": _scalar(t_pred), "imc_pred": _scalar(i_pred),
            "gt": _scalar(gt) if gt is not None else None,
            "agree": agree, "torch_correct": t_ok, "imc_correct": i_ok,
        })

    def report(self) -> Dict[str, Any]:
        rep: Dict[str, Any] = {
            "task": self.task.name,
            "n_samples": self.n,
            "n_with_gt": self.n_gt,
            "sw_hw_agreement_pct": _pct(self.agree, self.n),
        }
        if self.n_gt > 0:
            t = _pct(self.torch_correct, self.n_gt)
            i = _pct(self.imc_correct, self.n_gt)
            rep.update({
                "torch_vs_gt_acc_pct": t,
                "imc_vs_gt_acc_pct": i,
                "accuracy_drop_pct": (t - i) if (t is not None and i is not None) else None,
            })
        extra = {k: self._extra_sum[k] / self._extra_cnt[k]
                 for k in self._extra_sum if self._extra_cnt[k] > 0}
        if extra:
            rep["extra"] = extra
        return rep

    def format(self) -> str:
        r = self.report()
        L = ["=" * 60, "  ACCURACY REPORT", "=" * 60,
             f"  task             : {r['task']}",
             f"  samples          : {r['n_samples']}  (with GT: {r['n_with_gt']})"]
        if "torch_vs_gt_acc_pct" in r:
            L += [f"  Torch vs GT      : {_fmt_pct(r['torch_vs_gt_acc_pct'])}",
                  f"  IMC   vs GT      : {_fmt_pct(r['imc_vs_gt_acc_pct'])}",
                  f"  accuracy drop    : {_fmt_pct(r['accuracy_drop_pct'])} (Torch - IMC)"]
        else:
            L.append("  (no ground-truth labels provided → GT accuracy skipped)")
        L.append(f"  SW–HW agreement  : {_fmt_pct(r['sw_hw_agreement_pct'])}")
        if "extra" in r:
            L.append("  " + "-" * 56)
            for k, v in r["extra"].items():
                L.append(f"  {k:17s}: {v:.4f}")
        L.append("=" * 60)
        return "\n".join(L)

    def write_json(self, path: str, include_samples: bool = True) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        out = self.report()
        if include_samples:
            out["samples"] = self.samples
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Ground-truth label loading (config-driven, optional)                 ║
# ╚═══════════════════════════════════════════════════════════════════════╝
def load_labels(
    names: Sequence[str], *,
    labels_file: Optional[str] = None,
    label_from_filename: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Resolve ground-truth labels for image files, returning ``{name: label}``
    (missing entries simply absent).  Sources, in priority order:

      1. ``labels_file`` — a CSV ``filename,label`` (header optional) or a JSON
         ``{"filename": label}``.  Labels may be ints or class-name strings; if
         strings and ``class_names`` is given they're mapped to indices.
      2. ``label_from_filename`` — a regex applied to each basename whose first
         capture group is the label (int, or class name → index via class_names).

    Returns an empty dict if neither source is configured (GT metrics skipped).
    """
    name2label: Dict[str, Any] = {}

    def _to_idx(v):
        if class_names and isinstance(v, str) and not v.lstrip("-").isdigit():
            return class_names.index(v) if v in class_names else None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    if labels_file and os.path.exists(labels_file):
        if labels_file.lower().endswith(".json"):
            with open(labels_file) as f:
                raw = json.load(f)
            for k, v in raw.items():
                idx = _to_idx(v)
                if idx is not None:
                    name2label[os.path.basename(k)] = idx
        else:
            with open(labels_file) as f:
                for ln in f:
                    parts = [p.strip() for p in ln.replace("\t", ",").split(",")]
                    if len(parts) < 2 or parts[0].lower() in ("filename", "name", "file"):
                        continue
                    idx = _to_idx(parts[1])
                    if idx is not None:
                        name2label[os.path.basename(parts[0])] = idx

    if label_from_filename:
        rgx = re.compile(label_from_filename)
        for nm in names:
            base = os.path.basename(nm)
            if base in name2label:
                continue
            m = rgx.search(base)
            if m and m.groups():
                idx = _to_idx(m.group(1))
                if idx is not None:
                    name2label[base] = idx

    return name2label


# ── small helpers ───────────────────────────────────────────────────────────
def _pct(num: int, den: int) -> Optional[float]:
    return None if den <= 0 else 100.0 * num / den


def _fmt_pct(v: Optional[float]) -> str:
    return "  n/a" if v is None else f"{v:6.2f}%"


def _scalar(x):
    a = np.asarray(x)
    return int(a.ravel()[0]) if a.size == 1 else a.ravel().tolist()


__all__ = [
    "Task", "ClassificationTask", "LanguageModelingTask",
    "TASK_REGISTRY", "register_task", "get_task",
    "AccuracyAccumulator", "load_labels",
]