"""Experiment history: append-only JSON Lines file. The dashboard reads real records from it."""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DEFAULT_PATH = Path("runs") / "experiment_history.jsonl"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class HistoryStore:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)

    def append(self, record: Dict) -> Dict:
        rec = _jsonable(dict(record))
        rec["id"] = uuid.uuid4().hex[:8]
        rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    def load(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out: List[Dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue    # skip a corrupted line rather than crash the dashboard
        return out

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def summary_stats(records: List[Dict]) -> Dict:
    """Real statistics computed from stored experiments (nothing invented)."""
    phys = [r for r in records if r.get("kind") == "physics"]
    if not phys:
        return {"n_experiments": 0, "n_records": len(records), "by_system": {},
                "median_val_r2": None, "best_val_r2": None, "reached_ceiling": 0,
                "total_runtime_s": 0.0}
    val_r2 = [r["validation"]["r2"] for r in phys if r.get("validation", {}).get("r2") is not None]
    by_system: Dict[str, int] = {}
    for r in phys:
        by_system[r["system"]] = by_system.get(r["system"], 0) + 1
    reached = sum(
        1 for r in phys
        if r.get("noise_ceiling_r2") is not None and r["validation"]["r2"] is not None
        and r["validation"]["r2"] >= r["noise_ceiling_r2"] - 0.01
    )
    return {
        "n_experiments": len(phys),
        "n_records": len(records),
        "by_system": by_system,
        "median_val_r2": float(np.median(val_r2)) if val_r2 else None,
        "best_val_r2": float(np.max(val_r2)) if val_r2 else None,
        "reached_ceiling": reached,
        "total_runtime_s": float(sum(r.get("elapsed_seconds") or 0.0 for r in phys)),
    }
