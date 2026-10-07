"""Per-request system telemetry: latency, tokens, cost, cache hits, errors.

Kept in memory (bounded) and appended to data/telemetry/requests.jsonl.
No question text is stored, only a short hash.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

_LOCK = threading.Lock()
_RECENT: Deque[Dict] = deque(maxlen=2000)


def record(data_dir: Path, *, question: str, latency_s: float, tokens: int, llm_calls: int, provider: str,
           cached: bool, error: bool, kind: str, cost_per_1k: float = 0.0, rate_limited: bool = False) -> Dict:
    row = {"ts": round(time.time(), 3), "q": hashlib.sha256(question.encode()).hexdigest()[:10],
           "latency_s": round(latency_s, 3), "tokens": int(tokens), "llm_calls": int(llm_calls),
           "provider": provider or "", "cached": cached, "error": error, "kind": kind,
           "rate_limited": rate_limited, "cost": round(tokens / 1000 * cost_per_1k, 6)}
    with _LOCK:
        _RECENT.append(row)
        try:
            path = Path(data_dir) / "telemetry" / "requests.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            pass
    return row


def _pct(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


def summary(rows: Optional[List[Dict]] = None) -> Dict:
    rows = list(_RECENT) if rows is None else rows
    served = [r for r in rows if not r.get("rate_limited")]
    lat = [r["latency_s"] for r in served if not r["error"]]
    n = len(served)
    return {"requests": len(rows), "p50_latency_s": _pct(lat, 0.50), "p95_latency_s": _pct(lat, 0.95),
            "avg_tokens": round(sum(r["tokens"] for r in served) / n, 1) if n else None,
            "total_tokens": sum(r["tokens"] for r in served),
            "cost_per_request": round(sum(r["cost"] for r in served) / n, 6) if n else None,
            "cache_hit_rate": round(sum(r["cached"] for r in served) / n, 3) if n else None,
            "error_rate": round(sum(r["error"] for r in served) / n, 3) if n else None,
            "rate_limited": sum(1 for r in rows if r.get("rate_limited"))}


def reset() -> None:
    with _LOCK:
        _RECENT.clear()
