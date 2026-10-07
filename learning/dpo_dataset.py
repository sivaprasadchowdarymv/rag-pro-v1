"""
Export accumulated feedback as a preference dataset for future DPO / RLHF.

Pairs come from:
  1. the same question answered several times (regenerate), where one
     answer was rated 👍 and another 👎;
  2. different conversations asking the same question (normalised text).

Output: JSON lines {"prompt", "chosen", "rejected", "feedback_reason"}.
No training happens here; this only prepares data.

    python -m learning.dpo_dataset data/learning/feedback.jsonl dpo.jsonl
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, List


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower()).rstrip("?. ")


def build_pairs(entries: List[Dict], max_pairs_per_prompt: int = 5) -> List[Dict]:
    by_prompt: Dict[str, Dict[str, List[Dict]]] = {}
    for e in entries:
        bucket = by_prompt.setdefault(_norm(e.get("question", "")), {"up": [], "down": []})
        bucket["up" if e.get("rating", 0) > 0 else "down"].append(e)
    pairs = []
    for prompt, b in by_prompt.items():
        if not prompt:
            continue
        for good, bad in itertools.islice(itertools.product(b["up"], b["down"]), max_pairs_per_prompt):
            if good["answer"].strip() == bad["answer"].strip():
                continue
            pairs.append({"prompt": good["question"], "chosen": good["answer"], "rejected": bad["answer"],
                          "feedback_reason": ", ".join(bad.get("reasons", [])) or bad.get("comment", "")})
    return pairs


def export(feedback_path: Path, out_path: Path) -> int:
    entries = []
    if feedback_path.exists():
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                pass
    pairs = build_pairs(entries)
    out_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + ("\n" if pairs else ""),
                        encoding="utf-8")
    return len(pairs)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/learning/feedback.jsonl")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "dpo_dataset.jsonl")
    print(f"Wrote {export(src, dst)} preference pairs to {dst}")
