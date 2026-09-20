"""Reproduce the paper's main numbers from the released predictions.

Prints Table 2 (overall results), Table 3 (per-class scores), the per-tier
accuracies, and the per-language accuracies of Section 7.

Usage:  python analysis/reproduce_tables.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from common import LABELS, scored

METHODS = ["majority", "regex", "llm_3b", "llm_7b", "hybrid_3b", "hybrid_7b"]

for split in ("dev", "test"):
    d = scored(split)
    print(f"\n=== {split}: {len(d)} scored attributes, "
          f"regex coverage {100 * (d.tier == 'regex').mean():.2f}%")

    rows = []
    for m in METHODS:
        pred = pd.Series("text", index=d.index) if m == "majority" else d[m]
        rows.append({"method": m,
                     "accuracy": round(100 * accuracy_score(d["label"], pred), 2),
                     "macro_f1": round(f1_score(d["label"], pred, average="macro"), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    p, r, f, _ = precision_recall_fscore_support(d["label"], d["hybrid_3b"], labels=LABELS)
    print(pd.DataFrame({"type": LABELS, "P": p.round(2), "R": r.round(2), "F1": f.round(2)})
          .to_string(index=False))

    for tier in ("regex", "llm"):
        x = d[d.tier == tier]
        print(f"  {tier} tier: {(x['label'] == x.hybrid_3b).sum()}/{len(x)} "
              f"({100 * (x['label'] == x.hybrid_3b).mean():.2f}%)")
    for lang, x in d.groupby("language"):
        print(f"  {lang}: {(x['label'] == x.hybrid_3b).sum()}/{len(x)} "
              f"({100 * (x['label'] == x.hybrid_3b).mean():.2f}%)")
