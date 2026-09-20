"""Exact McNemar tests comparing the hybrid system with the baselines.

Runs on the released per-attribute predictions; the supervised baselines are
trained on the development set exactly as in ml_baselines.py.  Attributes are
not independent -- labels were annotated per unique name and propagated to all
occurrences -- so the p-values are optimistic and are reported as a check that
a difference does not rest on a handful of attributes.

Usage:  python analysis/significance.py
"""

import numpy as np
from scipy.stats import binomtest

from common import name_key, scored
from ml_baselines import fit, text_of

dev, test = scored("dev").reset_index(drop=True), scored("test").reset_index(drop=True)
Xd, Xt = dev["attribute"].map(text_of).values, test["attribute"].map(text_of).values
yt = test["label"].values
llm = (test["tier"] == "llm").values
seen = test["attribute"].map(name_key).isin(set(dev["attribute"].map(name_key))).values

preds = {"Hybrid 7B": test["hybrid_7b"].values,
         "Regex-only": test["regex"].values,
         "LLM-only 3B": test["llm_3b"].values}
for kind in ("tfidf-lr", "tfidf-svm"):
    p = fit(kind, Xd, dev["label"].values)(Xt)
    preds[f"Hybrid {kind}"] = np.where(llm, p, test["regex"].values)
    preds[f"{kind} alone"] = p

ours = test["hybrid_3b"].values == yt
subsets = {"all (1,259)": np.ones(len(test), bool),
           "unseen names": ~seen,
           "unseen & no rule": ~seen & llm}

for label, mask in subsets.items():
    print(f"== {label}: n={int(mask.sum())}")
    for name, p in preds.items():
        other = p == yt
        b = int((ours[mask] & ~other[mask]).sum())
        c = int((~ours[mask] & other[mask]).sum())
        if b + c == 0:
            continue
        print(f"   vs {name:18s} ours-only {b:3d}  other-only {c:3d}  "
              f"p = {binomtest(b, b + c, 0.5).pvalue:.4f}")
