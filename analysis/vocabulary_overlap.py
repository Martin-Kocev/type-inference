"""Vocabulary overlap between the two datasets, and the NUMERIC failure analysis
(Sections 5.2 and 6.2 of the paper).

Usage:  python analysis/vocabulary_overlap.py
"""

import pandas as pd

from common import LABELS, name_key, scored

A = scored().copy()
A["key"] = A["attribute"].map(name_key)
dev, test = A[A.split == "dev"], A[A.split == "test"].copy()
dev_keys = set(dev["key"])
test["seen"] = test["key"].isin(dev_keys)

print(f"distinct names: dev {dev['key'].nunique()}, test {test['key'].nunique()}, "
      f"shared {len(dev_keys & set(test['key']))} "
      f"({100 * len(dev_keys & set(test['key'])) / test['key'].nunique():.1f}% of test names)")
print(f"test attributes with a name seen in development: {100 * test['seen'].mean():.1f}%")

ski = set(dev.loc[dev.domain.str.startswith('ski'), "key"])
nonski = set(dev.loc[~dev.domain.str.startswith('ski'), "key"])
print(f"  seen in non-ski development domains: {100 * test['key'].isin(nonski).mean():.1f}%")
print(f"  seen only in ski development domains: "
      f"{100 * (test['key'].isin(ski) & ~test['key'].isin(nonski)).mean():.1f}%")
for t in LABELS:
    x = test[test.label == t]
    print(f"  {t:8s} n={len(x):4d} seen={100 * x['seen'].mean():.1f}%")

for seen, x in test.groupby("seen"):
    print(f"accuracy on {'seen' if seen else 'unseen'} names: "
          f"{100 * (x['label'] == x.hybrid_3b).mean():.2f}% (n={len(x)}, "
          f"regex coverage {100 * (x.tier == 'regex').mean():.1f}%)")

fail = test[(test.label == "numeric") & (test.hybrid_3b == "text")]
print(f"\nNUMERIC -> TEXT on the unseen test set: {len(fail)} of "
      f"{(test.label == 'numeric').sum()} attributes, {fail['key'].nunique()} distinct names")
print(f"  by tier: {fail.tier.value_counts().to_dict()}")
print(f"  LLM answer: {fail.llm_label.value_counts().to_dict()}, "
      f"confidence: {fail.llm_conf.value_counts().to_dict()}")
print(f"  names unseen during development: {(~fail['seen']).sum()} of {len(fail)}")
print(f"  by domain: {fail.domain.value_counts().to_dict()}")
print(pd.DataFrame(fail.groupby("key").agg(example=("attribute", "first"), n=("attribute", "size"))
                   .sort_values("n", ascending=False)).to_string())
