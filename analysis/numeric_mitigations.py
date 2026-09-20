"""Effect of the LLM confidence threshold, replayed on the released run.

Every LLM answer of the hybrid run is logged with its self-reported
confidence, so thresholding can be replayed exactly: on the LLM tier the
prediction becomes the model's answer when its confidence reaches theta and
TEXT otherwise; all other attributes keep their logged prediction.

Usage:  python analysis/numeric_mitigations.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from common import scored

A = scored().copy()


def with_threshold(theta=0.70, theta_numeric=None):
    theta_numeric = theta if theta_numeric is None else theta_numeric
    llm = A["tier"] == "llm"
    th = A["llm_label"].map(lambda l: theta_numeric if l == "numeric" else theta)
    return A["hybrid_3b"].where(~llm, A["llm_label"].where(A["llm_conf"] >= th, "text"))


def metrics(pred, split):
    m = A["split"] == split
    y, p = A.loc[m, "label"], pred[m]
    return {"accuracy": round(100 * accuracy_score(y, p), 2),
            "macro_f1": round(f1_score(y, p, average="macro"), 3),
            "numeric_recall": round(recall_score(y, p, labels=["numeric"], average=None)[0], 3),
            "numeric_to_text": int(((y == "numeric") & (p == "text")).sum()),
            "changed_vs_original": int((p != A.loc[m, "hybrid_3b"]).sum())}


variants = {"original (theta=0.70)": with_threshold()}
for theta in (0.5, 0.8, 0.9, 0.95, 0.99):
    variants[f"theta={theta}"] = with_threshold(theta)
for theta in (0.0, 0.5, 0.9):
    variants[f"theta=0.70, theta_numeric={theta}"] = with_threshold(0.70, theta)

res = {
    "llm_confidence_values": {s: A[(A.split == s) & (A.tier == "llm")]["llm_conf"].value_counts().to_dict()
                              for s in ("dev", "test")},
    "llm_calls_with_error_or_timeout": int(A[A.tier == "llm"]["llm_reason"].astype(str)
                                           .str.contains("error|timeout", case=False).sum()),
    "test_numeric_to_text_llm_answer": A[(A.split == "test") & (A.label == "numeric") & (A.hybrid_3b == "text")]
    ["llm_label"].value_counts().to_dict(),
    "variants": {k: {s: metrics(p, s) for s in ("dev", "test")} for k, p in variants.items()},
}
print({k: res[k] for k in ("llm_confidence_values", "llm_calls_with_error_or_timeout",
                           "test_numeric_to_text_llm_answer")})
print(pd.DataFrame([{"variant": k, "split": s, **v[s]} for k, v in res["variants"].items()
                    for s in ("dev", "test")]).to_string(index=False))
