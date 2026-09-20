"""Supervised name-only baselines (Section 5.6 and Table 2).

Character n-gram TF-IDF + logistic regression / linear SVM, trained on the
development set and evaluated once on the unseen test set, alone and as the
fallback in place of the LLM tier.  Because 54% of test attributes have a name
that also occurs in the development data, results are also split into seen and
unseen names, and development cross-validation uses name-disjoint folds.

Usage:  python analysis/ml_baselines.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import TI, name_key, scored


def text_of(name):
    n = str(name).strip().lower()
    return f"{n} {TI.latin_mk_to_cyrillic(n)}"


def fit(kind, X, y):
    if kind == "tfidf-lr":
        clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
                            LogisticRegression(C=10.0, max_iter=5000, class_weight="balanced"))
    else:
        clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True),
                            LinearSVC(C=1.0, class_weight="balanced"))
    clf.fit(X, y)
    return clf.predict


def main():
    dev, test = scored("dev").reset_index(drop=True), scored("test").reset_index(drop=True)
    Xd, Xt = dev["attribute"].map(text_of).values, test["attribute"].map(text_of).values
    yd, yt = dev["label"].values, test["label"].values
    test_llm = (test["tier"] == "llm").values
    seen = test["attribute"].map(name_key).isin(set(dev["attribute"].map(name_key))).values

    rows = [{"model": "hybrid 3B LLM (paper)",
             "test": round(100 * accuracy_score(yt, test.hybrid_3b), 2),
             "macro F1": round(f1_score(yt, test.hybrid_3b, average="macro"), 3),
             "seen names": round(100 * accuracy_score(yt[seen], test.hybrid_3b[seen]), 2),
             "unseen names": round(100 * accuracy_score(yt[~seen], test.hybrid_3b[~seen]), 2),
             "unseen & no rule": round(100 * accuracy_score(yt[~seen & test_llm],
                                                            test.hybrid_3b[~seen & test_llm]), 2),
             "dev CV (name-disjoint)": None}]

    for kind in ("tfidf-lr", "tfidf-svm"):
        cv = np.empty(len(dev), dtype=object)
        for tr, va in GroupKFold(n_splits=5).split(Xd, groups=dev["attribute"].map(name_key)):
            cv[va] = fit(kind, Xd[tr], yd[tr])(Xd[va])
        p = fit(kind, Xd, yd)(Xt)
        h = np.where(test_llm, p, test["regex"].values)
        rows.append({"model": kind, "test": round(100 * accuracy_score(yt, p), 2),
                     "macro F1": round(f1_score(yt, p, average="macro"), 3),
                     "seen names": round(100 * accuracy_score(yt[seen], p[seen]), 2),
                     "unseen names": round(100 * accuracy_score(yt[~seen], p[~seen]), 2),
                     "unseen & no rule": round(100 * accuracy_score(yt[~seen & test_llm],
                                                                    p[~seen & test_llm]), 2),
                     "dev CV (name-disjoint)": round(100 * accuracy_score(yd, cv), 2)})
        rows.append({"model": f"regex + {kind}", "test": round(100 * accuracy_score(yt, h), 2),
                     "macro F1": round(f1_score(yt, h, average="macro"), 3)})

    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
