"""How often finer-grained SQL types would apply (Section 7, taxonomy limits).

The ground truth has only the four coarse types, so candidates for finer types
are found with transparent name patterns and reported per coarse label.

Usage:  python analysis/finer_types.py
"""

import re

from common import TI, scored

PATTERNS = {
    "timestamp": r"(timestamp|datetime|date_?time|датум_?и_?време)",
    "time_interval": r"(време|time|период|period|интервал|interval|траење|трање|duration|термин)",
    "boolean": r"(^(is|has)_|^дали|^dali|платен|platen|активен|aktiven)",
}

A = scored().copy()
A["cyr"] = A["attribute"].map(lambda a: TI.latin_mk_to_cyrillic(a.strip().lower()))
taken = None
for name, rx in PATTERNS.items():  # timestamp first, so its names are counted once
    r = re.compile(rx, re.IGNORECASE)
    m = A.apply(lambda x: bool(r.search(x["attribute"].lower()) or r.search(x["cyr"])), axis=1)
    m = m if taken is None else (m & ~taken)
    taken = m if taken is None else (taken | m)
    x = A[m]
    print(f"== {name}")
    for split in ("dev", "test"):
        s = x[x.split == split]
        total = (A.split == split).sum()
        print(f"   {split}: {len(s)} ({100 * len(s) / total:.2f}% of scored), "
              f"coarse labels {s['label'].value_counts().to_dict()}, "
              f"top {s.groupby('attribute').size().sort_values(ascending=False).head(5).to_dict()}")
