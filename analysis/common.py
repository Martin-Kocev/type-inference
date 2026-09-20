"""Loading helpers shared by the analysis scripts."""

import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import type_inference as TI  # noqa: E402

LABELS = ["date", "integer", "numeric", "text"]


@lru_cache(maxsize=1)
def attributes() -> pd.DataFrame:
    """All extracted attributes, in the original processing order."""
    return pd.read_csv(ROOT / "data" / "attribute_predictions.csv", keep_default_na=False,
                       na_values=[""], dtype={"attribute": str})


def scored(split: str | None = None) -> pd.DataFrame:
    """Attributes subject to type inference: foreign keys excluded (see Section 6)."""
    df = attributes()
    df = df[df["tier"] != "fk"]
    return df if split is None else df[df["split"] == split]


def name_key(name: str) -> str:
    """Script- and separator-independent key: transliterate to Cyrillic, drop separators."""
    return re.sub(r"[\W_]+", "", TI.latin_mk_to_cyrillic(str(name).strip().lower()))
