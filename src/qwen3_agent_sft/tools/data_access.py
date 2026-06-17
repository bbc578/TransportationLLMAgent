from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/sample_business")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)
