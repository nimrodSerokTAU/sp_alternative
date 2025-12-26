from __future__ import annotations
from pathlib import Path
import pandas as pd

def check_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".parquet":
        return "parquet"
    if ext == ".csv":
        return "csv"
    return "unknown"


def read_features(features_file: str) -> pd.DataFrame:
    ft = check_file_type(features_file)
    if ft == "parquet":
        df = pd.read_parquet(features_file, engine="pyarrow")
    elif ft == "csv":
        df = pd.read_csv(features_file)
    else:
        raise ValueError(f"Unknown file type: {features_file}")

    df["code1"] = df["code1"].astype(str)
    df["code"] = df["code"].astype(str)
    return df
