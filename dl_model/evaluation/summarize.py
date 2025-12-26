from __future__ import annotations
from collections import defaultdict
import pandas as pd
from constants import GROUP_COL, ALIGNER_COL


def summarize_data(df: pd.DataFrame) -> dict:
    summary = {}
    for taxa_num in df["taxa_num"].unique():
        taxa_df = df[df["taxa_num"] == taxa_num]
        code1_dict = defaultdict(dict)

        for code1 in taxa_df[GROUP_COL].unique():
            sub_df = taxa_df[taxa_df[GROUP_COL] == code1]
            aligner_counts = sub_df.groupby(ALIGNER_COL)["code"].nunique().to_dict()
            code1_dict[code1] = aligner_counts

        summary[taxa_num] = {"unique_code1_count": taxa_df[GROUP_COL].nunique()}
    return summary
