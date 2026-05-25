
import pandas as pd
from pathlib import Path


def merge_large_csv_files(input_folder: str, output_file: str) -> None:
    """
    Merge all CSV files in input_folder into output_file.
    Appends rows and writes the header only once.
    Designed for large files to avoid holding everything in memory.
    """
    first = True

    for file in Path(input_folder).glob("*.csv"):
        df = pd.read_csv(file)

        # Add filename (without extension) as new column
        df["code1"] = file.stem

        df.to_csv(
            output_file,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        first = False

merge_large_csv_files(input_folder='<use your path here>',
                      output_file='<use your path here>')
