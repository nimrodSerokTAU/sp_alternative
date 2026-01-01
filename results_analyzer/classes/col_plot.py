import pandas as pd

class ColPlot:
    df: pd.DataFrame
    m_keys: list[str]
    color: list[str]
    hatch: list[str]
    xlabel: str
    ylabel: str
    alpha: float
    data_count: int

    def __init__(self, df: pd.DataFrame, alpha: float, color: list[str], hatch: list[str], names: list[str],
                 xlabel: str, ylabel: str, data_count: int):
        self.df = df
        self.color = color
        self.hatch = hatch
        self.alpha = alpha
        self.names = names
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.data_count = data_count




