import pandas as pd

from results_analyzer.classes.measure import Measure


class PDFPlot:
    true_scores: pd.Series
    x: list[float]
    y: list[float]
    measures: list[Measure]
    colors: list[str]
    markers: list[str]
    labels: list[str]
    s: int
    xlabel: str
    ylabel: str

    def __init__(self, true_scores: pd.Series, x: list[float], y: list[float], markers: list[str], colors: list[str],
                 labels: list[str], s: int, xlabel: str, ylabel: str):
        self.x = x
        self.y = y
        self.true_scores = true_scores
        self.colors = colors
        self.labels = labels
        self.markers = markers
        self.s = s
        self.xlabel = xlabel
        self.ylabel = ylabel





