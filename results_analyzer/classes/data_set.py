import numpy as np

from results_analyzer.classes.scatter_plot import ScatterPlot
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.measurement_data import MeasurementDataPerCode
from results_analyzer.constants import COLORS, NAMING
from results_analyzer.utils import normalize


class Point:
    distance: float
    score: float
    def __init__(self, distance, score):
        self.distance = distance
        self.score = score


class DataSet:
    code: str
    dist: list[float]
    measurementsData: list[MeasurementDataPerCode]


    def __init__(self, code: str, measures: list[Measure]):
        self.code = code
        self.dist = []
        self.measurementsData = []
        for measure in measures:
            self.measurementsData.append(MeasurementDataPerCode(measure.key, self.code))


    def append_sample(self, dist: str | float, scores: list[str | float]):
        if dist != '':
            self.dist.append(float(dist))
            for score_i in range(len(scores)):
                self.measurementsData[score_i].append_single_msa_score(scores[score_i])


    def normalize(self):
        for val in self.measurementsData:
            val.ordered_scores = normalize(val.ordered_scores)

    def calc_pearson(self, continue_missing_code: bool):
        x = np.array(self.dist)
        for val in self.measurementsData:
            val.fill_correlation(x, continue_missing_code)

    def create_scatter(self, measures: list[MeasurementDataPerCode]) -> ScatterPlot:
        x = np.array(self.dist)
        markers = ['o', '^', 's']
        sp = ScatterPlot(x, markers, 0.3, 0.7, 10, 'black', '--',
                         0.6, r'$d_{seq}$', "Score", 'lower right')
        y = []
        names = []
        colors = []
        r_vals = []
        labels = []
        for i in range(len(measures)):
            y.append(measures[i].ordered_scores)
            names.append(NAMING[measures[i].measure_key])
            r_vals.append(measures[i].r_value)
            colors.append(COLORS[measures[i].measure_key])
            labels.append(f'{names[i]} (r={r_vals[i]:.2f})')
        sp.set_data(y, names, colors, labels, r_vals, 0, 1, -0.1, 1.1, None, len(measures[0].ordered_scores))

        return sp

    def plot_zoom_scatter(self, measure: MeasurementDataPerCode, min_dist_score: float) -> ScatterPlot:
        x = np.array(self.dist)
        x_min: float = min(self.dist)
        x_max: float = max(self.dist)
        diff_x = x_max - x_min
        y_min: float = min(measure.ordered_scores)
        y_max: float = max(measure.ordered_scores)
        diff_y = y_max - y_min
        x_min -= diff_x * 0.05
        x_max += diff_x * 0.05
        y_min -= diff_y * 0.05
        y_max += diff_y * 0.05

        y = measure.ordered_scores
        name: str =  NAMING[measure.measure_key]

        sp = ScatterPlot(x, ['x'], 0.3, 0.7, 10, 'black', '--',
                          0.6, r'$d_{seq}$', "Score", None)
        sp.set_data([y], [name], [COLORS[measure.measure_key]], [f'{name} (r={measure.r_value:.2f})'],
                    [measure.r_value], x_min, x_max, y_min, y_max, min_dist_score, len(x))
        return sp


