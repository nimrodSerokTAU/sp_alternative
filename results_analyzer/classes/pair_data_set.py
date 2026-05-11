import math

import numpy as np

from results_analyzer.classes.scatter_plot import ScatterPlot
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.measurement_data import MeasurementDataPerCode
from results_analyzer.constants import COLORS, NAMING
from results_analyzer.utils import normalize


class PairDataSet:
    code: str
    measure_names: [str, str]
    r_vals: [float, float]

    def __init__(self, code: str):
        self.code = code
        self.measure_names = []
        self.r_vals = []

    def append_sample(self, measure_name: str,  r_val: float):
        self.measure_names.append(measure_name)
        self.r_vals.append(r_val)


class CorrCompareTwo:
    data: list[PairDataSet]
    measure_keys: list[str]

    def __init__(self, measure_keys: list[str]):
        self.data = []
        self.measure_keys = measure_keys

    def append_sample(self, sample: PairDataSet):
        self.data.append(sample)

    def create_scatter(self) -> ScatterPlot:
        x = np.array([x.r_vals[0] for x in self.data])
        y = [y.r_vals[1] for y in self.data]
        markers = ['o', '^', 's']
        names = [NAMING[self.measure_keys[0]], NAMING[self.measure_keys[1]]]
        sp = ScatterPlot(x, markers, 0.8, 0.7, 14, 'black', '--',
                         0.6, f"{names[0]} r Value", f"{names[1]} r Value", 'lower right',
                         True, False, False)


        colors = ['#e0690d']

        sp.set_data([y], names, colors, None, None, 0, 1.1,
                    0, 1.1, None, len(self.data))

        return sp
