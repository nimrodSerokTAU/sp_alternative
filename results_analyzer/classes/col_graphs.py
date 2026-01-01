import pandas as pd

from results_analyzer.classes.col_plot import ColPlot
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.measurement_data import MeasurementDataPerCode
from results_analyzer.constants import COLORS, NAMING, HATCHS

DIRECTIONAL_THRESHOLDS = [-1, 0.5, 0.7, 0.85, 0.95, 1]
NAMES = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']

class DataGroupCorr:
    name: str
    samples: dict[str, list[MeasurementDataPerCode]]
    min_val: float
    max_val: float
    description: str

    def __init__(self, name: str, min_val: float, max_val: float, measure_keys: list[str]):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.samples = {}
        for key in measure_keys:
            self.samples[key] = []

    def append_one_sample(self, sample: MeasurementDataPerCode):
        self.samples[sample.measure_key].append(sample)

class DataGroupMgr:
    data_groups: list[DataGroupCorr]
    names: list[str]
    directional_thresholds: list[float]
    measures_directions: dict[str, bool]
    dataset_counter: int

    def __init__(self, measures: list[Measure]):
        self.names = NAMES
        self.directional_thresholds = DIRECTIONAL_THRESHOLDS
        self.data_groups = []
        self.create_data_groups([x.key for x in measures])
        self.measures_directions = {}


    def create_data_groups(self, measure_keys: list[str]):
        for i in range(len(NAMES)):
            self.data_groups.append(DataGroupCorr(self.names[i], self.directional_thresholds[i],
                                                  self.directional_thresholds[i + 1], measure_keys))

    def set_measure_direction(self, measure_key: str, is_m_direction_positive: bool):
        self.measures_directions[measure_key] = is_m_direction_positive

    def add_sample(self, sample: MeasurementDataPerCode):
        r_value = sample.r_value if self.measures_directions[sample.measure_key] else -sample.r_value
        for d_group in self.data_groups:
            if d_group.min_val < r_value <= d_group.max_val:
                d_group.append_one_sample(sample)
                break

    def set_dataset_counter(self, dataset_counter: int):
        self.dataset_counter = dataset_counter

    def print_r(self, m_keys: list[str], multi_dataset_title: str) -> ColPlot:
        X = [x.name for x in self.data_groups]
        Y = []
        sum_per_measure: dict[str, int] = dict()
        for m_key in m_keys:
            sum_per_measure[m_key] = 0
        for dg in self.data_groups:
            for m_key in m_keys:
                sum_per_measure[m_key] += len(dg.samples[m_key])
        for dg in self.data_groups:
            dg_data = []
            for m_key in m_keys:
                dg_data.append(len(dg.samples[m_key]) / sum_per_measure[m_key])
            Y.append(dg_data)

        colors = [COLORS[key] for key in m_keys]
        hatches = [HATCHS[key] for key in m_keys]

        df = pd.DataFrame(Y, index=X)
        cl = ColPlot(df, 0.8, colors, hatches, [NAMING[key] for key in m_keys], multi_dataset_title,
                     'Datasets Density', self.dataset_counter)
        return cl

