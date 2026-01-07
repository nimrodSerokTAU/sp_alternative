import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from results_analyzer.classes.measure import Measure
from results_analyzer.constants import COLORS, NAMING, HATCHS


class PBDataSet:
    code: str
    winner: str

    def __init__(self, code: str, winner: str):
        self.code = code
        self.winner = winner

class DataGroup:
    measure_key: str
    samples: list[PBDataSet]
    percent: float

    def __init__(self, measure_key: str):
        self.measure_key = measure_key
        self.samples = []

    def append_one_sample(self, sample: PBDataSet):
        self.samples.append(sample)

class DataSource:
    name: str
    file_path: str
    index_in_file: int
    override_key: str
    data_groups: dict[str, DataGroup]
    external_to_key_dict: dict[str, str]
    total_count: int

    def __init__(self, name: str, file_path: str, override_key, index_in_file: int, measures: list[Measure]):
        self.name = name
        self.file_path = file_path
        self.index_in_file = index_in_file
        self.override_key = override_key
        self.total_count = 0
        self.data_groups = dict()
        self.external_to_key_dict = dict()
        for measure in measures:
            self.data_groups[measure.key] = DataGroup(measure.key)
            self.external_to_key_dict[measure.external_name] = measure.key


    def calc_results(self):
        for ds in self.data_groups.values():
            ds.percent = len(ds.samples) / self.total_count

    def calc_chi_square_df_1(self, keys:[str, str]) -> float:
        sample_a_count = len(self.data_groups[keys[0]].samples)
        sample_b_count = len(self.data_groups[keys[1]].samples)
        observed_frequencies: [int, int] = [sample_a_count, sample_b_count]
        equal_hypothesis = (sample_a_count + sample_b_count) / 2
        expected_frequencies: [float, float] = [equal_hypothesis, equal_hypothesis]
        chi2_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)
        return p_value


class StackedColGraphData:
    x: np.array
    data_by_labels: list[float]
    width: float
    bottom: np.array
    color: list[str]
    label: list[str]
    hatch: list[str]

    def __init__(self, x, data_by_labels: list[float], width: float, bottom: np.array, color: list[str], label: list[str], hatch: list[str]):
        self.x = x
        self.data_by_labels = data_by_labels
        self.width = width
        self.bottom = bottom
        self.color = color
        self.label = label
        self.hatch = hatch

class StackedColSubPlot:
    data: list[StackedColGraphData]
    data_sources: list[DataSource]
    ylabel: str # 'Dataset Percentage'
    categories: list[str]
    samples_num: int
    labels_list: list[dict]

    def __init__(self, data: list[StackedColGraphData], data_sources: list[DataSource], ylabel: str,
                 categories: list[str], samples_num: int, labels_list: list[dict], p_value_per_ds: list[float] | None):
        self.data = data
        self.data_sources = data_sources
        self.ylabel = ylabel
        self.categories = categories
        self.samples_num = samples_num
        self.labels_list = labels_list
        self.p_value_per_ds=p_value_per_ds


class StackedColGraph:
    data_sources: list[DataSource]
    categories: list[str]
    measures: list[Measure]
    subplot: StackedColSubPlot

    def __init__(self, dir_path: str, files_data: list[dict], input_data_sources: list[Measure], measures: list[Measure],
                 is_calc_chi_square: bool, keys:[str, str]):
        self.data_sources = []
        self.categories = []
        self.measures = measures
        data_sources_per_file: dict[str, list[DataSource]] = dict()
        for file_data in files_data:
            relative_file_path = file_data['relative_file_path']
            data_sources_per_file[file_data['series_name']] = []
            for i in range(len(input_data_sources)):
                ds_name = input_data_sources[i].presentation_name if len(files_data) == 1 else file_data['series_name']
                file_path = f'{dir_path}/{relative_file_path}'
                self.categories.append(ds_name)
                override_key = input_data_sources[i].key
                data_source = DataSource(ds_name, file_path, override_key, i + 1, measures)
                self.data_sources.append(data_source)
                data_sources_per_file[file_data['series_name']].append(data_source)

        self.read_files(data_sources_per_file)
        for ds in self.data_sources:
            ds.calc_results()
        p_value_per_ds: list[float] | None = self.calc_chi_square_for_two(is_calc_chi_square, keys)
        self.subplot = self.create_col_graph(p_value_per_ds)

    def read_files(self, data_sources_per_file: dict[str, list[DataSource]]):
        for data_sources in data_sources_per_file.values():
            self.read_file(data_sources)

    @staticmethod
    def read_file(data_sources: list[DataSource]):
        with open(data_sources[0].file_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            print(f"Header: {header}")
            for i, row in enumerate(csv_reader):
                dataset_key = row[0]
                for ds in data_sources:
                    if row[ds.index_in_file] and row[ds.index_in_file] != '':
                        res_key = ds.external_to_key_dict[row[ds.index_in_file]]
                        dataset_result = PBDataSet(dataset_key, res_key)
                        ds.data_groups[res_key].append_one_sample(dataset_result)
                        ds.total_count += 1

    def calc_chi_square_for_two(self, is_calc_chi_square: bool, keys:[str, str]) -> list[float] | None:
        if not is_calc_chi_square:
            return None
        return [data_source.calc_chi_square_df_1(keys) for data_source in self.data_sources]

    def create_col_graph(self, p_value_per_ds: list[float] | None) -> StackedColSubPlot:
        labels = []
        colors = []
        hatches = []
        data_by_labels = []
        labels_list = []

        samples_num: int = self.data_sources[0].total_count
        for i in range(len(self.measures)):
            labels.append(self.measures[i].presentation_name)
            colors.append(COLORS[self.measures[i].key])
            hatches.append(HATCHS[self.measures[i].key])
            data = []
            for d_source in self.data_sources:
                y = d_source.data_groups[self.measures[i].key].percent * 100
                data.append(y)
            data_by_labels.append(data)

        x = np.arange(len(self.categories))

        bottom = [0 for i in range(len(self.data_sources))]
        bottom = np.array(bottom)

        data_res: list[StackedColGraphData] = []
        for i in range(len(data_by_labels)):
            p_labels = [f'{NAMING[x.override_key]} Default' for x in self.data_sources] if i == 0 and labels[0] == 'Default' else labels[i]
            p_colors = [COLORS[x.override_key] for x in self.data_sources] if i == 0 and labels[0] == 'Default' else  colors[i]
            p_hatches = [HATCHS[x.override_key] for x in self.data_sources] if i == 0 and labels[
                0] == 'Default' else hatches[i]
            data_res.append(StackedColGraphData(x, data_by_labels[i], width=0.6, bottom=bottom, label=p_labels,
                                         color=p_colors, hatch=p_hatches))
            labels_list.append({'x':x, 'bottom':bottom, 'val':data_by_labels[i]})
            bottom = np.array(bottom) + np.array(data_by_labels[i])

        res = StackedColSubPlot(data=data_res, data_sources=self.data_sources, ylabel='Dataset Percentage',
                                categories=self.categories, samples_num=samples_num, labels_list=labels_list,
                                p_value_per_ds=p_value_per_ds)
        return res


def add_labels(x, h: list[float], values: list[float]):
    for i in range(len(x)):
        this_h = h[i] + values[i] * 0.75
        plt.text(i, this_h, f'{values[i]:.1f}%', ha='center', va='top')

def double_plot(data_by_labels: list[StackedColSubPlot], dir_path: str, identifier: str):

    # plt.figure(figsize=(26, 7), layout='constrained')
    fig, axs = plt.subplots(1, 2, figsize=(15, 7)) # len(data_by_labels)
    plt.rcParams['hatch.linewidth'] = 0.5
    plt.rcParams['hatch.color'] = '#404245'

    for j in range(len(data_by_labels)):
        plot_data = data_by_labels[j]
        for i in range(len(plot_data.data)):
            stacked_col_data: StackedColGraphData = plot_data.data[i]
            label = stacked_col_data.label
            if label == 'Model2':
                label = 'DL-model'
            if label == 'Tie (Model2 and Default)':
                label = 'Tie (DL-model and Default)'
            axs[j].bar(list(stacked_col_data.x), stacked_col_data.data_by_labels, width=0.6, bottom=list(stacked_col_data.bottom),
                          label=label, color=stacked_col_data.color, hatch=stacked_col_data.hatch)
            add_comp_labels(axs[j], plot_data.labels_list[i]['x'], plot_data.labels_list[i]['bottom'],  plot_data.labels_list[i]['val'])
            axs[j].set_ylabel(plot_data.ylabel, fontsize=12)
            axs[j].set_xticks(stacked_col_data.x, plot_data.categories, fontsize=12)

    handles, labels = [], []
    h, l = axs[0].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    fig.legend(handles, labels, ncol=3, loc='upper center', fontsize=11, bbox_to_anchor=(0.5, 0.1)) #loc='upper left',
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'{dir_path}/{identifier}_pick_best_{data_by_labels[0].samples_num}_emp_{data_by_labels[1].samples_num}_sim_datasets.tiff')
    plt.show()


def add_comp_labels(fig, x, h: list[float], values: list[float]):
    for i in range(len(x)):
        this_h = h[i] + values[i] * 0.75
        fig.text(i, this_h, f'{values[i]:.1f}%', ha='center', va='top', fontsize=12,
                 # fontweight='bold'
                 )


