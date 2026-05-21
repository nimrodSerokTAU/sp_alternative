import numpy as np

from results_analyzer.classes.data_set import DataSet
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.stacked_col_graphs import StackedColGraphData, StackedColSubPlot
from results_analyzer.file_reader import read_file


class Sampling:
    measures: list[Measure]
    data: list[StackedColGraphData]
    categories: list[str]
    measures_count: int

    def __init__(self, working_dir: str, relative_file_paths: list[str], label_name: str, measures: list[Measure],
                 multi_dataset_title: str, categories: list[str], presentation_bin_banes: list[str]) -> None:
        self.datasets_list = []
        self.measures = measures
        self.data = []
        self.categories = categories[1:]
        self.measures_count = 0
        self.analyze_files(working_dir, relative_file_paths, label_name, categories)


    def analyze_files(self, working_dir: str, relative_file_paths: list[str], label_name: str, groups: list[str]):
        summary_data: list[dict] = []
        data_from_files: list[list[DataSet]] = []
        for i, relative_file_path in enumerate(relative_file_paths):
            datasets_list = read_file(self.measures, f'{working_dir}/{relative_file_path}', label_name, -1,
                                      1, None, 0,
                                      1, '')
            data_from_files.append(datasets_list)
            sample_bin = {"identifier": groups[i], "data_set": {}}
            for ds in datasets_list:
                measurement = ds.measurementsData[0]
                min_index = min(
                    range(len(measurement.ordered_scores)),
                    key=lambda j: measurement.ordered_scores[j]
                )
                min_predicted_score = measurement.ordered_scores[min_index]
                matching_dist = ds.dist[min_index]
                sample_bin["data_set"][ds.code] = {"min_predicted_score": min_predicted_score, "matching_dist": matching_dist}
            summary_data.append(sample_bin)
        base = summary_data[0]
        cols_num = len(self.categories)
        self.measures_count = len(base["data_set"].keys())
        s_better = StackedColGraphData(x=np.array(range(cols_num)), data_by_labels=[0] * cols_num, width=1,
                                       bottom=np.array([0] * cols_num), color=[], label=groups[1:], hatch=[])
        s_tie = StackedColGraphData(x=np.array(range(cols_num)), data_by_labels=[0] * cols_num, width=1,
                                    bottom=np.array([0] * cols_num), color=[], label=groups[1:], hatch=[])
        s_worse = StackedColGraphData(x=np.array(range(cols_num)), data_by_labels=[0] * cols_num, width=1,
                                      bottom=np.array([0] * cols_num), color=[], label=groups[1:], hatch=[])
        self.data = [s_better, s_tie, s_worse]
        for i in range(1, len(groups)):
            sample_bin = summary_data[i]
            for code in base["data_set"].keys():
                base_dist = base["data_set"][code]["matching_dist"]
                sample_bin_dist = sample_bin["data_set"][code]["matching_dist"]
                if sample_bin_dist > base_dist:
                    s_better.data_by_labels[i - 1] += 1
                elif sample_bin_dist == base_dist:
                    s_tie.data_by_labels[i - 1] += 1
                else:
                    s_worse.data_by_labels[i - 1] += 1
        for i, level in enumerate(self.data):
            level.data_by_labels = [x / self.measures_count * 100 for x in level.data_by_labels]
            if i > 0:
                level.bottom = self.data[i - 1].bottom + self.data[i - 1].data_by_labels

    def create_col_graph(self) -> StackedColSubPlot:
        labels = ['Better than 20 Samples', 'Same as 20 Samples', 'Worse than 20 Samples']
        colors = ['#196B24', '#A6A6A6', '#C00000']
        hatches = ['\\\\', 'xx', '//']
        labels_list = []
        cols_count = len(self.categories)

        samples_num: int = self.measures_count
        x = self.data[0].x

        for i, level in enumerate(self.data):
            level.width = 0.6
            level.label = labels[i]
            level.color = colors[i]
            level.hatch = hatches[i]
            labels_list.append({'x': x, 'bottom': level.bottom, 'val': level.data_by_labels})

        res = StackedColSubPlot(data=self.data, data_sources=[], ylabel='Dataset Percentage',
                                categories=self.categories, samples_num=samples_num, labels_list=labels_list,
                                p_value_per_ds=None)
        return res
