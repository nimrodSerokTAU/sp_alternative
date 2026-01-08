import csv

from results_analyzer.classes.alternative_labels import AlternativeLabelFile
from results_analyzer.classes.col_plot import ColPlot
from results_analyzer.classes.scatter_plot import ScatterPlot
from results_analyzer.classes.col_graphs import DataGroupCorr, DataGroupMgr
from results_analyzer.constants import NAMING
from results_analyzer.classes.data_set import DataSet, Point
from results_analyzer.classes.measure import Measure
from results_analyzer.utils import is_float


class CorrelationAnalyzer:
    datasets_list: list[DataSet]
    measures: list[Measure]
    r_groups: list[DataGroupCorr]
    data_group_mgr: DataGroupMgr
    example_res: ScatterPlot
    accum_r: ColPlot
    zoom_in: list[ScatterPlot]

    def __init__(self, working_dir: str, relative_file_paths: list[str], label_name: str, measures: list[Measure],
                 example_dataset: str, print_zoom: bool, multi_dataset_title: str, points_count: int, dist_threshold: float,
                 alternative_label_file: AlternativeLabelFile | None = None, continue_missing_code: bool = False,
                 ignore_code: str = '') -> None:
        self.datasets_list = []
        self.measures = measures
        self.data_group_mgr = DataGroupMgr(self.measures)
        self.zoom_in = []
        is_multi_file = len(relative_file_paths) > 1
        data_from_files: list[list[DataSet]] = []
        ext_labels_dict = None
        if alternative_label_file:
            ext_labels_dict = self.calc_external_labels_dict(f'{working_dir}/{alternative_label_file.file_path}',
                                                             alternative_label_file)

        feature_file_code_inx = alternative_label_file.feature_file_code_inx if alternative_label_file else 1
        feature_file_dataset_inx = alternative_label_file.feature_file_dataset_inx if alternative_label_file else 0
        for i, relative_file_path in enumerate(relative_file_paths):
            datasets_list = self.read_file(f'{working_dir}/{relative_file_path}', label_name, i if is_multi_file else -1,
                                           dist_threshold, ext_labels_dict, feature_file_dataset_inx, feature_file_code_inx, ignore_code)
            data_from_files.append(datasets_list)
        if is_multi_file:
            self.arrange_multi_file_data(data_from_files)
        else:
            self.arrange_data(data_from_files[0], continue_missing_code)
        self.data_group_mgr.set_dataset_counter(len(self.datasets_list))
        for measure in self.measures:
            print (f'top_v of {measure.key} avg r is {sum(measure.r_values) / max(len(measure.r_values), 1)} for {len(measure.r_values)} samples' )
        example_cases: list[DataSet] = [x for x in self.datasets_list if x.code == example_dataset]
        if len(example_cases) > 0:
            example_case: DataSet = example_cases[0]
            self.example_res = self.get_dataset_scatter(example_case)
            if print_zoom:
                self.print_zoom_in(example_case, points_count)
        self.accum_r = self.data_group_mgr.print_r( [x.key for x in measures], multi_dataset_title)


    def read_file(self, file_path: str, label_name: str, file_index: int, dist_threshold: float,
                  ext_labels_dict: dict[dict:[str, float]] | None, feature_file_dataset_inx: int, feature_file_code_inx: int, ignore: str) -> list[DataSet]:
        measures = self.measures if file_index == -1 else [self.measures[file_index]]
        curr_code_set = DataSet('', measures)
        datasets_list: list[DataSet] = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            header_dict: dict[str, int] = {}
            for i, key in enumerate(header):
                header_dict[key] = i
            print(f"Header: {header}")
            label_col = header_dict[label_name]
            for i, row in enumerate(csv_reader):
                drop = False
                dataset_code = row[feature_file_dataset_inx]
                if dataset_code != curr_code_set.code:
                    curr_code_set = DataSet(dataset_code, measures)
                    datasets_list.append(curr_code_set)
                code = row[feature_file_code_inx]
                if code == ignore or code == dataset_code:
                    drop = True
                measures_data: list[str] = []
                for measure in measures:
                    if measure.external_name not in header_dict:
                        drop = True
                        continue
                    measure_col = header_dict[measure.external_name]
                    if row[measure_col] == '' or not is_float(row[label_col]):
                        drop = True
                    measures_data.append(row[measure_col])
                if not drop and row[label_col] != '' and float(row[label_col]) <= dist_threshold:
                    dist = row[label_col]
                    if float(dist) == 0:
                        debug = 2
                    if ext_labels_dict:
                        if dataset_code in ext_labels_dict and row[feature_file_code_inx] in ext_labels_dict[dataset_code]:
                            dist = ext_labels_dict[dataset_code][row[feature_file_code_inx]]
                        else:
                            dist = None
                    if dist:
                        curr_code_set.append_sample(dist=dist, scores=measures_data)
        return datasets_list


    def arrange_data(self, datasets_list: list[DataSet], continue_missing_code: bool):
        if continue_missing_code:
            datasets_list = [item for item in datasets_list if len(item.dist) > 2]

        self.datasets_list = datasets_list
        for rd in self.datasets_list:
            rd.calc_pearson(continue_missing_code)
            for measure_i in range(len(self.measures)):
                self.measures[measure_i].append_dataset_data(rd.measurementsData[measure_i])
        for measure_i in range(len(self.measures)):
            is_m_direction_positive: bool = sum(self.measures[measure_i].r_values) >= 0
            self.data_group_mgr.set_measure_direction(self.measures[measure_i].key, self.measures[measure_i].correlation_direction == 1)
            for rd in self.datasets_list:
                self.data_group_mgr.add_sample(rd.measurementsData[measure_i])

    @staticmethod
    def get_dataset_scatter(example: DataSet):
        example.normalize()
        return example.create_scatter(example.measurementsData)

    def print_zoom_in(self, example: DataSet, points_count: int):
        results: list[tuple [DataSet, float]] = get_top(example, points_count, self.measures)
        for result in results:
            self.zoom_in.append(result[0].plot_zoom_scatter(result[0].measurementsData[0], result[1]))

    @staticmethod
    def merge_data_from_files(data_from_files: list[list[DataSet]]) -> list[DataSet]:
        codes_dict: dict[str, DataSet] = {}
        for ds in data_from_files:
            for rd in ds:
                if rd.code not in codes_dict:
                    codes_dict[rd.code] = rd
                elif rd.dist == codes_dict[rd.code].dist:
                    codes_dict[rd.code].measurementsData.append(rd.measurementsData[0])
                else:
                    print('Error on merge_data_from_files')
                    return []
        return list(codes_dict.values())

    def arrange_multi_file_data(self, data_from_files: list[list[DataSet]]):
        measures_by_key = [m.key for m in self.measures]
        self.datasets_list = self.merge_data_from_files(data_from_files)
        for rd in self.datasets_list:
            rd.calc_pearson(True)
            for m in rd.measurementsData:
                measure_index = measures_by_key.index(m.measure_key)
                self.measures[measure_index].append_dataset_data(m)
        for measure_i in range(len(self.measures)):
            is_m_direction_positive: bool = sum(self.measures[measure_i].r_values) >= 0
            self.data_group_mgr.set_measure_direction(self.measures[measure_i].key, is_m_direction_positive)
        for rd in self.datasets_list:
            for m in rd.measurementsData:
                self.data_group_mgr.add_sample(m)

    def get_example_scatter(self):
        return self.example_res

    def get_r(self):
        return self.accum_r

    @staticmethod
    def calc_external_labels_dict(potential_labels_path: str, alternative_label_file: AlternativeLabelFile) -> dict[dict:[str, float]]:
        ext_labels_dict: dict[dict:[str, float]] = {}
        with open(potential_labels_path, 'r') as file:
            csv_reader = csv.reader(file)
            for i, row in enumerate(csv_reader):
                code = row[alternative_label_file.code_inx]
                data_set_code = row[alternative_label_file.dataset_inx]
                val = row[alternative_label_file.label_inx]
                if data_set_code not in ext_labels_dict:
                    ext_labels_dict[data_set_code] = {}
                ext_labels_dict[data_set_code][code] = val
        return ext_labels_dict


def get_top(original_ds: DataSet, points_count: int, measures: list[Measure]) -> list[tuple [DataSet, float]]:
    top_ds_list: list[tuple [DataSet, float]] = []
    correlation_dict: dict[str, int] = {}
    for m in measures:
        correlation_dict[m.key] = m.correlation_direction
    for measure_data in original_ds.measurementsData:
        measure_key = measure_data.measure_key
        points = [Point(original_ds.dist[i], measure_data.ordered_scores[i]) for i in
                                           range(len(original_ds.dist))]
        max_score: float = max(points, key=lambda p: p.score).score
        min_score: float = min(points, key=lambda p: p.score).score
        min_distance: float = min(points, key=lambda p: p.distance).distance
        sorted_points_by_score = sorted(points, key=lambda x: x.score, reverse=measure_data.r_value < 0)
        measure = Measure(measure_key, '', NAMING[measure_key], correlation_dict[measure_key])
        top_ds = DataSet(measure_data.code, [measure])
        min_dist_score: float = -1
        for i, p in enumerate(sorted_points_by_score):
            if p.distance == min_distance:
                min_dist_score = p.score
                break
        for i in range(points_count):
            p = sorted_points_by_score[i]
            normalized_score: float = normalize_score(min_score, max_score, p.score)
            top_ds.append_sample(dist=p.distance, scores=[normalized_score])
        top_ds.calc_pearson(False)
        measure_tuple: tuple [DataSet, float] = (top_ds, normalize_score(min_score, max_score, min_dist_score))
        top_ds_list.append(measure_tuple)
    return top_ds_list

def normalize_score(min_score: float, max_score: float, score: float) -> float:
    return (score - min_score) / (max_score - min_score)

