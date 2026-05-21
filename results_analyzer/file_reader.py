import csv

from results_analyzer.classes.data_set import DataSet
from results_analyzer.classes.measure import Measure
from results_analyzer.utils import is_float


def read_file(_measures: list[Measure], file_path: str, label_name: str, measure_file_index: int, dist_threshold: float,
              ext_labels_dict: dict[dict:[str, float]] | None, feature_file_dataset_inx: int,
              feature_file_code_inx: int, ignore: str) -> list[DataSet]:
    measures = _measures if measure_file_index == -1 else [_measures[measure_file_index]]
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
                    if dataset_code in ext_labels_dict and row[feature_file_code_inx] in ext_labels_dict[
                        dataset_code]:
                        dist = ext_labels_dict[dataset_code][row[feature_file_code_inx]]
                    else:
                        dist = None
                if dist:
                    curr_code_set.append_sample(dist=dist, scores=measures_data)
    return datasets_list
