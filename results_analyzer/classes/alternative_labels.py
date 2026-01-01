
class AlternativeLabelFile:
    file_path: str
    label_inx: int
    dataset_inx: int
    code_inx: int
    feature_file_dataset_inx: int
    feature_file_code_inx: int

    def __init__(self, file_path: str, label_inx: int, dataset_inx: int, code_inx: int,
                 feature_file_dataset_inx: int, feature_file_code_inx: int):
        self.file_path = file_path
        self.label_inx = label_inx
        self.dataset_inx = dataset_inx
        self.code_inx = code_inx
        self.feature_file_dataset_inx = feature_file_dataset_inx
        self.feature_file_code_inx = feature_file_code_inx




