from results_analyzer.classes.measurement_data import MeasurementDataPerCode


class Measure:
    key: str
    presentation_name: str | None
    external_name: str
    r_values: list[float]
    data_per_dataset: list[MeasurementDataPerCode]
    correlation_direction: int

    def __init__(self, key: str, external_name: str, presentation_name: str | None, correlation_direction: int):
        self.key = key
        self.external_name = external_name
        self.presentation_name = presentation_name
        self.r_values = []
        self.data_per_dataset = []
        self.correlation_direction = correlation_direction


    def append_dataset_data(self, dataset_data: MeasurementDataPerCode):
        self.data_per_dataset.append(dataset_data)
        self.r_values.append(dataset_data.r_value)

