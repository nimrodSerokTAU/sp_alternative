from classes.evo_model import EvoModel
from enums import SopCalcTypes, WeightMethods, StatsOutput

class Configuration:
    models: list[EvoModel]
    sop_clac_type: SopCalcTypes
    input_files_dir_name: str | None
    additional_weights: set[WeightMethods]
    stats_output: set[StatsOutput]

    def __init__(self, models_list: list[EvoModel],
                 sop_clac_type: SopCalcTypes = SopCalcTypes.EFFICIENT,
                 input_files_dir_name: str | None = None,
                 additional_weights: set[WeightMethods] = None,
                 stats_output = {StatsOutput.ALL}):
        self.models = models_list
        self.sop_clac_type = sop_clac_type
        self.input_files_dir_name = input_files_dir_name
        self.additional_weights = additional_weights if additional_weights is not None else []
        self.stats_output = stats_output

    # Blo 62: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62

