from enums import SopCalcTypes


class Configuration:
    gs_cost: int
    ge_cost: float
    gs_cost_extremities: int
    blosum_file_name: str
    sop_clac_type: SopCalcTypes
    input_files_dir_name: str | None
    is_analyze_per_dir: bool
    is_compute_correlation: bool

    def __init__(self, gs_cost: int, ge_cost: float, gs_cost_extremities: int, blosum_file_name: str,
                 sop_clac_type: SopCalcTypes = SopCalcTypes.EFFICIENT, input_files_dir_name: str | None = None,
                 is_analyze_per_dir: bool = False, is_compute_correlation: bool = False):
        self.gs_cost = gs_cost
        self.ge_cost = ge_cost
        self.gs_cost_extremities = gs_cost_extremities
        self.blosum_file_name = blosum_file_name
        self.sop_clac_type = sop_clac_type
        self.input_files_dir_name = input_files_dir_name
        self.is_analyze_per_dir = is_analyze_per_dir
        self.is_compute_correlation = is_compute_correlation

    # Blo 62: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62

