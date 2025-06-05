
class EvoModel:
    gs_cost: int
    ge_cost: float
    matrix_file_name: str
    name: str


    def __init__(self, gs_cost: int, ge_cost: float, matrix_file_name: str, name: str = None):
        self.gs_cost = gs_cost
        self.ge_cost = ge_cost
        self.matrix_file_name = matrix_file_name
        self.name = name if name is not None else matrix_file_name


    # Blo 62: https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62

