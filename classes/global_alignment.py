import os
import sys
from copy import copy

from classes.evo_model import EvoModel
from enums import AffineGapMatrixTypes
from utils import read_matching_matrix


class InternalCell:
    def __init__(self, row_inx: int, col_inx: int, children: list[any], matrix: str):
        self.matrix: str = matrix
        self.row_inx: int = row_inx
        self.col_inx: int = col_inx
        self.children: list[any] = children


def add_col_to_profile(profile: list[str], col: str) -> list[str]:
    profile.insert(0, col)
    return profile


class AlignmentCandidate:
    def __init__(self, profile_a: list[str], profile_b: list[str], arrow_from: InternalCell, child_inx: int):
        self.profile_a: list[str] = profile_a
        self.profile_b: list[str] = profile_b
        self.arrow_from: InternalCell | None = arrow_from
        self.child_inx: int | None = child_inx

    def add_codes_to_me(self, seq_a: str | None, seq_b: str | None, inx_a: int, inx_b: int):
        col_a = '-' if seq_a is None else seq_a[inx_a]
        col_b = '-' if seq_b is None else seq_b[inx_b]
        self.profile_a = add_col_to_profile(self.profile_a, col_a)
        self.profile_b = add_col_to_profile(self.profile_b, col_b)


def fill_matrix_cell_dict(matrix_cells_dict: dict[int, dict[int, dict[str, InternalCell]]], row_inx: int, col_inx: int,
                          prev_row: int, prev_col: int, m_val: float, x_val: float, y_val: float, max_val: float, key: str):
    children: list[InternalCell] = []
    if m_val == max_val:
        children.append(matrix_cells_dict[prev_row][prev_col][AffineGapMatrixTypes.M.value])
    if x_val == max_val:
        children.append(matrix_cells_dict[prev_row][prev_col][AffineGapMatrixTypes.X.value])
    if y_val == max_val:
        children.append(matrix_cells_dict[prev_row][prev_col][AffineGapMatrixTypes.Y.value])
    matrix_cells_dict[row_inx][col_inx][key] = InternalCell(row_inx, col_inx, children, key)


class GlobalAlign:
    def __init__(self, seq_a: str, seq_b: str, configuration: EvoModel):
        self.seq_a: str = seq_a
        self.seq_b: str = seq_b

        self.matching_matrix: list[list[int]] = []
        self.codes_dict_to_inx: dict[str, int] = {}
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0]
        matrix_file_path = os.path.join(script_dir, f'../input_config_files/{configuration.matrix_file_name}.txt')
        matching_matrix, codes_dict_to_inx = read_matching_matrix(matrix_file_path)
        self.matching_matrix = matching_matrix
        self.codes_dict_to_inx = codes_dict_to_inx
        self.gs = configuration.gs_cost
        self.ge = configuration.gs_cost
        self.matrix, self.alignment_root, self.best_score = self.calculate_alignment_matrix_and_root_affine_gaps()
        self.aligned_sequences: list[AlignmentCandidate] = self.get_alignment()

    def calculate_alignment_matrix_and_root_affine_gaps(self) -> tuple[list[list[float]], InternalCell, float]:
        matrix_m: list[list[float]] = []
        matrix_x: list[list[float]] = []
        matrix_y: list[list[float]] = []
        matrix_cells_dict: dict[int, dict[int, dict[str, InternalCell]]] = {}
        matrix_row_size: int = len(self.seq_a) + 1
        matrix_col_size: int = len(self.seq_b) + 1
        for i in range(matrix_row_size):
            matrix_m.append([0] * matrix_col_size)
            matrix_x.append([0] * matrix_col_size)
            matrix_y.append([0] * matrix_col_size)
        for row_inx in range(matrix_row_size):
            matrix_cells_dict[row_inx] = {}
            for col_inx in range(matrix_col_size):
                matrix_cells_dict[row_inx][col_inx] = {}
                if row_inx > 0 and col_inx > 0:
                    m_match = matrix_m[row_inx - 1][col_inx - 1]
                    x_match = matrix_x[row_inx - 1][col_inx - 1]
                    y_match = matrix_y[row_inx - 1][col_inx - 1]
                    max_val = max(m_match, x_match, y_match)
                    matrix_m[row_inx][col_inx] = max_val + self.compare_codes_score(row_inx, col_inx)
                    fill_matrix_cell_dict(matrix_cells_dict, row_inx, col_inx, row_inx - 1, col_inx - 1,
                                          m_match, x_match, y_match, max_val, AffineGapMatrixTypes.M.value)

                    m_x_gap = matrix_m[row_inx][col_inx - 1] + self.gs + self.ge
                    x_x_gap = matrix_x[row_inx][col_inx - 1] + self.ge
                    y_x_gap = matrix_y[row_inx][col_inx - 1] + self.gs + self.ge
                    max_val = max(m_x_gap, x_x_gap, y_x_gap)
                    matrix_x[row_inx][col_inx] = max_val
                    fill_matrix_cell_dict(matrix_cells_dict, row_inx, col_inx, row_inx, col_inx - 1,
                                          m_x_gap, x_x_gap, y_x_gap, max_val, AffineGapMatrixTypes.X.value)

                    m_y_gap = matrix_m[row_inx - 1][col_inx] + self.gs + self.ge
                    x_y_gap = matrix_x[row_inx - 1][col_inx] + self.ge + self.ge
                    y_y_gap = matrix_y[row_inx - 1][col_inx] + self.ge
                    max_val = max(m_y_gap, x_y_gap, y_y_gap)
                    matrix_y[row_inx][col_inx] = max_val
                    fill_matrix_cell_dict(matrix_cells_dict, row_inx, col_inx, row_inx - 1, col_inx,
                                          m_y_gap, x_y_gap, y_y_gap, max_val, AffineGapMatrixTypes.Y.value)

                elif col_inx > 0:
                    matrix_m[row_inx][col_inx] = -sys.maxsize
                    matrix_x[row_inx][col_inx] = col_inx * self.ge + self.gs
                    matrix_y[row_inx][col_inx] = -sys.maxsize
                    matrix_cells_dict[row_inx][col_inx] = {
                        'x': InternalCell(row_inx, col_inx, [
                            matrix_cells_dict[row_inx][col_inx - 1][AffineGapMatrixTypes.X.value]],
                                          AffineGapMatrixTypes.X.value),
                    }
                elif row_inx > 0:
                    matrix_m[row_inx][col_inx] = -sys.maxsize
                    matrix_x[row_inx][col_inx] = -sys.maxsize
                    matrix_y[row_inx][col_inx] = row_inx * self.ge + self.gs
                    matrix_cells_dict[row_inx][col_inx] = {
                        'y': InternalCell(row_inx, col_inx, [
                            matrix_cells_dict[row_inx - 1][col_inx][AffineGapMatrixTypes.Y.value]],
                                          AffineGapMatrixTypes.Y.value),
                    }
                else:
                    matrix_m[row_inx][col_inx] = 0
                    matrix_x[row_inx][col_inx] = -sys.maxsize
                    matrix_y[row_inx][col_inx] = -sys.maxsize
                    matrix_cells_dict[row_inx][col_inx] = {
                        'm': InternalCell(row_inx, col_inx, [], AffineGapMatrixTypes.M.value),
                        'x': InternalCell(row_inx, col_inx, [], AffineGapMatrixTypes.X.value),
                        'y': InternalCell(row_inx, col_inx, [], AffineGapMatrixTypes.Y.value),
                    }
            if row_inx > 1:
                del matrix_cells_dict[row_inx - 1]

        m_res = matrix_m[matrix_row_size - 1][matrix_col_size - 1]
        x_res = matrix_x[matrix_row_size - 1][matrix_col_size - 1]
        y_res = matrix_y[matrix_row_size - 1][matrix_col_size - 1]
        max_val = max(m_res, x_res, y_res)
        matrix_cells_dict[matrix_row_size] = {}
        matrix_cells_dict[matrix_row_size][matrix_col_size] = {}
        fill_matrix_cell_dict(matrix_cells_dict, matrix_row_size, matrix_col_size, matrix_row_size - 1,
                              matrix_col_size - 1, m_res, x_res, y_res, max_val, AffineGapMatrixTypes.M.value)
        root = matrix_cells_dict[matrix_row_size][matrix_col_size][AffineGapMatrixTypes.M.value]
        return matrix_m, root, max_val

    def compare_codes_score(self, row_inx: int, col_inx: int) -> float:
        if row_inx == 0 and col_inx == 0:
            return 0
        return self.seq_matrix_matching_score(row_inx, col_inx)

    def seq_matrix_matching_score(self, row_inx: int, col_inx: int) -> int:
        from_inx = self.codes_dict_to_inx[self.seq_a[row_inx - 1]]
        to_inx = self.codes_dict_to_inx[self.seq_b[col_inx - 1]]
        return self.matching_matrix[from_inx][to_inx]

    def print_matrix(self):
        print(self.matrix)

    def get_score(self) -> float:
        if not self.best_score:
            self.best_score = self.matrix[-1][-1]
        return self.best_score

    def get_alignment(self) -> list[AlignmentCandidate]:
        alignment_results: list[AlignmentCandidate] = []
        alignment_queue: list[AlignmentCandidate] = []
        if self.alignment_root is None:
            return []
        for child_inx in range(len(self.alignment_root.children)):
            alignment_queue.append(AlignmentCandidate([], [], self.alignment_root, child_inx))
        while len(alignment_queue) > 0:
            current_al_cnd: AlignmentCandidate = alignment_queue.pop(0)
            arrow_from: InternalCell = current_al_cnd.arrow_from
            arrow_to: InternalCell | None = arrow_from.children[current_al_cnd.child_inx]
            while arrow_to and arrow_to.row_inx >= 0 and arrow_to.col_inx >= 0:
                if arrow_to.row_inx == arrow_from.row_inx - 1 and arrow_to.col_inx == arrow_from.col_inx - 1:
                    if arrow_to.row_inx < len(self.seq_a):
                        current_al_cnd.add_codes_to_me(self.seq_a, self.seq_b, arrow_to.row_inx, arrow_to.col_inx)
                elif arrow_to.row_inx == arrow_from.row_inx - 1:
                    current_al_cnd.add_codes_to_me(self.seq_a, None, arrow_to.row_inx, -1)
                else:
                    current_al_cnd.add_codes_to_me(None, self.seq_b, -1, arrow_to.col_inx)
                if len(arrow_to.children) > 1:
                    for c_i in range(1, len(arrow_to.children)):
                        alignment_queue.append(AlignmentCandidate(copy(current_al_cnd.profile_a),
                                                                  copy(current_al_cnd.profile_b),
                                                                  arrow_to, c_i))
                arrow_from = arrow_to
                arrow_to = arrow_from.children[0] if len(arrow_from.children) else None
            current_al_cnd.arrow_from = None
            current_al_cnd.child_inx = None
            alignment_results.append(current_al_cnd)
        return alignment_results

    def print_results(self):
        print('The alignment score is:', self.get_score())
        print('The alignments are:\n')
        for a in self.aligned_sequences:
            print(a.seq_a)
            print(a.seq_b)
            print('')
        print('The alignment matrix is:')
        for row in self.matrix:
            print(row)

