from classes.config import Configuration
from classes.evo_model import EvoModel
from classes.msa import MSA
from pathlib import Path
from fpdf import FPDF
from enum import Enum

from classes.sp_score import SPScore
from distance_calc import translate_profile_naming, create_h_table, get_place_d
from enums import SopCalcTypes, DistanceType


class MSACompare:
    test_msa: MSA
    true_msa: MSA
    compared_msa: MSA | None
    is_compared_to_other: bool
    sop: SPScore

    def __init__(self, test_dataset_path: Path, true_dataset_path: Path, compared_dataset_path: Path | None,
                 is_compared_to_other: bool = False):
        self.test_msa = MSA('test_dataset')
        self.true_msa = MSA('true_dataset')
        self.compared_msa = MSA('compared_dataset')
        self.is_compared_to_other = is_compared_to_other
        self.test_msa.read_me_from_fasta(test_dataset_path)
        self.true_msa.read_me_from_fasta(true_dataset_path)
        self.compared_msa.read_me_from_fasta(compared_dataset_path)
        self.test_msa.order_sequences(self.true_msa.seq_names)
        self.compared_msa.order_sequences(self.true_msa.seq_names)
        configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                                     SopCalcTypes.EFFICIENT, 'comparison_files')
        self.sop = SPScore(configuration.models[0])


    @staticmethod
    def compute_dpos_distance_by_percent(profile_a: list[str], profile_b: list[str]):
        dpos_list: list[dict] = []
        profile_a_naming: list[list[str]] = translate_profile_naming(profile_a, DistanceType.D_POS)
        profile_b_naming: list[list[str]] = translate_profile_naming(profile_b, DistanceType.D_POS)
        seq_count: int = len(profile_a)
        profile_a_hpos: list[list[set[str]]] = create_h_table(profile_a_naming, DistanceType.D_POS)
        profile_b_hpos: list[list[set[str]]] = create_h_table(profile_b_naming, DistanceType.D_POS)
        dpos_count: int = 0
        dpos_sum: float = 0
        for i in range(seq_count):
            for j in range(len(profile_a_hpos[i])):
                # print(f"i:{i}, j:{j}")
                hpos_a_i: set[str] = profile_a_hpos[i][j]
                hpos_b_i: set[str] = profile_b_hpos[i][j]
                if len(hpos_a_i) > 0 or len(hpos_b_i) > 0:
                    dpos_i_j = get_place_d(hpos_a_i, hpos_b_i, DistanceType.D_POS)
                    if i == 21 and j == 17:
                        stop = True
                    if dpos_i_j > 0:
                        dpos_list.append({'i_inx': i, 'j_pos': j + 1, 'grade': dpos_i_j})
                        dpos_sum += dpos_i_j
                    dpos_count += 1

        return dpos_list, dpos_sum, dpos_count, profile_a_hpos, profile_a_naming

    def print_single_sop(self, first_p_th: int, second_p_th: int, file_path: Path, sequences: list[str]):
        """
        It appears that some alternatives are inserting a lot of gaps which does not affect the dPos dramatically, but do affect the SoP.
        For example:
        PRANK_b1#0015_hhT_tree_10_OP_0.21782315330463242_Split_15.fasta: total_SoP: 1,055,453, subs: 1,232,430, go: -158,230, ge: -18,747, msa_len: 683
        MUSCLE_diversified_replicate.none.84.afa:1094918:                total_SoP: 1,094,918, subs: 1,224,056, go: -113,780, ge: -15,358, msa_len: 645
        true                                                             total_SoP: 1,087,388, subs: 1,237,266, go: -132,640, ge: -17,238, msa_len: 686
        this means 38 more cols of gaps, each with 40 gaps...
        the squashing of the sequence is better for our SoP model even for mismatches with this cost parameters. probably no place to do it on empirical
        empirical has less seq....and less spaces.
        """
        msa_len = len(sequences[0])
        naive_sop_score = self.sop.compute_naive_sp_score(sequences)
        efficient_sop_score = self.sop.compute_efficient_sp(sequences)
        sp_score_subs, sp_score_gap_o, sp_score_gap_e = self.sop.compute_naive_sp_score_per_col(sequences)
        naive_sop_score_col = sum(sp_score_subs) + sum(sp_score_gap_o) + sum(sp_score_gap_e)
        sop_per_col = [sp_score_subs[i] + sp_score_gap_o[i] + sp_score_gap_e[i] for i in range(msa_len)]
        sorted_cols = sorted(sop_per_col)
        higher_score_th = sorted_cols[int(msa_len * 0.9)]
        seq_count = len(sequences)
        lower_score_th = (seq_count - 1) * (self.sop.ge_cost + self.sop.gs_cost)
        best = [i for i in range(msa_len) if sop_per_col[i] >= higher_score_th]
        worse = [i for i in range(msa_len) if sop_per_col[i] < lower_score_th]

        dict_i_j_color, dpos_grade = self.calc_colors(first_p_th, second_p_th, sequences)
        pdf = FPDF(format='letter', unit='in')
        pdf.add_page()
        pdf.set_font('Helvetica', '', 1)
        pdf.set_text_color(200, 200, 200)
        pdf.cell(0.03, 0.0, f'dpos_grade: {dpos_grade}')
        pdf.ln(0.05)
        for i_inx, seq in enumerate(sequences):
            for j_inx, char in enumerate(seq):
                if sop_per_col[j_inx] <= lower_score_th:
                    pdf.set_text_color(212, 19, 38)
                elif sop_per_col[j_inx] >= higher_score_th:
                    pdf.set_text_color(47, 96, 222)
                else:
                    pdf.set_text_color(200, 200, 200)
                pdf.cell(0.01, 0.0, char)
            pdf.ln(0.05)
        pdf.output(str(file_path), 'F')
        a = 1

    def print_single_dpos(self, first_p_th: int, second_p_th: int, file_path: Path, sequences: list[str]):
        dict_i_j_color, dpos_grade = self.calc_colors(first_p_th, second_p_th, sequences)
        pdf = FPDF(format='letter', unit='in')
        pdf.add_page()
        pdf.set_font('Helvetica', '', 1)
        pdf.set_text_color(200, 200, 200)
        pdf.cell(0.03, 0.0, f'dpos_grade: {dpos_grade}')
        pdf.ln(0.05)
        for i_inx, seq in enumerate(sequences):
            for j_inx, char in enumerate(seq):
                if i_inx in dict_i_j_color and j_inx in dict_i_j_color[i_inx]:
                    if dict_i_j_color[i_inx][j_inx] == ColorsOrder.RED and char != '-':
                        pdf.set_text_color(212, 19, 38)
                    elif dict_i_j_color[i_inx][j_inx] == ColorsOrder.ORANGE and char != '-':
                        pdf.set_text_color(232, 97, 30)
                    elif dict_i_j_color[i_inx][j_inx] == ColorsOrder.BLUE:
                        pdf.set_text_color(47, 96, 222)
                    elif dict_i_j_color[i_inx][j_inx] == ColorsOrder.LIGHT_BLUE:
                        pdf.set_text_color(84, 137, 184)
                pdf.cell(0.01, 0.0, char)
                pdf.set_text_color(200, 200, 200)
            pdf.ln(0.05)
        pdf.output(str(file_path), 'F')
        a = 1

    def calc_colors(self, first_p_th: int, second_p_th: int, sequences: list[str]) -> tuple[dict, float]:
        dpos_list, dpos_sum, dpos_count, profile_a_hpos, profile_a_naming = self.compute_dpos_distance_by_percent(
            sequences, self.true_msa.sequences)
        dpos_list.sort(key=lambda x: x['grade'], reverse=True)
        first_th_val: float = first_p_th * dpos_sum / 100
        second_th_val: float = second_p_th * dpos_sum / 100
        dpos_running_sum: float = 0
        first_p_last_item = {'grade': 2}
        second_p_last_item = {'grade': 2}
        dict_i_j_color = {}
        for dpos in dpos_list:
            dpos_running_sum += dpos['grade']
            if dpos_running_sum < first_th_val or dpos['grade'] == first_p_last_item['grade']:
                first_p_last_item = dpos
                add_color_to_s_char(dict_i_j_color, dpos['i_inx'], profile_a_naming, ColorsOrder.RED, dpos['j_pos'])
                add_color_to_h_pos(dict_i_j_color, ColorsOrder.BLUE, profile_a_hpos[dpos['i_inx']][dpos['j_pos'] - 1],
                                   profile_a_naming)
            elif dpos_running_sum < second_th_val or dpos['grade'] == second_p_last_item['grade']:
                second_p_last_item = dpos
                add_color_to_s_char(dict_i_j_color, dpos['i_inx'], profile_a_naming, ColorsOrder.ORANGE, dpos['j_pos'])
            else:
                break
        return dict_i_j_color, dpos_sum / dpos_count


class ColorsOrder(Enum):
    RED = 0
    ORANGE = 1
    BLUE = 2
    LIGHT_BLUE = 3


def add_color_to_s_char(dict_i_j_color: dict, i_inx: int, profile_a_naming: list[list[str]], color: ColorsOrder,
                        pos_j: int):
    pos = f'S^{i_inx + 1}_{pos_j}'
    j_inx: int = profile_a_naming[i_inx].index(pos)
    add_color_to_pos(dict_i_j_color, i_inx, j_inx, color)


def add_color_to_pos(dict_i_j_color: dict, i_inx: int, j_inx: int, color: ColorsOrder):
    if i_inx not in dict_i_j_color:
        dict_i_j_color[i_inx] = {}
    if j_inx in dict_i_j_color[i_inx]:
        color = dict_i_j_color[i_inx][j_inx] if color.value <= dict_i_j_color[i_inx][j_inx].value else color.value
    dict_i_j_color[i_inx][j_inx] = color


def add_color_to_h_pos(dict_i_j_color: dict, color: ColorsOrder, profile_a_hpos: set[str],
                       profile_a_naming: list[list[str]]):
    for other_pos_in_col in profile_a_hpos:
        other_pos_i_inx: int = get_i_inx_from_pos(other_pos_in_col)
        j_inx: int = profile_a_naming[other_pos_i_inx].index(other_pos_in_col)
        add_color_to_pos(dict_i_j_color, other_pos_i_inx, j_inx, color)


def get_i_inx_from_pos(pos: str) -> int:
    pos = pos.replace('^', '_')
    parts = pos.split('_')
    return int(parts[1]) - 1


def msa_comp_main():
    # TODO: calc SoP to all. use different methods, try to understand the diff in SoP, use visual
    comp: MSACompare = MSACompare(Path('../compThree/PRANK_b1#0015_hhT_tree_10_OP_0.21782315330463242_Split_15.fasta'),
                                  # Path('../compThree/AATF_TRUE.fas'),
                                  Path('../compThree/MUSCLE_diversified_replicate.none.84.afa'),
                                  Path('../compThree/AATF_TRUE.fas'))
    # print(f"{style.RED}{test1} {style.BLUE}{test2}{style.RESET}")
    comp.print_single_sop(5, 20, Path('../output/test1.pdf'), comp.test_msa.sequences)
    comp.print_single_sop(5, 20, Path('../output/comp1.pdf'), comp.compared_msa.sequences)
    comp.print_single_sop(5, 20, Path('../output/true1.pdf'), comp.true_msa.sequences)
