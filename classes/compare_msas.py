from classes.msa import MSA
from pathlib import Path
from fpdf import FPDF

from dpos import translate_profile_naming, create_hpos_table, get_place_dpos


class MSACompare:
    test_msa: MSA
    true_msa: MSA
    compared_msa: MSA | None
    is_compared_to_other: bool

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




    @staticmethod
    def compute_dpos_distance_by_percent(profile_a: list[str], profile_b: list[str]):
        dpos_list: list[dict] = []
        profile_a_naming: list[list[str]] = translate_profile_naming(profile_a)
        profile_b_naming: list[list[str]] = translate_profile_naming(profile_b)
        seq_count: int = len(profile_a)
        profile_a_hpos: list[list[set[str]]] = create_hpos_table(profile_a_naming)
        profile_b_hpos: list[list[set[str]]] = create_hpos_table(profile_b_naming)
        two_sets_size: int = (seq_count - 1) * 2
        dpos_count: int = 0
        dpos_sum: float = 0
        for i in range(seq_count):
            for j in range(len(profile_a_hpos[i])):
                print(f"i:{i}, j:{j}")
                hpos_a_i: set[str] = profile_a_hpos[i][j]
                hpos_b_i: set[str] = profile_b_hpos[i][j]
                if len(hpos_a_i) > 0 or len(hpos_b_i) > 0:
                    dpos_i_j = get_place_dpos(hpos_a_i, hpos_b_i, two_sets_size)
                    # if i == 4 and j == 236:
                    #     stop = True
                    if dpos_i_j > 0:
                        dpos_list.append({'i': i, 'j': j, 'g': dpos_i_j})
                        dpos_sum += dpos_i_j
                    dpos_count += 1

        return dpos_list, dpos_sum, dpos_count

    def print_single(self, first_p_th: int, second_p_th: int, file_path: Path, sequences: list[str]):
        dict_i_j_color, dpos_grade = self.calc_colors(first_p_th, second_p_th, sequences)
        pdf = FPDF(format='letter', unit='in')
        pdf.add_page()
        pdf.set_font('Helvetica', '', 1)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0.03, 0.0, f'dpos_grade: {dpos_grade}')
        pdf.ln(0.05)
        for i, seq in enumerate(sequences):
            j = 0
            for char in seq:
                if char != '-':
                    if i in dict_i_j_color and j in dict_i_j_color[i]:
                        if dict_i_j_color[i][j] == 'red':
                            pdf.set_text_color(212, 19, 38)
                        else:
                            pdf.set_text_color(232, 97, 30)
                    j += 1
                pdf.cell(0.01, 0.0, char)
                pdf.set_text_color(100, 100, 100)
            pdf.ln(0.05)
        pdf.output(str(file_path), 'F')
        a = 1

    def calc_colors(self, first_p_th: int, second_p_th: int, sequences: list[str]) -> tuple[dict, float]:
        dpos_list, dpos_sum, dpos_count = self.compute_dpos_distance_by_percent(sequences,
                                                                                self.true_msa.sequences)
        dpos_list.sort(key=lambda x: x['g'], reverse=True)
        first_th_val: float = first_p_th * dpos_sum / 100
        second_th_val: float = second_p_th * dpos_sum / 100
        dpos_running_sum: float = 0
        first_p_last_item = {'g': 2}
        second_p_last_item = {'g': 2}
        dict_i_j_color = {}
        for dpos in dpos_list:
            dpos_running_sum += dpos['g']
            if dpos_running_sum < first_th_val or dpos['g'] == first_p_last_item['g']:
                first_p_last_item = dpos
                add_color_to_pos(dict_i_j_color, dpos['i'], dpos['j'], 'red')
            elif dpos_running_sum < second_th_val or dpos['g'] == second_p_last_item['g']:
                second_p_last_item = dpos
                add_color_to_pos(dict_i_j_color, dpos['i'], dpos['j'], 'orange')
            else:
                break
        return dict_i_j_color, dpos_sum / dpos_count


def add_color_to_pos(dict_i_j_color: dict, i: int, j: int, color: str):
    if i not in dict_i_j_color:
        dict_i_j_color[i] = {}
    dict_i_j_color[i][j] = color


def msa_comp_main():
    comp: MSACompare = MSACompare(Path('../compThree/PRANK_b1#0018_htT_tree_12_OP_0.35247606317119684_Split_18.fasta'),
                                  Path('../compThree/NR4A2_TRUE.fas'),
                                  Path('../compThree/PRANK_b0#0041_hhT_tree_9_OP_0.23758512508472418_Split_41.fasta'))
    # print(f"{style.RED}{test1} {style.BLUE}{test2}{style.RESET}")
    comp.print_single(5, 20, Path('D:/test1.pdf'), comp.test_msa.sequences)
    comp.print_single(5, 20, Path('D:/comp1.pdf'), comp.compared_msa.sequences)
