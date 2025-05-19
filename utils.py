import math
from pathlib import Path


def read_matching_matrix(file_path: str) -> tuple[list[list[int]], dict[str, int]]:
    codes_dict_to_inx: dict[str, int] = {}
    match_matrix: list[list[int]] = []
    with open(file_path) as infile:
        for l_inx, line in enumerate(infile):
            line = line.strip()
            is_first_col = True
            if l_inx == 0:
                for c in line.split(' '):
                    if c != '':
                        codes_dict_to_inx[c] = len(codes_dict_to_inx)
            else:
                match_matrix.append([])
                for c in line.split(' '):
                    if c != '':
                        if not is_first_col:
                            match_matrix[l_inx - 1].append(int(c))
                        is_first_col = False
    return match_matrix, codes_dict_to_inx


def translate_to_matrix_index(letter: str, code_to_index_dict: dict[str, int]) -> int:  # TODO: check '*' Vs. '-'
    return code_to_index_dict[letter] if letter in code_to_index_dict else code_to_index_dict['*']


def get_keys_list(keys_structure: list) -> list[str]:
    res: list[str] = []
    add_children_to_list(keys_structure, res)
    return res


def add_children_to_list(keys_structure: list, res: list[str]):
    if type(keys_structure) is list:
        for item in keys_structure:
            if type(item) is list:
                add_children_to_list(item, res)
            else:
                res.append(item)
                print(len(res))
                print(item)


def calc_p_distance_from_other(aligned_seq: str, other_aligned_seq: str) -> float:
    changes_count = 0
    for i in range(len(aligned_seq)):
        if aligned_seq[i] != other_aligned_seq[i]:
            changes_count += 1
    return changes_count / len(aligned_seq)


def calc_kimura_distance_from_other(aligned_seq: str, other_aligned_seq: str) -> float:
    fractional_identity: float = calc_p_distance_from_other(aligned_seq, other_aligned_seq)
    print('fractional_identity: ', fractional_identity)
    kimura_exponent = 1 - fractional_identity - 0.2 * fractional_identity * fractional_identity
    if kimura_exponent < 0:
        return 2
    return -math.log(kimura_exponent)

