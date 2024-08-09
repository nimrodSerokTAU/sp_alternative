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


def translate_to_matrix_index(letter: str, code_to_index_dict: dict[str, int]) -> int:
    return code_to_index_dict[letter] if letter in code_to_index_dict else code_to_index_dict['*']

