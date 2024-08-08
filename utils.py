

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


def translate_seq_hpos(sequence: str, seq_index: int) -> list[str]:
    res: list[str] = []
    s_count: int = 1
    for c in sequence:
        if c != '-':
            res.append(f'S^{seq_index}_{s_count}')
            s_count += 1
        else:
            res.append(f'G^{seq_index}_{s_count - 1}')
    return res


def translate_profile_hpos(profile: list[str]) -> list[list[str]]:
    res: list[list[str]] = []
    for index, seq in enumerate(profile):
        res.append(translate_seq_hpos(seq, index + 1))
    return res


def get_column(profile: list[list[str]], col_index: int) -> list[str]:
    res: list[str] = []
    for s in profile:
        res.append(s[col_index])
    return res


def get_place_hpos(column: list[str], seq_index: int) -> set[str]:
    col: list[str] = column.copy()
    self_item: str = col.pop(seq_index)
    if self_item[0] == 'G':
        return set()
    return set(col)


# def compute_dpos_distance(profile_a: list[str], profile_b: list[str]):
#
#     profile_a_hpos: list[list[str]] = translate_profile_hpos(profile_a)
#     profile_b_hpos: list[list[str]] = translate_profile_hpos(profile_b)

