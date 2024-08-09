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


def translate_profile_naming(profile: list[str]) -> list[list[str]]:
    res: list[list[str]] = []
    for index, seq in enumerate(profile):
        res.append(translate_seq_hpos(seq, index + 1))
    return res


def get_column(profile: list[list[str]], col_index: int) -> list[str]:
    res: list[str] = []
    for s in profile:
        res.append(s[col_index])
    return res


def get_place_hpos(column: list[str], seq_index: int) -> set[str] | None:
    col: list[str] = column.copy()
    self_item: str = col.pop(seq_index)
    if self_item[0] == 'G':
        return None
    return set(col)


def get_place_dpos(set_a: set, set_b: set) -> float:
    symmetric_diff_size = len(set_a.difference(set_b)) + len(set_b.difference(set_a))
    return symmetric_diff_size / (len(set_a) + len(set_b))


def create_hpos_table(profile_naming: list[list[str]]) -> list[list[set[str]]]:
    seq_len: int = len(profile_naming[0])
    seq_count: int = len(profile_naming)
    count_20 = 0
    hpos_table: list[list[set[str]]] = [[] for i in range(seq_count)]
    for col_index in range(seq_len):
        col: list[str] = get_column(profile_naming, col_index)
        for i in range(seq_count):
            hpos = get_place_hpos(col, i)
            if hpos is not None:
                hpos_table[i].append(hpos)
                if i == 20 and col_index > 600:
                    count_20 += 1
    return hpos_table


def compute_dpos_distance(profile_a: list[str], profile_b: list[str]) -> float:
    dpos_list: list[float] = []
    profile_a_naming: list[list[str]] = translate_profile_naming(profile_a)
    profile_b_naming: list[list[str]] = translate_profile_naming(profile_b)
    seq_count: int = len(profile_a)
    profile_a_hpos: list[list[set[str]]] = create_hpos_table(profile_a_naming)
    profile_b_hpos: list[list[set[str]]] = create_hpos_table(profile_b_naming)
    for i in range(seq_count):
        for j in range(len(profile_a_hpos[i])):
            hpos_a_i: set[str] = profile_a_hpos[i][j]
            hpos_b_i: set[str] = profile_b_hpos[i][j]
            if len(hpos_a_i) > 0 or len(hpos_b_i) > 0:
                dpos_i_j = get_place_dpos(hpos_a_i, hpos_b_i)
                dpos_list.append(dpos_i_j)
    if len(dpos_list) > 0:
        return sum(dpos_list) / len(dpos_list)
    return -1
