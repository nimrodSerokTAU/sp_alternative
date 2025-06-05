from enums import DistanceType


def translate_seq_h(sequence: str, seq_index: int, distance_type: DistanceType) -> list[str]:
    res: list[str] = []
    s_count: int = 1
    for c in sequence:
        if c != '-':
            res.append(f'S^{seq_index}_{s_count}')
            s_count += 1
        else:
            if distance_type == DistanceType.D_POS:
                res.append(f'G^{seq_index}_{s_count - 1}')
            elif distance_type == DistanceType.D_SEQ:
                res.append('G')
            elif distance_type == DistanceType.D_SSP:
                res.append('-')
    return res


def translate_profile_naming(profile: list[str], distance_type: DistanceType) -> list[list[str]]:
    res: list[list[str]] = []
    for index, seq in enumerate(profile):
        res.append(translate_seq_h(seq, index + 1, distance_type))
    return res


def get_column(profile: list[list[str]], col_index: int) -> list[str]:
    res: list[str] = []
    for s in profile:
        res.append(s[col_index])
    return res


def get_place_h(column: list[str], seq_index: int) -> set[str] | None:
    col: list[str] = column.copy()
    self_item: str = col.pop(seq_index)
    if self_item[0] == 'G' or self_item[0] == '-':
        return None
    return set(col)


def get_place_d(set_a: set, set_b: set, distance_type: DistanceType) -> float:
    if distance_type == DistanceType.D_SSP:
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))
    else:
        symmetric_diff_size = len(set_a.difference(set_b)) + len(set_b.difference(set_a))
        return symmetric_diff_size / (len(set_a) + len(set_b))



def create_h_table(profile_naming: list[list[str]], distance_type: DistanceType) -> list[list[set[str]]]:
    seq_len: int = len(profile_naming[0])
    seq_count: int = len(profile_naming)
    h_table: list[list[set[str]]] = [[] for i in range(seq_count)]
    for col_index in range(seq_len):
        col: list[str] = get_column(profile_naming, col_index)
        for i in range(seq_count):
            h = get_place_h(col, i)
            if h is not None:
                if distance_type == DistanceType.D_POS:
                    h_table[i].append(h)
                elif distance_type == DistanceType.D_SEQ:
                    h_table[i].append(h)
                else:
                    h_table[i].append(h - {'-'})
    return h_table


def compute_distance(profile_a: list[str], profile_b: list[str], distance_type: DistanceType) -> float:
    d_list: list[float] = []
    numerator_sum: float = 0
    denominator_sum: float = 0
    profile_a_naming: list[list[str]] = translate_profile_naming(profile_a, distance_type)
    profile_b_naming: list[list[str]] = translate_profile_naming(profile_b, distance_type)
    seq_count: int = len(profile_a)
    profile_a_h: list[list[set[str]]] = create_h_table(profile_a_naming, distance_type)
    profile_b_h: list[list[set[str]]] = create_h_table(profile_b_naming, distance_type)
    for i in range(seq_count):
        for j in range(len(profile_a_h[i])):
            h_a_i: set[str] = profile_a_h[i][j]
            h_b_i: set[str] = profile_b_h[i][j]
            if len(h_a_i) > 0 or len(h_b_i) > 0:
                if distance_type == DistanceType.D_SSP:
                    numerator_sum += len(h_a_i.intersection(h_b_i))
                    denominator_sum += len(h_a_i.union(h_b_i))
                else:
                    d_i_j = get_place_d(h_a_i, h_b_i, distance_type)
                    d_list.append(d_i_j)
            else:
                d_list.append(0)
    if distance_type == DistanceType.D_SSP:
        return 1 - (numerator_sum / denominator_sum)
    elif len(d_list) > 0:
        return sum(d_list) / len(d_list)
    return -1
