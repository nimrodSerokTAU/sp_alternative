from classes.config import Configuration
from classes.gap_interval import GapInterval
from classes.node import Node
from classes.sp_score import SPScore
from classes.unrooted_tree import create_a_tree_from_newick
from enums import SopCalcTypes
from multi_msa_service import calc_multiple_msa_sp_scores
from dpos import translate_profile_naming, get_column, get_place_hpos, compute_dpos_distance


def test_sp_perfect():
    configuration: Configuration = Configuration(-1, -1, -1, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDCQEGHI',
        'ARNDCQEGHI',
        'ARNDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # (5 + 7 + 7 + 8 + 13 + 7 + 6 + 8 + 10 + 5) * 3 = 76 * 3
    assert res == 228


def test_sp_no_gaps():
    configuration: Configuration = Configuration(-1, -5, 0, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDCQEGHI',
        'AANDCQEGAI',
        'AANDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # RRR -> RAA : 21 -> 5  -4 = 1 -> -20
    # HHH -> HHA : 30 -> 10 -4 = 6 -> -24
    assert res == 184


def test_sp_local_gaps():
    configuration: Configuration = Configuration(-1, -5, 0, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDCQ-GHI',
        'AANDCQ-GAI',
        'AANDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # EEE -> --E : 18 -> -6 * 2 = -12 -> -30
    assert res == 154


def test_naive_algo_case_a_subs_only():
    configuration: Configuration = Configuration(0, 0, 0, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 91  # subs only: 91


def test_naive_algo_case_a_subs_and_ge():
    configuration: Configuration = Configuration(0, -5, 0, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 41  # ge cost only: -50


def test_naive_algo_case_a_subs_and_ge_and_gs():
    configuration: Configuration = Configuration(-1, -5, -1, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 35  # gs cost only: -6


def test_compute_sp_s_and_sp_ge():  # our function
    configuration: Configuration = Configuration(0, -5, 0, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    sp_score_subs: int
    sp_score_gap_e: int
    sp_score_subs, sp_score_gap_e, sp_match_count, sp_missmatch_count = sp.compute_sp_s_and_sp_ge(profile)
    res = {'sp_score_subs': sp_score_subs, 'sp_score_gap_e': sp_score_gap_e}
    assert res == {'sp_score_subs': 91, 'sp_score_gap_e': -40}  # this is correct without gs cost


def test_onl_gap_open_and_ext_cost_same():
    configuration: Configuration = Configuration(-1, -5, -1, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: int = sp.compute_sp_gap_open(profile)
    assert res == -6


def test_compute_efficient_sp():
    configuration: Configuration = Configuration(-1, -5, -1, 'Blosum50')
    sp: SPScore = SPScore(configuration)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: float = sp.compute_efficient_sp(profile)
    assert res == 45


def test_translate_profile_hpos():
    profile: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    res = translate_profile_naming(profile)
    assert res == [
        ['S^1_1', 'S^1_2', 'S^1_3', 'S^1_4', 'S^1_5', 'S^1_6', 'S^1_7', 'G^1_7'],
        ['S^2_1', 'G^2_1', 'G^2_1', 'S^2_2', 'S^2_3', 'S^2_4', 'S^2_5', 'S^2_6'],
        ['S^3_1', 'G^3_1', 'G^3_1', 'S^3_2', 'G^3_2', 'S^3_3', 'S^3_4', 'S^3_5']]


def test_get_specific_hpos():
    profile: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    trans_prof = translate_profile_naming(profile)
    res_2 = []
    res_3 = []
    for col_inx in range(len(profile[0])):
        col = get_column(trans_prof, col_inx)
        hpos_2 = get_place_hpos(col, 1)
        res_2.append(hpos_2)
        hpos_3 = get_place_hpos(col, 2)
        res_3.append(hpos_3)
    print(res_2)
    print(res_3)
    assert res_2 == [{'S^3_1', 'S^1_1'}, None, None, {'S^1_4', 'S^3_2'}, {'G^3_2', 'S^1_5'}, {'S^3_3', 'S^1_6'},
                     {'S^1_7', 'S^3_4'}, {'G^1_7', 'S^3_5'}]
    assert res_3 == [{'S^1_1', 'S^2_1'}, None, None, {'S^1_4', 'S^2_2'}, None, {'S^2_4', 'S^1_6'}, {'S^2_5', 'S^1_7'},
                     {'S^2_6', 'G^1_7'}]


def test_compute_dpos_distance_for_same():
    profile_a: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    profile_b: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    res = compute_dpos_distance(profile_a, profile_b)
    assert res == 0


def test_compute_dpos_distance_for_diff():
    profile_a: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    profile_b: list[str] = [
        'AATAT-TG',
        'A-A-TTAG',
        'A--A-TAG'
    ]
    res = compute_dpos_distance(profile_a, profile_b)
    assert round(res, 3) == 0.417


def test_dpos_for_diff_length():
    profile_a: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    profile_b: list[str] = [
        'A-ATAT-TG',
        'A-A--TTAG',
        'A--A-T-AG'
    ]
    res = compute_dpos_distance(profile_a, profile_b)
    assert round(res, 3) == 0.639


def test_dpos_for_diff_length_case_b():
    profile_a: list[str] = [
        'AAT',
        '--T',
        '-CG'
    ]
    profile_b: list[str] = [
        'AAT-',
        '---T',
        '--CG'
    ]
    res = compute_dpos_distance(profile_a, profile_b)
    assert round(res, 3) == 0.5


def test_dpos_for_diff_length_case_c():
    profile_a: list[str] = [
        'ATA',
        '-T-',
        'CG-'
    ]
    profile_b: list[str] = [
        'AT-A',
        '--T-',
        '-CG-'
    ]
    res = compute_dpos_distance(profile_a, profile_b)
    assert round(res, 3) == 0.5


def test_tree_from_newick():
    newick = ('((((Macropus:0.051803,Monodelphis:0.066021):0.016682,Sarcophilus:0.068964):0.114355,'
              '((Echinops:0.104144,(Loxodonta:0.076474,Procavia:0.076193):0.011550):0.013015,((Choloepus:0.056091,'
              'Dasypus:0.040600):0.013681,(((((Callithrix:0.032131,((((Gorilla:0.007042,(Homo:0.002445,'
              'Pan:0.002450):0.001237):0.003689,Pongo:0.007508):0.002384,Nomascus:0.015696):0.004674,'
              'Macaca:0.013752):0.012205):0.029730,((Microcebus:0.037066,Otolemur:0.050935):0.008091,'
              'Tarsius:0.064938):0.007305):0.003377,Tupaia:0.090946):0.000699,(((Cavia:0.116826,(Dipodomys:0.080386,'
              '(Mus:0.040313,Rattus:0.033872):0.122329):0.013314):0.000298,Ictidomys:0.062932):0.011064,'
              '(Ochotona:0.087746,Oryctolagus:0.057769):0.035947):0.002130):0.006931,(Erinaceus:0.094727,'
              '(((((Bos:0.062659,Tursiops:0.024374):0.009782,Sus:0.064336):0.006134,Vicugna:0.049961):0.021643,'
              'Equus:0.046559):0.002728,(Sorex:0.126677,((Myotis:0.050452,Pteropus:0.047740):0.006495,'
              '(Felis:0.042414,(Canis:0.036675,(Mustela:0.027691,'
              'Ailuropoda:0.037553):0.004251):0.008141):0.019485):0.000485):0.003533):0.000528):0.005645):0.012900):0'
              '.002853):0.133251):0.019558,Ornithorhynchus:0.195576);')
    roots = create_a_tree_from_newick(newick)
    assert add_nodes_recursively_to_list(roots) == [
        {'children_ids': [], 'id': 77, 'keys': ['Ornithorhynchus']},
        {'children_ids': [4, 75], 'id': 76,
         'keys': ['Macropus', 'Monodelphis', 'Sarcophilus', 'Echinops', 'Loxodonta', 'Procavia', 'Choloepus', 'Dasypus',
                  'Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca', 'Microcebus', 'Otolemur',
                  'Tarsius', 'Tupaia', 'Cavia', 'Dipodomys', 'Mus', 'Rattus', 'Ictidomys', 'Ochotona', 'Oryctolagus',
                  'Erinaceus', 'Bos', 'Tursiops', 'Sus', 'Vicugna', 'Equus',
                  'Sorex', 'Myotis', 'Pteropus', 'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [9, 74], 'id': 75,
         'keys': ['Echinops', 'Loxodonta', 'Procavia', 'Choloepus', 'Dasypus', 'Callithrix', 'Gorilla', 'Homo', 'Pan',
                  'Pongo', 'Nomascus', 'Macaca', 'Microcebus', 'Otolemur', 'Tarsius', 'Tupaia', 'Cavia',
                  'Dipodomys', 'Mus', 'Rattus', 'Ictidomys', 'Ochotona', 'Oryctolagus', 'Erinaceus', 'Bos',
                  'Tursiops', 'Sus', 'Vicugna', 'Equus', 'Sorex', 'Myotis', 'Pteropus', 'Felis',
                  'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [12, 73], 'id': 74,
         'keys': ['Choloepus', 'Dasypus', 'Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca',
                  'Microcebus', 'Otolemur', 'Tarsius', 'Tupaia', 'Cavia', 'Dipodomys', 'Mus', 'Rattus',
                  'Ictidomys', 'Ochotona', 'Oryctolagus', 'Erinaceus', 'Bos', 'Tursiops', 'Sus', 'Vicugna',
                  'Equus', 'Sorex', 'Myotis', 'Pteropus', 'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [47, 72], 'id': 73,
         'keys': ['Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca', 'Microcebus', 'Otolemur',
                  'Tarsius', 'Tupaia', 'Cavia', 'Dipodomys', 'Mus', 'Rattus', 'Ictidomys', 'Ochotona',
                  'Oryctolagus', 'Erinaceus', 'Bos', 'Tursiops', 'Sus', 'Vicugna', 'Equus',
                  'Sorex', 'Myotis', 'Pteropus', 'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [48, 71], 'id': 72,
         'keys': ['Erinaceus', 'Bos', 'Tursiops', 'Sus', 'Vicugna', 'Equus', 'Sorex', 'Myotis', 'Pteropus',
                  'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [57, 70], 'id': 71,
         'keys': ['Bos', 'Tursiops', 'Sus', 'Vicugna', 'Equus', 'Sorex', 'Myotis', 'Pteropus', 'Felis', 'Canis',
                  'Mustela', 'Ailuropoda']},
        {'children_ids': [58, 69], 'id': 70,
         'keys': ['Sorex', 'Myotis', 'Pteropus', 'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [61, 68], 'id': 69, 'keys': ['Myotis', 'Pteropus', 'Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [62, 67], 'id': 68, 'keys': ['Felis', 'Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [63, 66], 'id': 67, 'keys': ['Canis', 'Mustela', 'Ailuropoda']},
        {'children_ids': [64, 65], 'id': 66, 'keys': ['Mustela', 'Ailuropoda']},
        {'children_ids': [], 'id': 65, 'keys': ['Ailuropoda']},
        {'children_ids': [], 'id': 64, 'keys': ['Mustela']},
        {'children_ids': [], 'id': 63, 'keys': ['Canis']},
        {'children_ids': [], 'id': 62, 'keys': ['Felis']},
        {'children_ids': [59, 60], 'id': 61, 'keys': ['Myotis', 'Pteropus']},
        {'children_ids': [], 'id': 60, 'keys': ['Pteropus']},
        {'children_ids': [], 'id': 59, 'keys': ['Myotis']},
        {'children_ids': [], 'id': 58, 'keys': ['Sorex']},
        {'children_ids': [55, 56], 'id': 57, 'keys': ['Bos', 'Tursiops', 'Sus', 'Vicugna', 'Equus']},
        {'children_ids': [], 'id': 56, 'keys': ['Equus']},
        {'children_ids': [53, 54], 'id': 55, 'keys': ['Bos', 'Tursiops', 'Sus', 'Vicugna']},
        {'children_ids': [], 'id': 54, 'keys': ['Vicugna']},
        {'children_ids': [51, 52], 'id': 53, 'keys': ['Bos', 'Tursiops', 'Sus']},
        {'children_ids': [], 'id': 52, 'keys': ['Sus']},
        {'children_ids': [49, 50], 'id': 51, 'keys': ['Bos', 'Tursiops']},
        {'children_ids': [], 'id': 50, 'keys': ['Tursiops']},
        {'children_ids': [], 'id': 49, 'keys': ['Bos']},
        {'children_ids': [], 'id': 48, 'keys': ['Erinaceus']},
        {'children_ids': [33, 46], 'id': 47,
         'keys': ['Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca', 'Microcebus', 'Otolemur',
                  'Tarsius', 'Tupaia', 'Cavia', 'Dipodomys', 'Mus', 'Rattus', 'Ictidomys', 'Ochotona', 'Oryctolagus']},
        {'children_ids': [42, 45], 'id': 46,
         'keys': ['Cavia', 'Dipodomys', 'Mus', 'Rattus', 'Ictidomys', 'Ochotona', 'Oryctolagus']},
        {'children_ids': [43, 44], 'id': 45, 'keys': ['Ochotona', 'Oryctolagus']},
        {'children_ids': [], 'id': 44, 'keys': ['Oryctolagus']},
        {'children_ids': [], 'id': 43, 'keys': ['Ochotona']},
        {'children_ids': [40, 41], 'id': 42, 'keys': ['Cavia', 'Dipodomys', 'Mus', 'Rattus', 'Ictidomys']},
        {'children_ids': [], 'id': 41, 'keys': ['Ictidomys']},
        {'children_ids': [34, 39], 'id': 40, 'keys': ['Cavia', 'Dipodomys', 'Mus', 'Rattus']},
        {'children_ids': [35, 38], 'id': 39, 'keys': ['Dipodomys', 'Mus', 'Rattus']},
        {'children_ids': [36, 37], 'id': 38, 'keys': ['Mus', 'Rattus']},
        {'children_ids': [], 'id': 37, 'keys': ['Rattus']},
        {'children_ids': [], 'id': 36, 'keys': ['Mus']},
        {'children_ids': [], 'id': 35, 'keys': ['Dipodomys']},
        {'children_ids': [], 'id': 34, 'keys': ['Cavia']},
        {'children_ids': [31, 32], 'id': 33,
         'keys': ['Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus',
                  'Macaca', 'Microcebus', 'Otolemur', 'Tarsius', 'Tupaia']},
        {'children_ids': [], 'id': 32, 'keys': ['Tupaia']},
        {'children_ids': [25, 30], 'id': 31,
         'keys': ['Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca', 'Microcebus', 'Otolemur',
                  'Tarsius']},
        {'children_ids': [28, 29], 'id': 30, 'keys': ['Microcebus', 'Otolemur', 'Tarsius']},
        {'children_ids': [], 'id': 29, 'keys': ['Tarsius']},
        {'children_ids': [26, 27], 'id': 28, 'keys': ['Microcebus', 'Otolemur']},
        {'children_ids': [], 'id': 27, 'keys': ['Otolemur']},
        {'children_ids': [], 'id': 26, 'keys': ['Microcebus']},
        {'children_ids': [13, 24], 'id': 25,
         'keys': ['Callithrix', 'Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca']},
        {'children_ids': [22, 23], 'id': 24, 'keys': ['Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus', 'Macaca']},
        {'children_ids': [], 'id': 23, 'keys': ['Macaca']},
        {'children_ids': [20, 21], 'id': 22, 'keys': ['Gorilla', 'Homo', 'Pan', 'Pongo', 'Nomascus']},
        {'children_ids': [], 'id': 21, 'keys': ['Nomascus']},
        {'children_ids': [18, 19], 'id': 20, 'keys': ['Gorilla', 'Homo', 'Pan', 'Pongo']},
        {'children_ids': [], 'id': 19, 'keys': ['Pongo']},
        {'children_ids': [14, 17], 'id': 18, 'keys': ['Gorilla', 'Homo', 'Pan']},
        {'children_ids': [15, 16], 'id': 17, 'keys': ['Homo', 'Pan']},
        {'children_ids': [], 'id': 16, 'keys': ['Pan']},
        {'children_ids': [], 'id': 15, 'keys': ['Homo']},
        {'children_ids': [], 'id': 14, 'keys': ['Gorilla']},
        {'children_ids': [], 'id': 13, 'keys': ['Callithrix']},
        {'children_ids': [10, 11], 'id': 12, 'keys': ['Choloepus', 'Dasypus']},
        {'children_ids': [], 'id': 11, 'keys': ['Dasypus']},
        {'children_ids': [], 'id': 10, 'keys': ['Choloepus']},
        {'children_ids': [5, 8], 'id': 9, 'keys': ['Echinops', 'Loxodonta', 'Procavia']},
        {'children_ids': [6, 7], 'id': 8, 'keys': ['Loxodonta', 'Procavia']},
        {'children_ids': [], 'id': 7, 'keys': ['Procavia']},
        {'children_ids': [], 'id': 6, 'keys': ['Loxodonta']},
        {'children_ids': [], 'id': 5, 'keys': ['Echinops']},
        {'children_ids': [2, 3], 'id': 4, 'keys': ['Macropus', 'Monodelphis', 'Sarcophilus']},
        {'children_ids': [], 'id': 3, 'keys': ['Sarcophilus']},
        {'children_ids': [0, 1], 'id': 2, 'keys': ['Macropus', 'Monodelphis']},
        {'children_ids': [], 'id': 1, 'keys': ['Monodelphis']},
        {'children_ids': [], 'id': 0, 'keys': ['Macropus']}]


def test_multi():
    configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
                                                 SopCalcTypes.EFFICIENT, 'tests/comparison_files',
                                                 False)
    calc_multiple_msa_sp_scores(configuration)


def add_nodes_recursively_to_list(nodes_to_add: list[Node]) -> list:
    all_nodes: list = []
    while len(nodes_to_add) > 0:
        node = nodes_to_add.pop()
        all_nodes.append({'id': node.id, 'children_ids': [x.id for x in node.children], 'keys': node.keys})
        nodes_to_add += node.children
    return all_nodes
