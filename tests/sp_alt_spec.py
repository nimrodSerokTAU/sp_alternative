import os
from pathlib import Path

from classes.compare_msas import msa_comp_main
from classes.config import Configuration
from classes.dist_labels_stats import DistanceLabelsStats
from classes.entropy_stats import EntropyStats
from classes.evo_model import EvoModel
from classes.gaps_stats import GapStats
from classes.global_alignment import GlobalAlign
from classes.kmer_stats import KMerStats
from classes.msa import MSA
from classes.msa_basic_stats import BasicStats
from classes.neighbor_joining import NeighborJoining
from classes.node import Node
from classes.rooted_tree import RootedTree
from classes.sop_stats import SopStats
from classes.sp_score import SPScore
from classes.tree_stats import calc_parsimony, TreeStats
from classes.unrooted_tree import create_a_tree_from_newick, UnrootedTree
from classes.w_sop_stats import WSopStats
from enums import SopCalcTypes, RootingMethods, WeightMethods, DistanceType
from multi_msa_service import multiple_msa_calc_features_and_labels
from distance_calc import translate_profile_naming, get_column, get_place_h, compute_distance
from ete3 import Tree, TreeNode

newick_of_AATF = (
    '((((Macropus:0.051803,Monodelphis:0.066021):0.016682,Sarcophilus:0.068964):0.114355,((Echinops:0.104144,'
    '(Loxodonta:0.076474,Procavia:0.076193):0.011550):0.013015,((Choloepus:0.056091,Dasypus:0.040600):0.013681,'
    '(((((Callithrix:0.032131,((((Gorilla:0.007042,(Homo:0.002445,Pan:0.002450):0.001237):0.003689,Pongo:0.007508):0.002384,'
    'Nomascus:0.015696):0.004674,Macaca:0.013752):0.012205):0.029730,((Microcebus:0.037066,Otolemur:0.050935):0.008091,'
    'Tarsius:0.064938):0.007305):0.003377,Tupaia:0.090946):0.000699,(((Cavia:0.116826,(Dipodomys:0.080386,(Mus:0.040313,'
    'Rattus:0.033872):0.122329):0.013314):0.000298,Ictidomys:0.062932):0.011064,(Ochotona:0.087746,Oryctolagus:0.057769)'
    ':0.035947):0.002130):0.006931,(Erinaceus:0.094727,(((((Bos:0.062659,Tursiops:0.024374):0.009782,Sus:0.064336)'
    ':0.006134,Vicugna:0.049961):0.021643,Equus:0.046559):0.002728,(Sorex:0.126677,((Myotis:0.050452,Pteropus:0.047740)'
    ':0.006495,(Felis:0.042414,(Canis:0.036675,(Mustela:0.027691,Ailuropoda:0.037553):0.004251):0.008141):0.019485)'
    ':0.000485):0.003533):0.000528):0.005645):0.012900):0.002853):0.133251):0.019558,Ornithorhynchus:0.195576);'
)

matrix_case_nj: list[list[float]] = [
    [0, 5, 9, 9, 8],
    [5, 0, 10, 10, 9],
    [9, 10, 0, 8, 7],
    [9, 10, 8, 0, 3],
    [8, 9, 7, 3, 0],
]
keys_case_nj = ['a', 'b', 'c', 'd', 'e']


def test_sp_perfect():
    configuration: Configuration = Configuration([EvoModel(-1, -1, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDCQEGHI',
        'ARNDCQEGHI',
        'ARNDCQEGHI']
    res: list[int] = sp.compute_naive_sp_score(profile)
    # (5 + 7 + 7 + 8 + 13 + 7 + 6 + 8 + 10 + 5) * 3 = 76 * 3
    assert res[0] == 228


def test_sp_no_gaps():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDCQEGHI',
        'AANDCQEGAI',
        'AANDCQEGHI']
    res: list[int] = sp.compute_naive_sp_score(profile)
    # RRR -> RAA : 21 -> 5  -4 = 1 -> -20
    # HHH -> HHA : 30 -> 10 -4 = 6 -> -24
    assert res[0] == 184


def test_sp_local_gaps():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDCQ-GHI',
        'AANDCQ-GAI',
        'AANDCQEGHI']
    res: list[int] = sp.compute_naive_sp_score(profile)
    # EEE -> --E : 18 -> -6 * 2 = -12 -> -30
    assert res[0] == 154


def test_naive_algo_case_a_subs_only():
    configuration: Configuration = Configuration([EvoModel(0, 0, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: list[int] = sp.compute_naive_sp_score(profile, [[1, 1, 1]])
    assert res[0] == 91  # subs only: 91


def test_naive_algo_case_a_subs_and_ge():
    configuration: Configuration = Configuration([EvoModel(0, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: list[int] = sp.compute_naive_sp_score(profile, [[1, 1, 1]])
    assert res[0] == 41  # ge cost only: -50


def test_naive_algo_case_a_subs_and_ge_and_gs():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: list[int] = sp.compute_naive_sp_score(profile)
    assert res[0] == 35  # gs cost only: -6


def test_naive_algo_case_a_subs_and_ge_and_gs_with_weights():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: list[int] = sp.compute_naive_sp_score(profile, [[1, 1, 1], [2, 2, 2]])
    assert res == [35, 140]  # gs cost only: -6


def test_compute_sp_s_and_sp_ge():  # our function
    configuration: Configuration = Configuration([EvoModel(0, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    sp_score_subs: int
    sp_score_gap_e: int
    sp_score_subs, sp_score_gap_e, sp_match_count, sp_missmatch_count, ge_count = sp.compute_sp_s_and_sp_ge(profile)
    naive_score = sp.compute_naive_sp_score(profile)
    res = {'sp_score_subs': sp_score_subs, 'sp_score_gap_e': sp_score_gap_e, 'total': sp_score_subs + sp_score_gap_e}
    assert res == {'sp_score_subs': 91, 'sp_score_gap_e': -50, 'total':naive_score[0]}  # this is correct without gs cost


def test_only_gap_open_and_ext_cost_same():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # ARNDC---HI
    # AA-DCQ--AI
    # ----> 2

    # ARNDC---HI
    # AA--CQEGHI
    # ----> 2

    # AA-DCQ--AI
    # AA--CQEGHI
    # ----> 2

    res: tuple [int, int] = sp.compute_sp_gap_open(profile)
    assert res == (-6, 6)


def test_compute_efficient_sp():
    configuration: Configuration = Configuration([EvoModel(-1, -5, 'Blosum50')])
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: float = sp.compute_efficient_sp(profile)
    naive_res: list[float] = sp.compute_naive_sp_score(profile)
    assert res == naive_res[0]


def test_compare_naive_sop_to_efficient():
    configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                                 SopCalcTypes.EFFICIENT, 'comparison_files')
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        '-EETTEESLKRIVADNENRAEQVHLYLSTTFVIADPEPKYGIVRSKDMNWYEQKTHKFLGMGPVLGVQFAF',
        'YEETSEESL-RIAADNENRAE-VHLYLGTNFVIADPEPKW--LRSKDVNWYDQRTH-FLGMGPVLGIQFLI',
        'YEETSEES----VADNENRAE-VHLILSTNFVIADPEPKWG-LRSKDMNWYDQRTH--LGMGPVLGIQFLF',
        'YEETSEESLKRIVADNENRAEKVHLILSTNFVIADPEPKWG--RSKDMNWYDQRTHKFLGMGPVLGIQFLF']
    res_naive: list[int] = sp.compute_naive_sp_score(profile)
    res_efficient: float = sp.compute_efficient_sp(profile)
    assert res_naive[0] == res_efficient


def test_compare_naive_sop_to_efficient_min_example():
    configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                                 SopCalcTypes.EFFICIENT, 'comparison_files')
    sp: SPScore = SPScore(configuration.models[0])
    profile: list[str] = [
        'LLKYR-K',
        'Y--ERAK',
        'YL----K',
        'YLKE-AK']
    res_naive: list[int] = sp.compute_naive_sp_score(profile)
    sp_score_subs, sp_score_gap_o, sp_score_gap_e = sp.compute_naive_sp_score_per_col(profile)
    res_efficient: float = sp.compute_efficient_sp(profile)
    assert res_naive[0] == res_efficient


def test_translate_profile_hpos():
    profile: list[str] = [
        'AATATTG-',
        'A--ATTAG',
        'A--A-TAG'
    ]
    res = translate_profile_naming(profile, DistanceType.D_POS)
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
    trans_prof = translate_profile_naming(profile, DistanceType.D_POS)
    res_2 = []
    res_3 = []
    for col_inx in range(len(profile[0])):
        col = get_column(trans_prof, col_inx)
        hpos_2 = get_place_h(col, 1)
        res_2.append(hpos_2)
        hpos_3 = get_place_h(col, 2)
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
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
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
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.417


def compute_dpos_distance_for_diff_case_b():
    profile_a: list[str] = [
        'ABCDEFGH',
        'A-CDEFGH',
        'AB-DEFGH',
        'AB-DEFGH'
    ]
    profile_b: list[str] = [
        'ABCDEFGH',
        'A-CDEFGH',
        'AB-DEFGH',
        'AB-DEFGH'
    ]
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.417


def test_dpos_for_diff_length_case_a():
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
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.639


def test_dpos_for_diff_length_case_qu_a():
    profile_a: list[str] = [
        'GCATCATT-G',
        'GC---ATTAG',
        'GC---AT-AG'
    ]
    profile_b: list[str] = [
        'GCATCATT-G',
        'GCA---TTAG',
        'GCA----TAG'
    ]
    profile_c: list[str] = [
        'GCATCATT-G-',
        'GC---ATT-AG',
        'GC---A-TA-G'
    ]
    res_ab = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    res_ac = compute_distance(profile_a, profile_c, DistanceType.D_POS)
    assert round(res_ab, 3) == 0.364
    assert round(res_ac, 3) == 0.295


def test_dpos_for_diff_length_case_qu_b():
    true_profile: list[str] = [
        'GAAGTTAGACATC',
        'GA--------ATC',
        'GA--------ATG',
        'GT--------ATG',
        'GT--------ATC',
    ]
    profile_a: list[str] = [
        'GAAGTTAGACATC',
        'GAA--------TC',
        'GAA--------TG',
        'GTA--------TG',
        'GTA--------TC',
    ]
    profile_b: list[str] = [
        'GAAGTTAGACATC',
        'GA----A----TC',
        'GA----A----TG',
        'GT----A----TG',
        'GT----A----TC',
    ]
    profile_c: list[str] = [
        'GAAGTTAGACATC',
        'GA------A--TC',
        'GA------A--TG',
        'GT------A--TG',
        'GT------A--TC',
    ]

    res_a = compute_distance(true_profile, profile_a, DistanceType.D_POS)
    res_b = compute_distance(true_profile, profile_b, DistanceType.D_POS)
    res_c = compute_distance(true_profile, profile_c, DistanceType.D_POS)
    assert round(res_a, 3) == 0.303
    assert round(res_b, 3) == 0.182
    assert round(res_c, 3) == 0.121


def test_dseq_for_diff_length_case_qu_a():
    profile_a: list[str] = [
        'GCATCATT-G',
        'GC---ATTAG',
        'GC---AT-AG'
    ]
    profile_b: list[str] = [
        'GCATCATT-G',
        'GCA---TTAG',
        'GCA----TAG'
    ]
    profile_c: list[str] = [
        'GCATCATT-G-',
        'GC---ATT-AG',
        'GC---A-TA-G'
    ]
    res_ab = compute_distance(profile_a, profile_b, DistanceType.D_SEQ)
    res_ac = compute_distance(profile_a, profile_c, DistanceType.D_SEQ)
    assert round(res_ab, 3) == 0.273
    assert round(res_ac, 3) == 0.265


def test_dssp_for_diff_length_case_qu_a():
    profile_a: list[str] = [
        'GCATCATT-G',
        'GC---ATTAG',
        'GC---AT-AG'
    ]
    profile_b: list[str] = [
        'GCATCATT-G',
        'GCA---TTAG',
        'GCA----TAG'
    ]
    profile_c: list[str] = [
        'GCATCATT-G-',
        'GC---ATT-AG',
        'GC---A-TA-G'
    ]
    res_ab = compute_distance(profile_a, profile_b, DistanceType.D_SSP)
    res_ac = compute_distance(profile_a, profile_c, DistanceType.D_SSP)
    assert round(res_ab, 3) == 0.381
    assert round(res_ac, 3) == 0.4


def test_d_seq_for_diff_length_case_qu_b():
    true_profile: list[str] = [
        'GAAGTTAGACATC',
        'GA--------ATC',
        'GA--------ATG',
        'GT--------ATG',
        'GT--------ATC',
    ]
    profile_a: list[str] = [
        'GAAGTTAGACATC',
        'GAA--------TC',
        'GAA--------TG',
        'GTA--------TG',
        'GTA--------TC',
    ]
    profile_b: list[str] = [
        'GAAGTTAGACATC',
        'GA----A----TC',
        'GA----A----TG',
        'GT----A----TG',
        'GT----A----TC',
    ]
    profile_c: list[str] = [
        'GAAGTTAGACATC',
        'GA------A--TC',
        'GA------A--TG',
        'GT------A--TG',
        'GT------A--TC',
    ]

    res_a = compute_distance(true_profile, profile_a, DistanceType.D_SEQ)
    res_b = compute_distance(true_profile, profile_b, DistanceType.D_SEQ)
    res_c = compute_distance(true_profile, profile_c, DistanceType.D_SEQ)
    assert round(res_a, 3) == 0.091
    assert round(res_b, 3) == 0.091
    assert round(res_c, 3) == 0.091


def test_d_ssp_for_diff_length_case_qu_b():
    true_profile: list[str] = [
        'GAAGTTAGACATC',
        'GA--------ATC',
        'GA--------ATG',
        'GT--------ATG',
        'GT--------ATC',
    ]
    profile_a: list[str] = [
        'GAAGTTAGACATC',
        'GAA--------TC',
        'GAA--------TG',
        'GTA--------TG',
        'GTA--------TC',
    ]
    profile_b: list[str] = [
        'GAAGTTAGACATC',
        'GA----A----TC',
        'GA----A----TG',
        'GT----A----TG',
        'GT----A----TC',
    ]
    profile_c: list[str] = [
        'GAAGTTAGACATC',
        'GA------A--TC',
        'GA------A--TG',
        'GT------A--TG',
        'GT------A--TC',
    ]

    res_a = compute_distance(true_profile, profile_a, DistanceType.D_SSP)
    res_b = compute_distance(true_profile, profile_b, DistanceType.D_SSP)
    res_c = compute_distance(true_profile, profile_c, DistanceType.D_SSP)
    assert round(res_a, 3) == 0.148
    assert round(res_b, 3) == 0.148
    assert round(res_c, 3) == 0.148

def test_dpos_for_diff_length_case_c():
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
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.5


def test_dpos_for_diff_length_case_d():
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
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.5


def test_tree_from_newick():
    roots, all_nodes = create_a_tree_from_newick(newick_of_AATF)
    assert add_nodes_dto_to_list(all_nodes) == [
        {'children_ids': [4, 75], 'id': 76, 'keys': {
            'Ailuropoda', 'Bos', 'Callithrix', 'Canis', 'Cavia', 'Choloepus', 'Dasypus', 'Dipodomys', 'Echinops',
            'Equus', 'Erinaceus', 'Felis', 'Gorilla', 'Homo', 'Ictidomys', 'Loxodonta', 'Macaca', 'Macropus',
            'Microcebus', 'Monodelphis', 'Mus', 'Mustela', 'Myotis', 'Nomascus', 'Ochotona', 'Oryctolagus', 'Otolemur',
            'Pan', 'Pongo', 'Procavia', 'Pteropus', 'Rattus', 'Sarcophilus', 'Sorex', 'Sus', 'Tarsius', 'Tupaia',
            'Tursiops', 'Vicugna'}},
        {'children_ids': [9, 74], 'id': 75, 'keys': {
            'Ailuropoda', 'Bos', 'Callithrix', 'Canis', 'Cavia', 'Choloepus', 'Dasypus', 'Dipodomys', 'Echinops',
            'Equus', 'Erinaceus', 'Felis', 'Gorilla', 'Homo', 'Ictidomys', 'Loxodonta', 'Macaca', 'Microcebus',
            'Mus', 'Mustela', 'Myotis', 'Nomascus', 'Ochotona', 'Oryctolagus', 'Otolemur', 'Pan', 'Pongo', 'Procavia',
            'Pteropus', 'Rattus', 'Sorex', 'Sus', 'Tarsius', 'Tupaia', 'Tursiops', 'Vicugna'}},
        {'children_ids': [12, 73], 'id': 74, 'keys': {
            'Ailuropoda', 'Bos', 'Callithrix', 'Canis', 'Cavia', 'Choloepus', 'Dasypus', 'Dipodomys', 'Equus',
            'Erinaceus', 'Felis', 'Gorilla', 'Homo', 'Ictidomys', 'Macaca', 'Microcebus', 'Mus', 'Mustela', 'Myotis',
            'Nomascus', 'Ochotona', 'Oryctolagus', 'Otolemur', 'Pan', 'Pongo', 'Pteropus', 'Rattus', 'Sorex', 'Sus',
            'Tarsius', 'Tupaia', 'Tursiops', 'Vicugna'}},
        {'children_ids': [47, 72], 'id': 73, 'keys': {
            'Ailuropoda', 'Bos', 'Callithrix', 'Canis', 'Cavia', 'Dipodomys', 'Equus', 'Erinaceus', 'Felis', 'Gorilla',
            'Homo', 'Ictidomys', 'Macaca', 'Microcebus', 'Mus', 'Mustela', 'Myotis', 'Nomascus', 'Ochotona',
            'Oryctolagus', 'Otolemur', 'Pan', 'Pongo', 'Pteropus', 'Rattus', 'Sorex', 'Sus', 'Tarsius', 'Tupaia',
            'Tursiops', 'Vicugna'}},
        {'children_ids': [33, 46], 'id': 47, 'keys': {
            'Callithrix', 'Cavia', 'Dipodomys', 'Gorilla', 'Homo', 'Ictidomys', 'Macaca', 'Microcebus', 'Mus',
            'Nomascus', 'Ochotona', 'Oryctolagus', 'Otolemur', 'Pan', 'Pongo', 'Rattus', 'Tarsius', 'Tupaia'}},
        {'children_ids': [48, 71], 'id': 72, 'keys': {
            'Ailuropoda', 'Bos', 'Canis', 'Equus', 'Erinaceus', 'Felis', 'Mustela', 'Myotis', 'Pteropus', 'Sorex',
            'Sus', 'Tursiops', 'Vicugna'}},
        {'children_ids': [57, 70], 'id': 71, 'keys': {
            'Ailuropoda', 'Bos', 'Canis', 'Equus', 'Felis', 'Mustela', 'Myotis', 'Pteropus', 'Sorex', 'Sus', 'Tursiops',
            'Vicugna'}},
        {'children_ids': [31, 32], 'id': 33, 'keys': {
            'Callithrix', 'Gorilla', 'Homo', 'Macaca', 'Microcebus', 'Nomascus', 'Otolemur', 'Pan', 'Pongo', 'Tarsius',
            'Tupaia'}},
        {'children_ids': [25, 30], 'id': 31, 'keys': {
            'Callithrix', 'Gorilla', 'Homo', 'Macaca', 'Microcebus', 'Nomascus', 'Otolemur', 'Pan', 'Pongo',
            'Tarsius'}},
        {'children_ids': [58, 69], 'id': 70, 'keys': {
            'Ailuropoda', 'Canis', 'Felis', 'Mustela', 'Myotis', 'Pteropus', 'Sorex'}},
        {'children_ids': [42, 45], 'id': 46, 'keys': {
            'Cavia', 'Dipodomys', 'Ictidomys', 'Mus', 'Ochotona', 'Oryctolagus', 'Rattus'}},
        {'children_ids': [13, 24], 'id': 25, 'keys': {
            'Callithrix', 'Gorilla', 'Homo', 'Macaca', 'Nomascus', 'Pan', 'Pongo'}},
        {'children_ids': [61, 68], 'id': 69, 'keys': {'Pteropus', 'Mustela', 'Canis', 'Felis', 'Ailuropoda', 'Myotis'}},
        {'children_ids': [22, 23], 'id': 24, 'keys': {'Pan', 'Gorilla', 'Macaca', 'Homo', 'Pongo', 'Nomascus'}},
        {'children_ids': [55, 56], 'id': 57, 'keys': {'Vicugna', 'Sus', 'Equus', 'Bos', 'Tursiops'}},
        {'children_ids': [40, 41], 'id': 42, 'keys': {'Cavia', 'Mus', 'Dipodomys', 'Ictidomys', 'Rattus'}},
        {'children_ids': [20, 21], 'id': 22, 'keys': {'Pan', 'Gorilla', 'Homo', 'Pongo', 'Nomascus'}},
        {'children_ids': [62, 67], 'id': 68, 'keys': {'Canis', 'Felis', 'Mustela', 'Ailuropoda'}},
        {'children_ids': [53, 54], 'id': 55, 'keys': {'Sus', 'Vicugna', 'Bos', 'Tursiops'}},
        {'children_ids': [34, 39], 'id': 40, 'keys': {'Dipodomys', 'Cavia', 'Rattus', 'Mus'}},
        {'children_ids': [18, 19], 'id': 20, 'keys': {'Homo', 'Pongo', 'Pan', 'Gorilla'}},
        {'children_ids': [63, 66], 'id': 67, 'keys': {'Canis', 'Mustela', 'Ailuropoda'}},
        {'children_ids': [51, 52], 'id': 53, 'keys': {'Sus', 'Bos', 'Tursiops'}},
        {'children_ids': [35, 38], 'id': 39, 'keys': {'Dipodomys', 'Rattus', 'Mus'}},
        {'children_ids': [28, 29], 'id': 30, 'keys': {'Microcebus', 'Otolemur', 'Tarsius'}},
        {'children_ids': [14, 17], 'id': 18, 'keys': {'Homo', 'Pan', 'Gorilla'}},
        {'children_ids': [5, 8], 'id': 9, 'keys': {'Procavia', 'Loxodonta', 'Echinops'}},
        {'children_ids': [2, 3], 'id': 4, 'keys': {'Monodelphis', 'Macropus', 'Sarcophilus'}},
        {'children_ids': [64, 65], 'id': 66, 'keys': {'Mustela', 'Ailuropoda'}},
        {'children_ids': [59, 60], 'id': 61, 'keys': {'Pteropus', 'Myotis'}},
        {'children_ids': [49, 50], 'id': 51, 'keys': {'Bos', 'Tursiops'}},
        {'children_ids': [43, 44], 'id': 45, 'keys': {'Oryctolagus', 'Ochotona'}},
        {'children_ids': [36, 37], 'id': 38, 'keys': {'Rattus', 'Mus'}},
        {'children_ids': [26, 27], 'id': 28, 'keys': {'Microcebus', 'Otolemur'}},
        {'children_ids': [15, 16], 'id': 17, 'keys': {'Homo', 'Pan'}},
        {'children_ids': [10, 11], 'id': 12, 'keys': {'Dasypus', 'Choloepus'}},
        {'children_ids': [6, 7], 'id': 8, 'keys': {'Procavia', 'Loxodonta'}},
        {'children_ids': [0, 1], 'id': 2, 'keys': {'Monodelphis', 'Macropus'}},
        {'children_ids': [], 'id': 77, 'keys': {'Ornithorhynchus'}},
        {'children_ids': [], 'id': 65, 'keys': {'Ailuropoda'}},
        {'children_ids': [], 'id': 64, 'keys': {'Mustela'}},
        {'children_ids': [], 'id': 63, 'keys': {'Canis'}},
        {'children_ids': [], 'id': 62, 'keys': {'Felis'}},
        {'children_ids': [], 'id': 60, 'keys': {'Pteropus'}},
        {'children_ids': [], 'id': 59, 'keys': {'Myotis'}},
        {'children_ids': [], 'id': 58, 'keys': {'Sorex'}},
        {'children_ids': [], 'id': 56, 'keys': {'Equus'}},
        {'children_ids': [], 'id': 54, 'keys': {'Vicugna'}},
        {'children_ids': [], 'id': 52, 'keys': {'Sus'}},
        {'children_ids': [], 'id': 50, 'keys': {'Tursiops'}},
        {'children_ids': [], 'id': 49, 'keys': {'Bos'}},
        {'children_ids': [], 'id': 48, 'keys': {'Erinaceus'}},
        {'children_ids': [], 'id': 44, 'keys': {'Oryctolagus'}},
        {'children_ids': [], 'id': 43, 'keys': {'Ochotona'}},
        {'children_ids': [], 'id': 41, 'keys': {'Ictidomys'}},
        {'children_ids': [], 'id': 37, 'keys': {'Rattus'}},
        {'children_ids': [], 'id': 36, 'keys': {'Mus'}},
        {'children_ids': [], 'id': 35, 'keys': {'Dipodomys'}},
        {'children_ids': [], 'id': 34, 'keys': {'Cavia'}},
        {'children_ids': [], 'id': 32, 'keys': {'Tupaia'}},
        {'children_ids': [], 'id': 29, 'keys': {'Tarsius'}},
        {'children_ids': [], 'id': 27, 'keys': {'Otolemur'}},
        {'children_ids': [], 'id': 26, 'keys': {'Microcebus'}},
        {'children_ids': [], 'id': 23, 'keys': {'Macaca'}},
        {'children_ids': [], 'id': 21, 'keys': {'Nomascus'}},
        {'children_ids': [], 'id': 19, 'keys': {'Pongo'}},
        {'children_ids': [], 'id': 16, 'keys': {'Pan'}},
        {'children_ids': [], 'id': 15, 'keys': {'Homo'}},
        {'children_ids': [], 'id': 14, 'keys': {'Gorilla'}},
        {'children_ids': [], 'id': 13, 'keys': {'Callithrix'}},
        {'children_ids': [], 'id': 11, 'keys': {'Dasypus'}},
        {'children_ids': [], 'id': 10, 'keys': {'Choloepus'}},
        {'children_ids': [], 'id': 7, 'keys': {'Procavia'}},
        {'children_ids': [], 'id': 6, 'keys': {'Loxodonta'}},
        {'children_ids': [], 'id': 5, 'keys': {'Echinops'}},
        {'children_ids': [], 'id': 3, 'keys': {'Sarcophilus'}},
        {'children_ids': [], 'id': 1, 'keys': {'Monodelphis'}},
        {'children_ids': [], 'id': 0, 'keys': {'Macropus'}}]


def test_dpos_for_diff_length_case_e():
    profile_a: list[str] = [
        'ATAG',
        '-T-G',
        'CG-G'
    ]
    profile_b: list[str] = [
        'AT-AG',
        '--T-G',
        '-CG-G'
    ]
    res = compute_distance(profile_a, profile_b, DistanceType.D_POS)
    assert round(res, 3) == 0.333


def test_multi():
    configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                                 SopCalcTypes.EFFICIENT, 'tests/comparison_files',
                                                 {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG,
                                                  WeightMethods.CLUSTAL_MID_ROOT,
                                                  WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})
    multiple_msa_calc_features_and_labels(configuration)


def add_nodes_recursively_to_list(nodes_to_add: list[Node]) -> list:
    all_nodes: list = []
    while len(nodes_to_add) > 0:
        node = nodes_to_add.pop()
        all_nodes.append({'id': node.id, 'children_ids': [x.id for x in node.children], 'keys': node.keys})
        nodes_to_add += node.children
    return all_nodes


def add_nodes_dto_to_list(nodes_list: list[Node]) -> list:
    all_nodes_dto: list = []
    while len(nodes_list) > 0:
        node = nodes_list.pop()
        all_nodes_dto.append({'id': node.id, 'children_ids': [x.id for x in node.children], 'keys': node.keys})
    all_nodes_dto.sort(key=lambda x: x['id'], reverse=True)
    all_nodes_dto.sort(key=lambda x: len(x['keys']), reverse=True)
    return all_nodes_dto


def test_unrooting_and_tree_from_ours():
    tree_from_newick = Tree(newick_of_AATF)
    tree_from_newick.unroot()
    # unrooted_newick_ete = tree_from_newick.write()
    our_unrooted_tree = UnrootedTree.create_from_newick_str(newick_of_AATF)
    tree_from_ours = build_e_tree_from_ours(our_unrooted_tree)
    rf, max_parts, common_attrs, edges1, edges2, discard_t1, discard_t2 = tree_from_newick.robinson_foulds(
        t2=tree_from_ours,
        unrooted_trees=True)
    assert rf == 0


def test_tree_comparison_case_a():
    tree_from_newick = Tree(newick_of_AATF)
    leaf_names: list[str] = tree_from_newick.get_leaf_names()
    tree_from_newick.unroot()
    branches_a: list[str] = map_branches_of_tree(tree_from_newick, set(leaf_names), sorted(leaf_names)[0])
    print_branches_ordered_list(branches_a)
    our_unrooted_tree = UnrootedTree.create_from_newick_str(newick_of_AATF)
    branches_b: list[str] = list(our_unrooted_tree.get_internal_edges_set())
    print_branches_ordered_list(branches_b)


def test_rf_for_nj_using_ours():
    unrooted_tree = UnrootedTree.create_from_newick_str(newick_of_AATF)
    msa = MSA('AATF')
    msa.read_me_from_fasta(Path('./comparison_files/AATF/MSA.MAFFT.aln.With_Names'))
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                          SopCalcTypes.EFFICIENT, 'comparison_files')
    sp_score: SPScore = SPScore(config.models[0])
    msa.build_nj_tree()
    nj_unrooted_tree = msa.tree
    rf = unrooted_tree.calc_rf(nj_unrooted_tree)
    assert rf == 34


def test_rf_for_nj_using_newick():
    tree_from_newick = Tree(newick_of_AATF)
    tree_from_newick.unroot()
    nj_newick = (
        '(((((((Macropus:0,Nomascus:0):0,(Choloepus:0,Myotis:0):0):0,(Tarsius:0,Erinaceus:0):0):0,(Callithrix:0,Mustela:0):0):0,'
        '((((Monodelphis:0,Gorilla:0):0,(Oryctolagus:0,Tursiops:0):0):0,(Cavia:0,Ailuropoda:0):0):0,((Echinops:0,Microcebus:0):0,'
        '((Pan:0,Sorex:0):0,(Rattus:0,Canis:0):0):0):0):0):0,(((Macaca:0,Ornithorhynchus:0):0,(Ictidomys:0,Equus:0):0):0,(Ochotona:0,Sus:0):0):0):0,'
        '((((Sarcophilus:0,Homo:0):0,(Mus:0,Pteropus:0):0):0,(Tupaia:0,Felis:0):0):0,((Procavia:0,Dasypus:0):0,(Pongo:0,'
        'Bos:0):0):0):0,'
        '((Loxodonta:0,Otolemur:0):0,(Dipodomys:0,Vicugna:0):0):0);')
    nj_tree_from_newick = Tree(nj_newick)
    rf, max_parts, common_attrs, edges1, edges2, discard_t1, discard_t2 = tree_from_newick.robinson_foulds(
        t2=nj_tree_from_newick,
        unrooted_trees=True)
    print("RF distance is %s over a total of %s" % (rf, max_parts))
    print("Partitions in tree2 that were not found in tree1:", edges1 - edges2)
    print("Partitions in tree1 that were not found in tree2:", edges2 - edges1)
    a = 1
    assert rf == 74


def test_neighbor_joining():
    nodes: list[Node] = []
    for key in keys_case_nj:
        nodes.append(Node(node_id=len(nodes), keys={key}, children=[], branch_length=1, children_bl_sum=0))
    tree_calculation: UnrootedTree = NeighborJoining(matrix_case_nj, nodes).tree_res
    bl_list: list[float] = tree_calculation.get_branches_lengths_list()
    bl_list.sort()
    assert bl_list == [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0]


def test_parsimony():  # TODO: continue from here
    n_a: Node = Node(node_id=0, keys={'a'}, children=[], children_bl_sum=0)
    n_b: Node = Node(node_id=1, keys={'b'}, children=[], children_bl_sum=0)
    n_c: Node = Node(node_id=2, keys={'c'}, children=[], children_bl_sum=0)
    n_d: Node = Node(node_id=3, keys={'d'}, children=[], children_bl_sum=0)
    n_e: Node = Node(node_id=4, keys={'e'}, children=[], children_bl_sum=0)
    n_a_b: Node = Node.create_from_children([n_a, n_b], 5)
    n_a_b_c: Node = Node.create_from_children([n_a_b, n_c], 6)
    anchor: Node = Node.create_from_children([n_a_b_c, n_d, n_e], 7)
    all_nodes: list[Node] = [n_a, n_b, n_c, n_d, n_e, n_a_b, n_a_b_c, anchor]
    aln: list[str] = [
        'AYCDDDW',
        'AVVDDDW',
        'AYCDDDW',
        'AVVDDDW',
        'APVDDDW'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    res = calc_parsimony(UnrootedTree(anchor=anchor, all_nodes=all_nodes), aln, names)
    assert res == [0, 3, 2, 0, 0, 0, 0]


def test_msa_stats():
    aln: list[str] = [
        'AT-CGC-GGT',
        'ACATG-T-GA',
        'AT-CG--GGT',
        'ATC-GA-GGA',
        'TTATGCTGGA'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    true_aln: list[str] = [
        'AT-CGC-GGT',
        'ACATG-TG-A',
        'AT-CG--GGT',
        'AT-CGA-GGA',
        'TTATGCTGGA'
    ]
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                          SopCalcTypes.EFFICIENT, 'comparison_files',
                                          {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG,
                                           WeightMethods.CLUSTAL_MID_ROOT,
                                           WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})
    true_msa: MSA = create_msa_from_seqs_and_names('true', true_aln, names)
    inferred_msa: MSA = create_msa_from_seqs_and_names('inferred', aln, names)

    sp: SPScore = SPScore(config.models[0])
    basic_stats = BasicStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len(),
                             ['code', 'taxa_num', 'msa_len'])
    assert basic_stats.get_my_features_as_list() == ['inferred', 5, 10]
    dist_labels_stats = DistanceLabelsStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    dist_labels_stats.set_my_dpos_dist_from_true(true_msa.sequences, inferred_msa.sequences)
    assert dist_labels_stats.get_my_features_as_list() == ['inferred', 0.182, 0.132, 0.134]

    entropy_stats = EntropyStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    entropy_stats.calc_entropy(inferred_msa.sequences)
    assert entropy_stats.get_my_features_as_list() == ['inferred', 0.4, 9, 0.364, 0, 0.092, 0.0, 0.637, 0.0, 0.693]

    gaps_stats = GapStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    gaps_stats.calc_gaps_values(inferred_msa.sequences)
    assert gaps_stats.get_my_features_as_list() == ['inferred', 1.125, 8, 7, 1, 0, 0, 1.25, 4, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 2, 0, 1, 5]
    k_mer_stats = KMerStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    k_mer_stats.set_k_mer_features(inferred_msa.sequences)
    assert k_mer_stats.get_my_features_as_list() == ['inferred', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # TODO: handle this?

    inferred_msa.build_nj_tree()
    true_msa.build_nj_tree()
    tree_stats = TreeStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    tree_stats.set_tree_stats(inferred_msa.tree.get_branches_lengths_list(), inferred_msa.tree, inferred_msa.sequences, inferred_msa.seq_names)
    assert tree_stats.get_my_features_as_list() == ['inferred', 1.676, 0.183, 0.093, 0.396, 0.03, 0.304, -1.523, 0.174, 0.491, 0.026, 12, 0.872]
    dist_labels_stats.set_rf_from_true(inferred_msa.tree, true_msa.tree)
    data_to_print, col_names = dist_labels_stats.get_print_rf()
    assert data_to_print == ['inferred', 0]
    assert col_names == ['code','rf_from_true']

    sop_stats = SopStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    sop_stats.set_my_sop_score_parts(sp, inferred_msa.sequences)
    assert sop_stats.get_my_features_as_list() == ['inferred', -12.0, -0.12, 2.51, -2.5, -0.13, 251.0, 0.47, 0.22, 26, 25, 22]

    w_sop_stats = WSopStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    w_sop_stats.calc_seq_weights(config.additional_weights, inferred_msa.sequences, inferred_msa.seq_names, inferred_msa.tree)
    w_sop_stats.calc_w_sp(inferred_msa.sequences, sp)
    assert w_sop_stats.get_my_features_as_list() == ['inferred', -1.135, -0.586, -7.131, -6.979]


def build_e_tree_from_ours(tree: UnrootedTree) -> Tree:
    new_tree_root = Tree()
    add_child_to_tree(new_tree_root, tree.anchor)
    return new_tree_root


def add_child_to_tree(father: TreeNode, our_node: Node):
    if len(our_node.children) == 0:
        name = list(our_node.keys)[0]
        father.add_child(name=name)
    else:
        child = father.add_child(name=str(our_node.id))
        for our_child in our_node.children:
            add_child_to_tree(child, our_child)


def map_branches_of_tree(tree: Tree, tree_keys: set[str], differentiator_key: str) -> list[str]:
    branches: list[str] = []
    for n in tree.traverse("preorder"):
        if len(n.children) > 0:
            keys: set[str] = set()
            leaves: list[TreeNode] = n.get_leaves()
            for leaf in leaves:
                keys.add(leaf.name)
            other_side = tree_keys.difference(keys)
            if len(keys) > len(other_side) or (len(keys) == len(other_side) and differentiator_key in other_side):
                keys = other_side
            if len(keys) > 0:
                keys_list = list(keys)
                keys_list.sort()
                branches.append(','.join(keys_list))
    return branches


def map_branches_of_our_tree(tree: UnrootedTree) -> list[str]:
    branches: list[str] = []
    branches_set = tree.get_internal_edges_set()
    for b in list(branches_set):
        keys_list = list(b)
        keys_list.sort()
        branches.append(','.join(keys_list))
    return branches


def print_branches_ordered_list(branches: list[str]):
    branches.sort(key=lambda x: x[0])
    branches.sort(key=lambda x: len(x), reverse=True)
    print('branches:')
    for b in branches:
        print(b)


def create_msa_from_seqs_and_names(data_name: str, seqs: list[str], names: list[str]) -> MSA:
    msa = MSA(data_name)
    for i in range(len(seqs)):
        msa.add_sequence_to_me(seqs[i], names[i])
    return msa


def test_henikoff_w():
    aln: list[str] = [
        'AT-CGC',
        'ACATG-',
        'AT-CG-',
        'ATC-GA',
        'TTATGC'
    ]
    msa = MSA('test')
    msa.sequences = aln
    w_sop = WSopStats(msa.dataset_name, msa.get_taxa_num(), msa.get_msa_len())
    seq_weights_with_gap, seq_weights_no_gap = w_sop.compute_seq_w_henikoff_vars(msa.sequences)
    res = {'seq_weights_with_gap': seq_weights_with_gap, 'seq_weights_no_gap': seq_weights_no_gap}
    assert res == {
        'seq_weights_no_gap': [
             0.15454545454545454,
             0.22272727272727275,
             0.10909090909090909,
             0.24545454545454548,
             0.2681818181818182,
        ],
        'seq_weights_with_gap': [
             0.15714285714285717,
             0.21071428571428572,
             0.15714285714285717,
             0.2642857142857143,
             0.21071428571428572,
        ],
    }

def test_mid_point_rooting():
    aln: list[str] = [
        'AT-CGC-GGT',
        'ACATG-T-GA',
        'AT-CG--GGT',
        'ATC-GA-GGA',
        'TTATGCTGGA'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                          SopCalcTypes.EFFICIENT, 'comparison_files')
    inferred_msa: MSA = create_msa_from_seqs_and_names('inferred', aln, names)

    sp: SPScore = SPScore(config.models[0])
    sop_stats = SopStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
    sop_stats.set_my_sop_score_parts(sp, inferred_msa.sequences)
    inferred_msa.build_nj_tree()
    path, max_dist = inferred_msa.tree.longest_path()
    tree = RootedTree.root_tree(inferred_msa.tree, RootingMethods.LONGEST_PATH_MID)
    res = {'lp_length': max_dist, 'tree_a_length': tree.root.children[0].branch_length, 'tree_a_keys': tree.root.children[0].keys,
           'bl_a': round(tree.all_nodes[0].branch_length, 3), 'bl_b': round(tree.all_nodes[1].branch_length, 3),
           'bl_c': round(tree.all_nodes[2].branch_length, 3), 'bl_d': round(tree.all_nodes[3].branch_length, 3),
           'bl_e': round(tree.all_nodes[4].branch_length, 3), 'bl_a_c': round(tree.all_nodes[5].branch_length, 3),
           'bl_a_c_d': round(tree.all_nodes[6].branch_length, 3), 'bl_b_e': round(tree.all_nodes[7].branch_length, 3),
    }
    assert res == {'lp_length': 1.3641359567812426, 'tree_a_length': 0.22007006899467785, 'tree_a_keys': {'b', 'e'},
                   'bl_a': 0.082, 'bl_a_c': 0.329, 'bl_a_c_d': 0.271, 'bl_b': 0.462, 'bl_b_e': 0.22,
                   'bl_c': 0.026, 'bl_d': 0.183, 'bl_e': 0.104,}


def test_mid_point_rooting_case_b():
    unrooted = create_unrooted_tree_for_test()
    path, max_dist = unrooted.longest_path()
    tree = RootedTree.root_tree(unrooted, RootingMethods.LONGEST_PATH_MID)
    res = {'lp_length': max_dist, 'tree_a_length': round(tree.root.children[0].branch_length, 1), 'tree_a_keys': sorted(list(tree.root.children[0].keys)),
           'bl_a': round(tree.all_nodes[0].branch_length, 1), 'bl_b': tree.all_nodes[1].branch_length, 'bl_c': tree.all_nodes[2].branch_length,
           'bl_d': tree.all_nodes[3].branch_length, 'bl_e': tree.all_nodes[4].branch_length, 'bl_a_c': tree.all_nodes[5].branch_length,
           'bl_b_e': tree.all_nodes[7].branch_length, 'bl_b_e_d': round(tree.all_nodes[6].branch_length, 1)}
    tree.calc_clustal_w()
    res['a_w'] = tree.all_nodes[0].weight
    res['c_w'] = tree.all_nodes[2].weight
    res['e_w'] = round(tree.all_nodes[4].weight, 3)
    assert res == {'bl_b_e_d': 0.2, 'bl_a': 0.2, 'bl_a_c': 0.4, 'bl_b': 0.1, 'bl_b_e': 0.3, 'bl_c': 0.15, 'bl_d': 0.25,
                    'bl_e': 0.05, 'lp_length': 1.2, 'tree_a_keys': ['b', 'd', 'e'], 'tree_a_length': 0.2,
                    'a_w': 0.4, 'c_w': 0.35, 'e_w': 0.267}


def test_differential_sum_rooting(): # TODO: check this
    unrooted = create_unrooted_tree_for_test()
    path, max_dist = unrooted.longest_path()
    tree = RootedTree.root_tree(unrooted, RootingMethods.MIN_DIFFERENTIAL_SUM)
    res = {'lp_length': max_dist, 'tree_a_length': round(tree.root.children[0].branch_length, 3), 'tree_a_keys': sorted(list(tree.root.children[0].keys)),
           'bl_a': round(tree.all_nodes[0].branch_length, 3), 'bl_b': tree.all_nodes[1].branch_length, 'bl_c': tree.all_nodes[2].branch_length,
           'bl_d': tree.all_nodes[3].branch_length, 'bl_e': tree.all_nodes[4].branch_length, 'bl_a_c': round(tree.all_nodes[5].branch_length, 3),
           'bl_b_e': tree.all_nodes[7].branch_length, 'bl_b_e_d': round(tree.all_nodes[6].branch_length, 1)}
    tree.calc_clustal_w()
    res['a_w'] = round(tree.all_nodes[0].weight, 3)
    res['c_w'] = round(tree.all_nodes[2].weight, 3)
    res['e_w'] = round(tree.all_nodes[4].weight, 3)
    assert res == {'bl_b_e_d': 0.1, 'bl_a': 0.2, 'bl_a_c': 0.475, 'bl_b': 0.1, 'bl_b_e': 0.3, 'bl_c': 0.15, 'bl_d': 0.25,
                    'bl_e': 0.05, 'lp_length': 1.2, 'tree_a_keys': ['a', 'c'], 'tree_a_length': 0.475,
                    'a_w': 0.745, 'c_w': 0.345, 'e_w': 0.242}


def test_differential_sum_rooting_case_of_no_solution():
    unrooted = create_unrooted_tree_for_test()
    unrooted.all_nodes[0].set_branch_length(0.6)
    unrooted.all_nodes[1].set_branch_length(0.7)
    unrooted.all_nodes[2].set_branch_length(0.2)
    unrooted.all_nodes[3].set_branch_length(0.9)
    unrooted.all_nodes[4].set_branch_length(0.2)
    unrooted.all_nodes[5].set_branch_length(0.3)
    unrooted.all_nodes[6].set_branch_length(0.1)

    path, max_dist = unrooted.longest_path()
    tree = RootedTree.root_tree(unrooted, RootingMethods.MIN_DIFFERENTIAL_SUM)
    res = {'lp_length': round(max_dist, 2), 'tree_a_length': round(tree.root.children[0].branch_length, 3), 'tree_a_keys': sorted(list(tree.root.children[0].keys)),
           'bl_a': round(tree.all_nodes[0].branch_length, 3), 'bl_b': tree.all_nodes[1].branch_length, 'bl_c': tree.all_nodes[2].branch_length,
           'bl_d': tree.all_nodes[3].branch_length, 'bl_e': tree.all_nodes[4].branch_length, 'bl_a_c': round(tree.all_nodes[5].branch_length, 3),
           'bl_b_e': tree.all_nodes[7].branch_length, 'bl_b_e_d': round(tree.all_nodes[6].branch_length, 3)}
    tree.calc_clustal_w()
    res['a_w'] = round(tree.all_nodes[0].weight, 3)
    res['c_w'] = round(tree.all_nodes[2].weight, 3)
    res['e_w'] = round(tree.all_nodes[4].weight, 3)
    assert res == {'bl_b_e_d': 0.01, 'bl_a': 0.6, 'bl_a_c': 0.29, 'bl_b': 0.7, 'bl_b_e': 0.1, 'bl_c': 0.2, 'bl_d': 0.9,
                    'bl_e': 0.2, 'lp_length': 1.8, 'tree_a_keys': ['a', 'c'], 'tree_a_length': 0.29,
                    'a_w': 0.745, 'c_w': 0.345, 'e_w': 0.253}

def create_unrooted_tree_for_test() -> UnrootedTree:
    node_a = Node(node_id=0, keys={'a'}, children=[], children_bl_sum=0, branch_length=0.2)
    node_b = Node(node_id=1, keys={'b'}, children=[], children_bl_sum=0, branch_length=0.1)
    node_c = Node(node_id=2, keys={'c'}, children=[], children_bl_sum=0, branch_length=0.15)
    node_d = Node(node_id=3, keys={'d'}, children=[], children_bl_sum=0, branch_length=0.25)
    node_e = Node(node_id=4, keys={'e'}, children=[], children_bl_sum=0, branch_length=0.05)
    node_a_c = Node.create_from_children([node_a, node_c], 5)
    node_a_c.set_branch_length(0.6)
    node_a_c_d = Node.create_from_children([node_a_c, node_d], 6)
    node_a_c_d.set_branch_length(0.3)
    anchor = Node.create_from_children([node_a_c_d, node_b, node_e], 7)
    node_a.set_a_father(node_a_c)
    node_c.set_a_father(node_a_c)
    node_a_c.set_a_father(node_a_c_d)
    node_d.set_a_father(node_a_c_d)
    node_a_c_d.set_a_father(anchor)
    node_b.set_a_father(anchor)
    node_e.set_a_father(anchor)
    return UnrootedTree(anchor=anchor,
                            all_nodes=[node_a, node_b, node_c, node_d, node_e, node_a_c, node_a_c_d, anchor])


def test_single_msas():
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                                 SopCalcTypes.EFFICIENT, 'tests/comparison_files',
                                                 {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG,
                                                  WeightMethods.CLUSTAL_MID_ROOT,
                                                  WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})

    all_msa_ws = calc_single_msas(config)
    assert all_msa_ws == [
        ['bali_phy_msa.8.fasta', 401.38, 402.452, 1607.46, 1396.584],
        ['MUSCLE_diversified_replicate.none.216.afa', 401.042, 399.857, 2136.42, 1578.571],
        ['PRANK_b1#0003_hhT_tree_3_OP_0.38975399874169636_Split_3.fasta', 400.989, 402.289, 2314.664, 1444.722],
        ['PRANK_b1#0024_hhT_tree_21_OP_0.2963379796789501_Split_24.fasta', 403.128, 405.134, 2286.363, 1451.505]]

def test_global_alignment_blosum_affine_gap_case_a():
    config: Configuration = Configuration([EvoModel(-4, -0.5, 'Blosum62')],
                                          SopCalcTypes.EFFICIENT, 'tests/comparison_files',
                                          {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG,
                                           WeightMethods.CLUSTAL_MID_ROOT,
                                           WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})
    ga = GlobalAlign('PAWHEAE', 'HEAGAWGHEE', config.models[0])
    ga.print_matrix()
    ga.get_score()
    res_seq = list(map(lambda x: {'seq_a': x.profile_a, 'seq_b': x.profile_b}, ga.aligned_sequences))
    assert res_seq == [
        {'seq_a': ['-', '-', '-', 'P', 'A', 'W', 'H', 'E', 'A', 'E'],
         'seq_b': ['H', 'E', 'A', 'G', 'A', 'W', 'G', 'H', 'E', 'E']},
        {'seq_a': ['-', '-', '-', 'P', 'A', 'W', '-', 'H', 'E', 'A', 'E'],
         'seq_b': ['H', 'E', 'A', 'G', 'A', 'W', 'G', 'H', 'E', '-', 'E']},
        {'seq_a': ['P', '-', '-', '-', 'A', 'W', 'H', 'E', 'A', 'E'],
         'seq_b': ['H', 'E', 'A', 'G', 'A', 'W', 'G', 'H', 'E', 'E']},
        {'seq_a': ['P', '-', '-', '-', 'A', 'W', '-', 'H', 'E', 'A', 'E'],
         'seq_b': ['H', 'E', 'A', 'G', 'A', 'W', 'G', 'H', 'E', '-', 'E']}
    ]

def test_create_alternative_msas_by_realign():
    aln: list[str] = [
        'AT-CGC-GG-TT',
        'ACATG-T-GAAT',
        'AT-CG--GGATT',
        'ATC-GA-GG-AT',
        'TTATGCTGG-A-'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    true_aln: list[str] = [
        'AT-CGC-GG-TT',
        'ACATG-TG-AAT',
        'AT-CG--GGATT',
        'AT-CGA-GG-AT',
        'TTATGCTGG-A-'
    ]
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                          SopCalcTypes.EFFICIENT, 'comparison_files',
                                          {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG,
                                           WeightMethods.CLUSTAL_MID_ROOT,
                                           WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})
    true_msa: MSA = create_msa_from_seqs_and_names('true', true_aln, names)
    inferred_msa: MSA = create_msa_from_seqs_and_names('inferred', aln, names)
    res = inferred_msa.create_alternative_msas_by_realign(config)
    msa_list: list[MSA] = []
    msa_list_stats: list[DistanceLabelsStats] = []
    for msa_data in res:
        alt_msa:MSA = create_msa_from_seqs_and_names('alt', msa_data, names)
        alt_dops_stats = DistanceLabelsStats(alt_msa.dataset_name, len(alt_msa.sequences), len(alt_msa.sequences[0]))
        alt_dops_stats.set_my_dpos_dist_from_true(alt_msa.sequences, true_msa.sequences)
        msa_list.append(alt_msa)
        msa_list_stats.append(alt_dops_stats)
    dpos_stats = DistanceLabelsStats(inferred_msa.dataset_name, len(inferred_msa.sequences), len(inferred_msa.sequences[0]))
    dpos_stats.set_my_dpos_dist_from_true(inferred_msa.sequences, true_msa.sequences)
    dpos_ratio = abs(dpos_stats.dpos_from_true - msa_list_stats[0].dpos_from_true) / dpos_stats.dpos_from_true
    assert dpos_ratio <= 3


def test_create_alternative_msas_by_moving_smallest():
    aln: list[str] = [
        'AT-CGC-GG-TT',
        'ACATG-T-GAAT',
        'AT-CG--GGATT',
        'ATC-GA-GG-AT',
        'TTATGCTGG-A-'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    true_aln: list[str] = [
        'AT-CGC-GG-TT',
        'ACATG-TG-AAT',
        'AT-CG--GGATT',
        'AT-CGA-GG-AT',
        'TTATGCTGG-A-'
    ]
    true_msa: MSA = create_msa_from_seqs_and_names('true', true_aln, names)
    inferred_msa: MSA = create_msa_from_seqs_and_names('inferred', aln, names)
    res = inferred_msa.create_alternative_msas_by_moving_one_part()
    msa_list: list[MSA] = []
    dpos_list: list[float] = []
    for inx, msa_data in enumerate(res):
        print(inx)
        alt_msa:MSA = create_msa_from_seqs_and_names('alt', msa_data, names)
        alt_msa_dpos_stats = DistanceLabelsStats(alt_msa.dataset_name, len(alt_msa.sequences), len(alt_msa.sequences[0]))
        alt_msa_dpos_stats.set_my_dpos_dist_from_true(alt_msa.sequences, true_msa.sequences)
        dpos_list.append(alt_msa_dpos_stats.dpos_from_true)
        msa_list.append(alt_msa)
    inferred_msa_dpos_stats = DistanceLabelsStats(inferred_msa.dataset_name, len(inferred_msa.sequences), len(inferred_msa.sequences[0]))
    inferred_msa_dpos_stats.set_my_dpos_dist_from_true(inferred_msa.sequences, true_msa.sequences)
    dpos_ratio = abs(inferred_msa_dpos_stats.dpos_from_true - dpos_list[0]) / inferred_msa_dpos_stats.dpos_from_true
    assert dpos_ratio <= 1


def calc_single_msas(config: Configuration):
    all_msa_ws: list[list[float]] = []
    sp: SPScore = SPScore(config.models[0])
    project_path: Path = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute()
    dir_path: Path = Path(str(project_path) + '/msa_to_test')
    file_names = os.listdir(dir_path)
    for inferred_file_name in file_names:
        msa_name = inferred_file_name
        print(msa_name)
        inferred_msa = MSA(msa_name)
        inferred_msa.read_me_from_fasta(Path(os.path.join(str(dir_path), inferred_file_name)))
        inferred_msa.build_nj_tree()
        w_sop_stats = WSopStats(inferred_msa.dataset_name, inferred_msa.get_taxa_num(), inferred_msa.get_msa_len())
        w_sop_stats.calc_seq_weights(config.additional_weights, inferred_msa.sequences, inferred_msa.seq_names, inferred_msa.tree)
        w_sop_stats.calc_w_sp(inferred_msa.sequences, sp)
        all_msa_ws.append(w_sop_stats.get_my_features_as_list())
    return all_msa_ws

def test_henikoff_with_gaps_value():
    # Create a simple test alignment with known gap patterns
    aln: list[str] = [
        'AT-CGC',
        'ACATG-',
        'AT-CG-',
        'ATC-GA',
        'TTATGC'
    ]
    names: list[str] = ['a', 'b', 'c', 'd', 'e']
    
    # Create MSA and configuration
    msa = MSA('test')
    msa.sequences = aln
    msa.seq_names = names
    
    config: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62')],
                                        SopCalcTypes.EFFICIENT, 'comparison_files',
                                        {WeightMethods.HENIKOFF_WG})
    
    # Calculate sequence weights
    msa.build_nj_tree()
    w_sop_stats = WSopStats(msa.dataset_name, msa.get_taxa_num(), msa.get_msa_len())
    w_sop_stats.calc_seq_weights(config.additional_weights, msa.sequences, msa.seq_names, msa.tree)
    
    # Compute SP score with weights
    sp = SPScore(config.models[0])
    sop_w_options = w_sop_stats.calc_w_sp(msa.sequences, sp)
    
    # Set the weights in stats
    henikoff_with_gaps = w_sop_stats.henikoff_with_gaps
    
    # The expected value is the actual calculated SP score with Henikoff weights
    expected_value = -1.682110969387752
    
    # Check if the henikoff_with_gaps value is set correctly
    assert abs(henikoff_with_gaps - expected_value) < 1e-10, \
        f"Expected henikoff_with_gaps to be {expected_value}, but got {henikoff_with_gaps}"

    def test_comp_3():
        res = msa_comp_main()
        assert res is None