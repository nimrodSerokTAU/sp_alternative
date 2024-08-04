from classes.gap_interval import GapInterval
from classes.sp_score import SPScore


def test_sp_perfect():
    sp: SPScore = SPScore(-1, -1, -1)
    profile: list[str] = [
        'ARNDCQEGHI',
        'ARNDCQEGHI',
        'ARNDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # (5 + 7 + 7 + 8 + 13 + 7 + 6 + 8 + 10 + 5) * 3 = 76 * 3
    assert res == 228


def test_sp_no_gaps():
    sp: SPScore = SPScore(-1, -5, 0)
    profile: list[str] = [
        'ARNDCQEGHI',
        'AANDCQEGAI',
        'AANDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # RRR -> RAA : 21 -> 5  -4 = 1 -> -20
    # HHH -> HHA : 30 -> 10 -4 = 6 -> -24
    assert res == 184


def test_sp_local_gaps():
    sp: SPScore = SPScore(-1, -5, 0)
    profile: list[str] = [
        'ARNDCQ-GHI',
        'AANDCQ-GAI',
        'AANDCQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    # EEE -> --E : 18 -> -6 * 2 = -12 -> -30
    assert res == 154


def test_naive_algo_case_a_subs_only():
    sp: SPScore = SPScore(0, 0, 0)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 91  # subs only: 91


def test_naive_algo_case_a_subs_and_ge():
    sp: SPScore = SPScore(0, -5, 0)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 41  # ge cost only: -50


def test_naive_algo_case_a_subs_and_ge_and_gs():
    sp: SPScore = SPScore(-1, -5, -1)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    res: int = sp.compute_naive_sp_score(profile)
    assert res == 35  # gs cost only: -6


def test_compute_sp_s_and_sp_ge():  # our function
    sp: SPScore = SPScore(0, -5, 0)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 15 + 1 + (-6 -6) + (8 -5 -6) + 39 + (7 -6 -6) + (-6 -5) + (-5, -5) + 6 + 15
    sp_score_subs: int
    sp_score_gap_e: int
    sp_score_subs, sp_score_gap_e = sp.compute_sp_s_and_sp_ge(profile)
    res = {'sp_score_subs': sp_score_subs, 'sp_score_gap_e': sp_score_gap_e}
    assert res == {'sp_score_subs': 91, 'sp_score_gap_e': -40}  # this is correct without gs cost


def test_onl_gap_open_and_ext_cost_same():  # this is not correct...TODO: debug this
    sp: SPScore = SPScore(-1, -5, -1)
    profile: list[str] = [
        'ARNDC---HI',
        'AA-DCQ--AI',
        'AA--CQEGHI']
    # 5 gap openings: 2, 1, 2, 1 ? why not 6 ?
    res: int = sp.compute_sp_gap_open(profile)
    assert res == -6


# def restricted():  # ?
#     sp: SPScore = SPScore(-1, -5, 0)
#     profile: list[str] = [
#         'ARNDC---HI',
#         'AA-DCQ--AI',
#         'AA--CQEGHI']
#     intervals_list: list[list[GapInterval]] = []
#     res: int = 0
#     for seq in profile:
#         intervals_list.append(sp.compute_gap_intervals(list(seq)))
#     for i in range(len(intervals_list)):
#         for j in range(i + 1, len(intervals_list)):
#             res += compute_pairwise_restricted_gap_intervals(iter(intervals_list[i]), iter(intervals_list[j]),
#                                                              sp.gs_cost, sp.ge_cost)
#     assert res == -10



