from classes.msa_basic_stats import BasicStats
from classes.unrooted_tree import UnrootedTree
from dpos import compute_distance
from enums import DistanceType


class DistanceLabelsStats(BasicStats):

    ssp_from_true: float
    dseq_from_true: float
    dpos_from_true: float
    rf_from_true: int

    def __init__(self, code: str, taxa_num: int, msa_len: int):
        super().__init__(code, taxa_num, msa_len,
              [
                  'code', 'ssp_from_true', 'dseq_from_true', 'dpos_from_true',
              ])
        self.ssp_from_true = -1
        self.dseq_from_true = -1
        self.dpos_from_true = -1
        self.rf_from_true = -1

    def set_my_dpos_dist_from_true(self, inferred_msa: list[str], true_msa: list[str]): # TODO: fix this
        self.dpos_from_true = compute_distance(true_msa, inferred_msa, DistanceType.D_POS)
        self.ssp_from_true = compute_distance(true_msa, inferred_msa, DistanceType.D_SSP)

    def set_rf_from_true(self, my_tree: UnrootedTree, true_tree: UnrootedTree):
        self.rf_from_true = my_tree.calc_rf(true_tree)

    def get_print_rf(self) -> tuple[list, list]:
        return [self.code, self.rf_from_true], ['code','rf_from_true']
