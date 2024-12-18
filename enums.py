from enum import Enum


class SopCalcTypes:
    NAIVE = 0
    EFFICIENT = 1

class RootingMethods(Enum):
    LONGEST_PATH_MID = 'longest_path_mid'
    MIN_DIFFERENTIAL_SUM = 'min_differential_sum'

class WeightMethods(Enum):
    HENIKOFF_WG = 'henikoff_with_gaps'
    HENIKOFF_WOG = 'henikoff_without_gaps'
    CLUSTAL_MID_ROOT = 'clustal_mid_root'