from enum import Enum


class SopCalcTypes:
    NAIVE = 0
    EFFICIENT = 1

class RootingMethods:
    LONGEST_PATH_MID = 0
    MIN_DIFFERENTIAL_SUM = 1

class WeightMethods(Enum):
    HENIKOFF_WG = 'henikoff_with_gaps'
    HENIKOFF_WOG = 'henikoff_without_gaps'
    CLUSTAL_MID_ROOT = 'clustal_mid_root'