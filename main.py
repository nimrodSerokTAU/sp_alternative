from classes.config import Configuration
from enums import SopCalcTypes, WeightMethods
from multi_msa_service import calc_multiple_msa_sp_scores

configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
                                             SopCalcTypes.EFFICIENT, 'comparison_files', False, False,
                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              })

if __name__ == '__main__':
    calc_multiple_msa_sp_scores(configuration)
