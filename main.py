from classes.config import Configuration
from enums import SopCalcTypes
from multi_msa_service import calc_multiple_msa_sp_scores

configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
                                             SopCalcTypes.EFFICIENT, 'comparison_files')

if __name__ == '__main__':
    calc_multiple_msa_sp_scores(configuration)
