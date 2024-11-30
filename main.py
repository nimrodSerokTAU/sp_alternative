import sys
import os

from classes.config import Configuration
from enums import SopCalcTypes
from multi_msa_service import calc_multiple_msa_sp_scores

# configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
#                                              SopCalcTypes.EFFICIENT, '/Users/kpolonsky/Downloads/TEST/MSAs_ALL_200K', True, False)
# configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
#                                              SopCalcTypes.EFFICIENT, '/groups/pupko/kseniap/Simulated_Dataset_TreeBase/ALL_MSAs_TreeBase', True, False)
code = sys.argv[1]
folder = f'/groups/pupko/kseniap/Bali-Phy_OrthoMaM_MSAs/{code}/'
# folder = f'/groups/pupko/kseniap/BaliBase4/ALL_MSAs_BaliBase/{code}/'
configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
                                             SopCalcTypes.EFFICIENT, folder, True, False)


if __name__ == '__main__':
    if not os.path.exists(f'/groups/pupko/kseniap/sp_alternative/output/comparison_results_{code}.csv'):
        calc_multiple_msa_sp_scores(configuration)
