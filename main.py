import sys
import os

from classes.config import Configuration
from enums import SopCalcTypes, WeightMethods
from multi_msa_service import calc_multiple_msa_sp_scores

# configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
#                                              SopCalcTypes.EFFICIENT, '/Users/kpolonsky/Downloads/TEST/MSAs_ALL_200K', True, False)

configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
                                             SopCalcTypes.EFFICIENT, '/Users/kpolonsky/Downloads/OrthoMaM_final_MSAs/ADNP',  True, False,
                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})

# configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
#                                              SopCalcTypes.EFFICIENT, '/Users/kpolonsky/Downloads/OrthoMaM_final_MSAs/',  True, False,
#                                              )


# code = sys.argv[1]
# folder = f'/groups/pupko/kseniap/Bali-Phy_OrthoMaM_MSAs/{code}/'
# # folder = f'/groups/pupko/kseniap/BaliBase4/ALL_MSAs_BaliBase/{code}/'
# configuration: Configuration = Configuration(-10, -0.5, 0, 'Blosum62',
#                                              SopCalcTypes.EFFICIENT, folder, True, False,
#                                              {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
#                                               WeightMethods.CLUSTAL_DIFFERENTIAL_SUM})


if __name__ == '__main__':
    # if not os.path.exists(f'/groups/pupko/kseniap/sp_alternative2/output/comparison_results_{code}.csv'):
    #     calc_multiple_msa_sp_scores(configuration)
    calc_multiple_msa_sp_scores(configuration)
