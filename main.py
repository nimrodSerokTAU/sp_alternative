import sys
import os

from classes.config import Configuration
from classes.evo_model import EvoModel
from enums import SopCalcTypes, WeightMethods, StatsOutput
from multi_msa_service import multiple_msa_calc_features_and_labels

configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62'), EvoModel(-10, -0.5, 'Blosum50')],
                                             SopCalcTypes.EFFICIENT, 'comparison_files',
                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              WeightMethods.CLUSTAL_DIFFERENTIAL_SUM},
                                             {StatsOutput.ALL})

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
    multiple_msa_calc_features_and_labels(configuration)

# TODO: consider alternatives creation.