import sys
import os

from classes.config import Configuration
from classes.evo_model import EvoModel
from enums import SopCalcTypes, WeightMethods, StatsOutput
from multi_msa_service import multiple_msa_calc_features_and_labels

code = sys.argv[1]
print("Code:", code)
folder = f'/Users/kpolonsky/Documents/sp_alternative/comparison_files/{code}/'
# folder = f'/groups/pupko/kseniap/BaliBase4/RV11-50/ALL_MSAs_BaliBase11-50/{code}/'
#folder = f'/groups/pupko/kseniap/BaliBase4/ALL_MSAs_BaliBase/{code}/'
#folder = f'/groups/pupko/kseniap/ENSEMBLsim_dataset/ALL_MSAs_Ensemble/{code}'
#folder = f'/groups/pupko/kseniap/BaliBase4/RV10/ALL_MSAs_BaliBase/{code}'
#folder = f'/groups/pupko/kseniap/HOMSTRAD/Homstrad/ALL_MSAs/{code}'
#folder = f'/groups/pupko/kseniap/OXBENCH/ALL_MSAs_fixed/{code}'
# folder = f'/groups/pupko/kseniap/OrthoMaM/OrthoMaM_final_MSAs/{code}/'
configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'BLOSUM62'), EvoModel(-6, -0.5, 'BLOSUM62'),
                                              EvoModel(-10, -1, 'BLOSUM62'), EvoModel(-6, -1, 'BLOSUM62'),
                                              EvoModel(-10, -0.2, 'BLOSUM62'), EvoModel(-6, -0.2, 'BLOSUM62'),
                                              EvoModel(-10, -0.5, 'PAM250'), EvoModel(-6, -0.5, 'PAM250'),
                                              EvoModel(-10, -1, 'PAM250'), EvoModel(-6, -1, 'PAM250'),
                                              EvoModel(-10, -0.2, 'PAM250'), EvoModel(-6, -0.2, 'PAM250')],
                                             SopCalcTypes.EFFICIENT, folder,

                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              WeightMethods.CLUSTAL_DIFFERENTIAL_SUM},
                                             [5, 10, 20], {StatsOutput.ALL})

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
    #     multiple_msa_calc_features_and_labels(configuration)


    multiple_msa_calc_features_and_labels(configuration)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

