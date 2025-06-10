from classes.config import Configuration
from classes.evo_model import EvoModel
from enums import SopCalcTypes, WeightMethods, StatsOutput
from multi_msa_service import multiple_msa_calc_features_and_labels

configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'Blosum62'), EvoModel(-10, -0.5, 'Blosum50')],
                                             SopCalcTypes.EFFICIENT, 'comparison_files',
                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              WeightMethods.CLUSTAL_DIFFERENTIAL_SUM},
                                             [5, 10, 20], {StatsOutput.ALL})

if __name__ == '__main__':
    multiple_msa_calc_features_and_labels(configuration)

# TODO: consider alternatives creation.