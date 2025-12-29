import time

from classes.config import Configuration
from classes.evo_model import EvoModel
from enums import SopCalcTypes, WeightMethods, StatsOutput
from multi_msa_service import multiple_msa_calc_features_and_labels

configuration: Configuration = Configuration([EvoModel(-10, -0.5, 'BLOSUM62'), EvoModel(-10, -0.5, 'BLOSUM50')],
                                             SopCalcTypes.EFFICIENT, '../../20251113/10686/one',
                                             {WeightMethods.HENIKOFF_WG, WeightMethods.HENIKOFF_WOG, WeightMethods.CLUSTAL_MID_ROOT,
                                              WeightMethods.CLUSTAL_DIFFERENTIAL_SUM},
                                             [5, 10, 20], {StatsOutput.ALL})

if __name__ == '__main__':
    start_time = time.time()
    multiple_msa_calc_features_and_labels(configuration)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

# TODO: consider alternatives creation.
