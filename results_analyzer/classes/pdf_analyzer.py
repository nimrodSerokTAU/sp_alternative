from scipy.stats import gaussian_kde
import pandas as pd
import seaborn as sns

from results_analyzer.classes.pdf_plot import PDFPlot
from results_analyzer.constants import MARKERS, COLORS
from results_analyzer.classes.measure import Measure


class PDFAnalyzer:
    measures_dict: dict[str, Measure]
    dataset_name: str
    pdf_plot: PDFPlot

    def __init__(self, data_dir: str, features_file_name: str, prediction_file_model1: str,
                 prediction_file_model2: str, measures: list[Measure], dataset_name: str):

        self.measures = dict()
        for measure in measures:
            self.measures[measure.key] = measure
        self.dataset_name = dataset_name
        self.data = pd.DataFrame()

        self.read_data(f'{data_dir}/{features_file_name}',
                       f'{data_dir}/{prediction_file_model1}',
                       f'{data_dir}/{prediction_file_model2}')
        self.analyze_results_per_code()

    def read_data(self, features_file: str, prediction_file_model1: str, prediction_file_model2: str) -> None:
        df1 = pd.read_csv(features_file)
        df2 = pd.read_csv(prediction_file_model1)
        df3 = pd.read_csv(prediction_file_model2)

        for df in [df1, df2, df3]:
            df['code1'] = df['code1'].astype(str)

        merged_df = pd.merge(df1, df2, on=['code', 'code1'], how='inner')
        merged_df = pd.merge(merged_df, df3, on=['code', 'code1'], how='inner')
        # filtering original TRUE MSA if present
        df = merged_df[~merged_df['code'].str.contains('test_original', na=False)]
        print(df.shape)

        self.data = df

    def filter_code_data(self, code: str) -> pd.DataFrame:
        code_df = self.data[self.data['code1'] == code]
        exclude_list = ['test_original']
        mask = code_df['code'].str.contains('|'.join(exclude_list), case=False, na=False)
        code_df = code_df[~mask]

        return code_df

    def get_scores(self, df: pd.DataFrame, code: str) -> dict:
        return {
            'model1': df.loc[df[self.measures['model1'].external_name].idxmin()][self.measures['true_min'].external_name],
            'model2': df.loc[df[self.measures['model2'].external_name].idxmin()][self.measures['true_min'].external_name],
            'sop': df.loc[df[self.measures['sop'].external_name].idxmax()][self.measures['true_min'].external_name],
            'true_min': df[self.measures['true_min'].external_name].min(),
            'mafft': df.loc[df['code'] == self.measures['mafft'].external_name][self.measures['true_min'].external_name].values[0],
            'prank': df.loc[df['code'] == self.measures['prank'].external_name][self.measures['true_min'].external_name].values[0],
            'baliphy': df.loc[df['code'] == f'{self.measures["baliphy"].external_name}.{code}.fas'][self.measures['true_min'].external_name].values[0],
            'muscle': df.loc[df['code'] == f'{self.measures["muscle"].external_name}.{code}.fas'][self.measures['true_min'].external_name].values[0]
        }

    def plot_kde_with_annotations(self, scores: dict, true_scores: pd.Series, code: str) -> None:
        kde = gaussian_kde(true_scores)
        score_densities = {k: kde([v])[0] for k, v in scores.items()}

        sns.kdeplot(true_scores, color='black', label=f'True Scores ({code})', linewidth=1.5, clip=(0, None))
        number_of_seq = len(true_scores)


        x: list[float] = []
        y: list[float] = []
        markers: list[str] = []
        colors: list[str] = []
        labels: list[str] = []
        for measure_key in self.measures.keys():
            x.append(scores[measure_key])
            y.append(score_densities[measure_key])
            markers.append(MARKERS[measure_key])
            colors.append(COLORS[measure_key])
            labels.append(self.measures[measure_key].presentation_name)

        self.pdf_plot = PDFPlot(true_scores, x, y, markers, colors, labels, 100, '$d_{seq}$ from "true" MSA', 'Density')


    def plot_single_code_results(self, code: str) -> bool:
        try:
            code_df = self.filter_code_data(code)
            true_scores = code_df[self.measures['true_min'].external_name].values
            scores = self.get_scores(code_df, code)
            self.plot_kde_with_annotations(scores, true_scores, code)
            return True
        except Exception as e:
            print(f"Failed to process code '{code}': {e}\n")
            return False


    def analyze_results_per_code(self) -> None:
        unique_codes = self.data['code1'].unique()
        for code in unique_codes:
            if code != self.dataset_name:
                continue
            if not self.plot_single_code_results(code):
                continue
