from results_analyzer.classes.alternative_labels import AlternativeLabelFile
from results_analyzer.classes.plot_layout import PlotLayout
from results_analyzer.classes.stacked_col_graphs import StackedColGraph
from results_analyzer.classes.correlation_analyzer import CorrelationAnalyzer
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.pdf_analyzer import PDFAnalyzer

empirical_example: str = 'BBA0058'
sim_example: str = '126014'
data_dir = '../data_analyzer/corr'  # use your own path
output_path = f'{data_dir}/output/layout'

# fig 1 a, c:
fig1Emp = CorrelationAnalyzer(data_dir,
                              ['input/features_w_aligner_model_1_empr_20251116.csv'],
                    'dseq_from_true',
                              [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=-1),
                     Measure(key='entropy_sum', external_name='entropy_sum', presentation_name='Entropy Sum', correlation_direction=1)],
                              empirical_example, False, 'Correlation with Distance from True',
                              0, 0.75, None, True,
                              'test_original.fasta')

# # # fig 1 b, d:
fig1Sim = CorrelationAnalyzer(data_dir,
                              ['input/features_w_aligner_model_1_sim_20251116.csv'],
                    'dseq_from_true',
                              [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=-1),
                     Measure(key='entropy_sum', external_name='entropy_sum', presentation_name='Entropy Sum', correlation_direction=1)],
                              sim_example, False,  'Correlation with Distance from True',
                              0, 0.75, None, True, '')
#
pl1 = PlotLayout(14, 12, 2, 2, output_path, 'Fig1')
pl1.double_col_plot([fig1Emp.get_example_scatter(), fig1Emp.get_r(), fig1Sim.get_example_scatter(), fig1Sim.get_r()], ['a', 'c', 'b', 'd'], {})

# fig 2 a, c, fig 3 b, c:
fig2Emp = CorrelationAnalyzer(data_dir,
                              ['input/features_w_predictions_model_1_empr_20251116.csv'],
                              'dseq_from_true',
                              [Measure(key='model1', external_name='predicted_score', presentation_name='Model1', correlation_direction=1),
                               Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=-1)],
                              empirical_example, True, 'Correlation with Distance from True', 401, 0.75) # BBA0150

# fig 2 b, d:
fig2Sim = CorrelationAnalyzer(data_dir,
                              ['input/features_w_predictions_model_1_sim_20251116.csv'],
                              'dseq_from_true',
                              [Measure(key='model1', external_name='predicted_score', presentation_name='Model1', correlation_direction=1),
                              Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=-1)],
                              sim_example, False, 'Correlation with Distance from True', 401, 0.75)

pl2 = PlotLayout(14, 12, 2, 2, output_path, 'Fig2')
pl2.double_col_plot([fig2Emp.get_example_scatter(), fig2Emp.get_r(), fig2Sim.get_example_scatter(), fig2Sim.get_r()],  ['a', 'c', 'b', 'd'],
                    {0: [{"center": (0.31,0.96), "width": 0.15, "height": 0.1, "angle": 35, "color": "#1fad1a"},
                     {"center": (0.3,0.04), "width": 0.11, "height": 0.08, "angle": 60, "color": "#1fad1a"}],
                     2: [{"center": (0.055,0.985), "width": 0.08, "height": 0.05, "angle": 35, "color": "#1fad1a"},
                     {"center": (0.035,0.05), "width": 0.04, "height": 0.12, "angle": 0, "color": "#1fad1a"}]})

# # # fig 3:
fig3a = PDFAnalyzer(data_dir=data_dir,
                    features_file_name='/input/features_w_aligner_model_1_empr_20251116.csv',
                    prediction_file_model1='/input/prediction_DL_0_mode1_dseq_from_true_20251116.csv',
                    prediction_file_model2='/input/prediction_DL_0_mode2_dseq_from_true_20251116.csv',
                    measures = [
                    Measure(key='true_min', external_name='dseq_from_true', presentation_name='Min Alt MSA', correlation_direction=0),
                    Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='Max SoP',correlation_direction=-0),
                    Measure(key='model1',  external_name='predicted_score_x', presentation_name='Model 1 Predicted',correlation_direction=0),
                    Measure(key='model2', external_name='predicted_score_y', presentation_name='Model 2 Predicted',correlation_direction=0),
                    Measure(key='mafft', external_name='MSA.MAFFT.aln.With_Names', presentation_name='MAFFT Default', correlation_direction=0),
                    Measure(key='prank', external_name='MSA.PRANK.aln.With_Names', presentation_name='PRANK Default', correlation_direction=0),
                    Measure(key='muscle', external_name='MSA.MUSCLE.aln.best', presentation_name='Muscle Default', correlation_direction=0),
                    Measure(key='baliphy', external_name='MSA.BALIPHY.aln.best', presentation_name='BAli-Phy Default', correlation_direction=0)
                ], dataset_name=empirical_example)

pl3 = PlotLayout(14, 7, 1, 1, output_path, 'Fig3')
pl3.plot_with_inset(empirical_example, fig3a.pdf_plot, fig2Emp.zoom_in)

# fig 4 a:
sc4a = StackedColGraph(data_dir,
                       [{'relative_file_path': 'input/empirical/pick_me_trio_overall_v0_20251116.csv', 'series_name': 'Empirical'},
                      {'relative_file_path': 'input/simulated/pick_me_trio_overall_v0_20251116.csv', 'series_name': 'Simulated'}],
                       [Measure(key='', external_name='', presentation_name=None, correlation_direction=0)],
                       [Measure(key='sop', external_name='SoP', presentation_name='SoP',correlation_direction=0),
                           Measure(key='model2', external_name='Predicted', presentation_name='Model2',correlation_direction=0),
                           Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and SoP)',correlation_direction=0)],
                       True, ['sop', 'model2'])


fig4b = StackedColGraph(data_dir,
                        [{'relative_file_path': 'input/empirical/pick_me_trio_v0_20251116.csv', 'series_name': None}],
                        [Measure(key='mafft', external_name='', presentation_name='MAFFT',correlation_direction=0),
                 Measure(key='prank', external_name='', presentation_name='PRANK',correlation_direction=0),
                 Measure(key='muscle', external_name='', presentation_name='Muscle',correlation_direction=0),
                 Measure(key='baliphy', external_name='', presentation_name='BAli-Phy',correlation_direction=0)],
                        [Measure(key='default', external_name='Default', presentation_name='Default',correlation_direction=0),
                           Measure(key='model2', external_name='Predicted', presentation_name='Model2',correlation_direction=0),
                           Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and Default)',correlation_direction=0)],
                        False, [])

#
# # fig 4 c:
fig4c = StackedColGraph(data_dir,
                        [{'relative_file_path': 'input/simulated/pick_me_trio_v0_20251116.csv', 'series_name': None}],
                        [Measure(key='mafft', external_name='', presentation_name='MAFFT',correlation_direction=0),
                                  Measure(key='prank', external_name='', presentation_name='PRANK',correlation_direction=0),
                                  Measure(key='muscle', external_name='', presentation_name='Muscle',correlation_direction=0),
                                  Measure(key='baliphy', external_name='', presentation_name='BAli-Phy',correlation_direction=0)],
                        [Measure(key='default', external_name='Default', presentation_name='Default',correlation_direction=0),
                           Measure(key='model2', external_name='Predicted', presentation_name='Model2',correlation_direction=0),
                           Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and Default)',correlation_direction=0)],
                        False, [])

pl4 = PlotLayout(16, 7, 1, 1, output_path, 'Fig4')
pl4.triple_plot(sc4a.subplot, [fig4b.subplot, fig4c.subplot], ['Empirical Data', 'Simulated Data'])


######## Supplementary
# fig S2 a:
fig_s2a_emp = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_aligner_model_1_empr_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=1)],
                    '', True, f'Correlation with RPS', 0, 1,
                                  AlternativeLabelFile('input/combined_scores_vs_true_empr_261125_dedup.csv', 1, 3, 0, 0, 1),
                                  False)
# fig S2 b:
fig_s2b_sim = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_aligner_model_1_sim_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=1)],
                    '', False, f'Correlation with RPS', 0, 1,
                                  AlternativeLabelFile('input/combined_scores_vs_true_sim_130925.csv', 1, 3, 0, 0, 1))
#
# fig S1 c:
fig_s2c_emp = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_aligner_model_1_empr_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=1)],
                    '', True, f'Correlation with TCS', 0, 1,
                                  AlternativeLabelFile('input/combined_scores_vs_true_empr_261125_dedup.csv', 2, 3, 0, 0, 1),
                                  False)
# fig S1 d:
fig_s2d_sim = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_aligner_model_1_sim_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP', correlation_direction=1)],
                    '', False, f'Correlation with TCS', 0, 1,
                                  AlternativeLabelFile('input/combined_scores_vs_true_sim_130925.csv', 2, 3, 0, 0, 1))

plS2 = PlotLayout(14, 12, 2, 2, output_path, 'FigS2')
plS2.double_col_plot([fig_s2a_emp.get_r(), fig_s2c_emp.get_r(), fig_s2b_sim.get_r(), fig_s2d_sim.get_r()], ['a', 'c', 'b', 'd'], {})

# fig S3:
fig_s3a_emp = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_predictions_model_1_empr_20251116.csv',
                                     'input/features_w_predictions_model_2_empr_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='model1', external_name='predicted_score', presentation_name='Model1', correlation_direction=1),
                              Measure(key='model2', external_name='predicted_score', presentation_name='Model2', correlation_direction=1)],
                                  empirical_example, False, 'Correlation with Distance from True', 0, 0.75)

fig_s3b_sim = CorrelationAnalyzer(data_dir,
                                  ['input/features_w_predictions_model_1_sim_20251116.csv',
                                     'input/features_w_predictions_model_2_sim_20251116.csv'],
                    'dseq_from_true',
                                  [Measure(key='model1', external_name='predicted_score', presentation_name='Model1', correlation_direction=1),
                              Measure(key='model2', external_name='predicted_score', presentation_name='Model2', correlation_direction=1)],
                                  sim_example, False,  'Correlation with Distance from True', 0, 0.75)

pl_s_3 = PlotLayout(14, 12, 2, 2, output_path, 'FigS3')
pl_s_3.double_col_plot([fig_s3a_emp.get_example_scatter(), fig_s3a_emp.get_r(), fig_s3b_sim.get_example_scatter(), fig_s3b_sim.get_r()], ['a', 'c', 'b', 'd'], {})


