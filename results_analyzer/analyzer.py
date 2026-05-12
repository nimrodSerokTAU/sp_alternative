from results_analyzer.classes.alternative_labels import AlternativeLabelFile
from results_analyzer.classes.plot_layout import PlotLayout
from results_analyzer.classes.stacked_col_graphs import StackedColGraph
from results_analyzer.classes.correlation_analyzer import CorrelationAnalyzer
from results_analyzer.classes.measure import Measure
from results_analyzer.classes.pdf_analyzer import PDFAnalyzer

empirical_example: str = 'BBA0169'
sim_example: str = '23410'
data_dir = 'D:/PhDB/papers/first/figures_data'  # use your own path
output_path = f'{data_dir}/output'

# # # fig 1 a, c:
# fig1Emp = CorrelationAnalyzer(data_dir,
#                               ['input/BAliBASE/features_w_aligner_1_dseq_from_true.csv'],
#                               'dseq_from_true',
#                               [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
#                                        correlation_direction=-1),
#                                         Measure(key='entropy_sum', external_name='entropy_sum', presentation_name='Entropy Sum',
#                                        correlation_direction=1)],
#                               empirical_example, False, 'Correlation with Distance from True',
#                               0, 0.75, None, True,
#                               'test_original.fasta')
#
# # # # fig 1 b, d:
# fig1Sim = CorrelationAnalyzer(data_dir,
#                               ['input/OrthoMaM/orthomam_features_w_aligner_090426.csv'],
#                               'dseq_from_true',
#                               [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
#                                        correlation_direction=-1),
#                                         Measure(key='entropy_sum', external_name='entropy_sum', presentation_name='Entropy Sum',
#                                        correlation_direction=1)],
#                               sim_example, False, 'Correlation with Distance from True',
#                               0, 0.75, None, True, '')
# #
# pl1 = PlotLayout(14, 12, 2, 2, output_path, 'Fig1')
# pl1.double_col_plot([fig1Emp.get_example_scatter(), fig1Emp.get_r_scatter(), fig1Sim.get_example_scatter(), fig1Sim.get_r_scatter()],
#                     ['a', 'c', 'b', 'd'], {})

# fig 2 a, c, fig 3 b, c:
fig2_emp = CorrelationAnalyzer(data_dir,
                               ['input/BAliBASE/model_1/features_w_predictions.csv'],
                              'dseq_from_true',
                               [
                                    Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
                                            correlation_direction=-1),
                                    Measure(key='model1', external_name='predicted_score', presentation_name='Model1',
                                            correlation_direction=1)
                               ],
                               empirical_example, True, 'Correlation with Distance from True', 401, 0.75)  # BBA0150

# fig 2 b, d:
fig2_sim = CorrelationAnalyzer(data_dir,
                               ['input/OrthoMaM/model_1/features_w_predictions.csv'],
                              'dseq_from_true',
                               [
                                    Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
                                            correlation_direction=-1),
                                    Measure(key='model1', external_name='predicted_score', presentation_name='Model1',
                                            correlation_direction=1)
                               ],
                               sim_example, False, 'Correlation with Distance from True', 401, 0.5)

pl2 = PlotLayout(14, 12, 2, 2, output_path, 'Fig2')
pl2.double_col_plot([fig2_emp.get_example_scatter(), fig2_emp.get_r_scatter(), fig2_sim.get_example_scatter(), fig2_sim.get_r_scatter()],
                    ['a', 'c', 'b', 'd'],
                    {0: [{"center": (0.21, 0.97), "width": 0.13, "height": 0.1, "angle": 35, "color": "#1fad1a"},
                         {"center": (0.2, 0.04), "width": 0.11, "height": 0.08, "angle": 60, "color": "#1fad1a"}],
                     2: [{"center": (0.05, 0.99), "width": 0.08, "height": 0.09, "angle": 35, "color": "#1fad1a"},
                         {"center": (0.03, 0.06), "width": 0.04, "height": 0.18, "angle": 0, "color": "#1fad1a"}]})


# pl2b = PlotLayout(14, 12, 2, 2, output_path, 'Fig2b')
# pl2b.double_col_plot([fig2_emp.get_example_scatter(), fig2_emp.get_r_curve(), fig2_sim.get_example_scatter(), fig2_sim.get_r_curve()],
#                     ['a', 'c', 'b', 'd'],
#                      {0: [{"center": (0.21, 0.97), "width": 0.13, "height": 0.1, "angle": 35, "color": "#1fad1a"},
#                           {"center": (0.2, 0.04), "width": 0.11, "height": 0.08, "angle": 60, "color": "#1fad1a"}],
#                       2: [{"center": (0.05, 0.99), "width": 0.08, "height": 0.09, "angle": 35, "color": "#1fad1a"},
#                           {"center": (0.03, 0.06), "width": 0.04, "height": 0.18, "angle": 0, "color": "#1fad1a"}]})

# # # # fig 3:
# fig3a = PDFAnalyzer(data_dir=data_dir,
#                     features_file_name='input/BAliBASE/features_w_aligner_1_dseq_from_true.csv',
#                     prediction_file_model1='/input/BAliBASE/model_1/prediction_DL_0_mode1_dseq_from_true.csv',
#                     prediction_file_model2='/input/BAliBASE/model_2/prediction_DL_0_mode2_dseq_from_true.csv',
#                     measures=[
#                         Measure(key='true_min', external_name='dseq_from_true', presentation_name='Min Alt MSA',
#                                 correlation_direction=0),
#                         Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='Max SoP',
#                                 correlation_direction=-0),
#                         Measure(key='model1', external_name='predicted_score_x', presentation_name='Model 1 Predicted',
#                                 correlation_direction=0),
#                         Measure(key='model2', external_name='predicted_score_y', presentation_name='Model 2 Predicted',
#                                 correlation_direction=0),
#                         Measure(key='mafft', external_name='MSA.MAFFT.aln.With_Names',
#                                 presentation_name='MAFFT Default', correlation_direction=0),
#                         Measure(key='prank', external_name='MSA.PRANK.aln.With_Names',
#                                 presentation_name='PRANK Default', correlation_direction=0),
#                         Measure(key='muscle', external_name='MSA.MUSCLE.aln.best', presentation_name='Muscle Default',
#                                 correlation_direction=0),
#                         Measure(key='baliphy', external_name='MSA.BALIPHY.aln.best',
#                                 presentation_name='BAli-Phy Default', correlation_direction=0)
#                     ], dataset_name=empirical_example)
#
# pl3 = PlotLayout(14, 7, 1, 1, output_path, 'Fig3')
# pl3.plot_with_inset(empirical_example, fig3a.pdf_plot, fig2_emp.zoom_in)

# # fig 4 a:
# sc4a = StackedColGraph(data_dir,
#                        [{'relative_file_path': 'input/BAliBASE/model_2/pick_me_trio_overall_v0.csv',
#                          'series_name': 'Empirical'},
#                         {'relative_file_path': 'input/OrthoMaM/model_2/pick_me_trio_overall_v0.csv',
#                          'series_name': 'Simulated'}],
#                        [Measure(key='', external_name='', presentation_name=None, correlation_direction=0)],
#                        [Measure(key='sop', external_name='SoP', presentation_name='SoP', correlation_direction=0),
#                         Measure(key='model2', external_name='Predicted', presentation_name='Model2',
#                                 correlation_direction=0),
#                         Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and SoP)',
#                                 correlation_direction=0)],
#                        True, ['sop', 'model2'], False)
#
# fig4b = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/BAliBASE/model_2/pick_me_trio_v0.csv', 'series_name': None}],
#                         [Measure(key='mafft', external_name='', presentation_name='MAFFT', correlation_direction=0),
#                          Measure(key='prank', external_name='', presentation_name='PRANK', correlation_direction=0),
#                          Measure(key='muscle', external_name='', presentation_name='Muscle', correlation_direction=0),
#                          Measure(key='baliphy', external_name='', presentation_name='BAli-Phy',
#                                  correlation_direction=0)],
#                         [Measure(key='default', external_name='Default', presentation_name='Default',
#                                  correlation_direction=0),
#                          Measure(key='model2', external_name='Predicted', presentation_name='Model2',
#                                  correlation_direction=0),
#                          Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and Default)',
#                                  correlation_direction=0)],
#                         False, [], False)
#
# #
# # # fig 4 c:
# fig4c = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/model_2/pick_me_trio_v0.csv', 'series_name': None}],
#                         [Measure(key='mafft', external_name='', presentation_name='MAFFT', correlation_direction=0),
#                          Measure(key='prank', external_name='', presentation_name='PRANK', correlation_direction=0),
#                          Measure(key='muscle', external_name='', presentation_name='Muscle', correlation_direction=0),
#                          Measure(key='baliphy', external_name='', presentation_name='BAli-Phy',
#                                  correlation_direction=0)],
#                         [Measure(key='default', external_name='Default', presentation_name='Default',
#                                  correlation_direction=0),
#                          Measure(key='model2', external_name='Predicted', presentation_name='Model2',
#                                  correlation_direction=0),
#                          Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and Default)',
#                                  correlation_direction=0)],
#                         False, [], False)
#
# pl4 = PlotLayout(16, 7, 1, 1, output_path, 'Fig4')
# pl4.triple_plot(sc4a.subplot, [fig4b.subplot, fig4c.subplot], ['Empirical Data', 'Simulated Data'], [1, 2, 2])


# # ######## Supplementary




# # fig S2 a:
# fig_s2a_emp = CorrelationAnalyzer(data_dir,
#                                   ['input/BAliBASE/features_w_aligner_1_dseq_from_true.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5',
#                                            presentation_name='SoP', correlation_direction=1)],
#                                   '', True, f'Correlation with RPS', 0, 1,
#                                   AlternativeLabelFile('input/combined_scores_vs_true_empr_261125_dedup.csv', 1, 3, 0,
#                                                        0, 1),
#                                   False)
# # fig S2 b:
# fig_s2b_sim = CorrelationAnalyzer(data_dir,
#                                   ['input/OrthoMaM/orthomam_features_w_aligner_090426.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5',
#                                            presentation_name='SoP', correlation_direction=1)],
#                                   '', False, f'Correlation with RPS', 0, 1,
#                                   AlternativeLabelFile('input/combined_scores_vs_true_sim_130925.csv', 1, 3, 0, 0, 1))
# #
# # fig S1 c:
# fig_s2c_emp = CorrelationAnalyzer(data_dir,
#                                   ['input/BAliBASE/features_w_aligner_1_dseq_from_true.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5',
#                                            presentation_name='SoP', correlation_direction=1)],
#                                   '', True, f'Correlation with TCS', 0, 1,
#                                   AlternativeLabelFile('input/combined_scores_vs_true_empr_261125_dedup.csv', 2, 3, 0,
#                                                        0, 1),
#                                   False)
# # fig S1 d:
# fig_s2d_sim = CorrelationAnalyzer(data_dir,
#                                   ['input/OrthoMaM/orthomam_features_w_aligner_090426.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5',
#                                            presentation_name='SoP', correlation_direction=1)],
#                                   '', False, f'Correlation with TCS', 0, 1,
#                                   AlternativeLabelFile('input/combined_scores_vs_true_sim_130925.csv', 2, 3, 0, 0, 1))
#
# plS2 = PlotLayout(14, 12, 2, 2, output_path, 'FigS2')
# plS2.double_col_plot([fig_s2a_emp.get_r(), fig_s2c_emp.get_r(), fig_s2b_sim.get_r(), fig_s2d_sim.get_r()],
#                      ['a', 'c', 'b', 'd'], {})

# # fig S3:
# fig_s3a_emp = CorrelationAnalyzer(data_dir,
#                                   ['input/BAliBASE/model_1/features_w_predictions.csv',
#                                    'input/BAliBASE/model_2/features_w_predictions.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='model1', external_name='predicted_score', presentation_name='Model1',
#                                            correlation_direction=1),
#                                    Measure(key='model2', external_name='predicted_score', presentation_name='Model2',
#                                            correlation_direction=1)],
#                                   empirical_example, False, 'Correlation with Distance from True', 0, 0.75)
#
# fig_s3b_sim = CorrelationAnalyzer(data_dir,
#                                   ['input/OrthoMaM/model_1/features_w_predictions.csv',
#                                    'input/OrthoMaM/model_2/features_w_predictions.csv'],
#                                   'dseq_from_true',
#                                   [Measure(key='model1', external_name='predicted_score', presentation_name='Model1',
#                                            correlation_direction=1),
#                                    Measure(key='model2', external_name='predicted_score', presentation_name='Model2',
#                                            correlation_direction=1)],
#                                   sim_example, False, 'Correlation with Distance from True', 0, 0.75)
#
# pl_s_3 = PlotLayout(14, 12, 2, 2, output_path, 'FigS3')
# pl_s_3.double_col_plot(
#     [fig_s3a_emp.get_example_scatter(), fig_s3a_emp.get_r(), fig_s3b_sim.get_example_scatter(), fig_s3b_sim.get_r()],
#     ['a', 'c', 'b', 'd'], {})


# # fig 4 a, b
# fig_s_4_nuc = CorrelationAnalyzer(data_dir,
#                               ['input/Nuc_PAM/model_1/features_w_predictions.csv'],
#                               'dseq_from_true',
#                               [
#                                         Measure(key='sop', external_name='sp_NucleotidesPAM250_GO_-1.5_GE_0', presentation_name='SoP',
#                                                 correlation_direction=-1),
#                                         Measure(key='model1', external_name='predicted_score', presentation_name='Model1',
#                                                 correlation_direction=1),
#                                ],
#                               '40_S39', False, 'Correlation with Distance from True', 401, 0.75)
#
#
# pl5 = PlotLayout(14, 6, 1, 2, output_path, 'FigS4')
# pl5.double_col_plot([fig_s_4_nuc.get_example_scatter(), fig_s_4_nuc.get_r_scatter()],
#                     ['a', 'b'], {})


# # fig S 5 a:
# fig_s_5a = StackedColGraph(data_dir,
#                        [{'relative_file_path': 'input/Nuc_PAM/model_2/pick_me_trio_overall_v0.csv', 'series_name': 'Nuc_PAM'}],
#                        [Measure(key='', external_name='', presentation_name=None, correlation_direction=0)],
#                        [Measure(key='sop', external_name='SoP', presentation_name='SoP', correlation_direction=0),
#                         Measure(key='model2', external_name='Predicted', presentation_name='Model2',
#                                 correlation_direction=0),
#                         Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and SoP)',
#                                 correlation_direction=0)],
#                        True, ['sop', 'model2'], False)
#
# # fig S 5 b"
# fig_s_5b = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/subsampling_summary.csv', 'series_name': 'Samples'}],
#                         [Measure(key='20', external_name='', presentation_name='20 Samples', correlation_direction=0),
#                                          Measure(key='50', external_name='', presentation_name='50 Samples', correlation_direction=0),
#                                          Measure(key='100', external_name='', presentation_name='100 Samples', correlation_direction=0),
#                                          Measure(key='200', external_name='', presentation_name='200 Samples', correlation_direction=0),
#                                          Measure(key='400', external_name='', presentation_name='400 Samples', correlation_direction=0),
#                                          Measure(key='800', external_name='', presentation_name='800 Samples', correlation_direction=0),
#                                          Measure(key='1600', external_name='', presentation_name='1600 Samples', correlation_direction=0)],
#                         [Measure(key='sop', external_name='SoP', presentation_name='SoP', correlation_direction=0),
#                                   Measure(key='model2', external_name='Predicted', presentation_name='Model2',
#                                           correlation_direction=0),
#                                   Measure(key='tie', external_name='Tie', presentation_name='Tie (Model2 and SoP)',
#                                           correlation_direction=0)],
#                         True, ['sop', 'model2'], True)
#
# pl5 = PlotLayout(16, 7, 1, 2, output_path, 'FigS5')
# pl5.triple_plot(fig_s_5a.subplot, [fig_s_5b.subplot], ['Different Samples Size'], [1, 4])

# fig_s_6a = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/subsampling_summary.csv', 'series_name': 'Samples'}],
#                         [Measure(key='20', external_name='', presentation_name='20 Samples', correlation_direction=0),
#                                          Measure(key='50', external_name='', presentation_name='50 Samples', correlation_direction=0),
#                                          Measure(key='100', external_name='', presentation_name='100 Samples', correlation_direction=0),
#                                          Measure(key='200', external_name='', presentation_name='200 Samples', correlation_direction=0),
#                                          Measure(key='400', external_name='', presentation_name='400 Samples', correlation_direction=0),
#                                          Measure(key='800', external_name='', presentation_name='800 Samples', correlation_direction=0),
#                                          Measure(key='1600', external_name='', presentation_name='1600 Samples', correlation_direction=0)],
#                         [Measure(key='mafft', external_name='MAFFT_Default', presentation_name='MAFFT', correlation_direction=0),
#                                   Measure(key='model2', external_name='MAFFT_Predicted', presentation_name='Model2',
#                                           correlation_direction=0),
#                                   Measure(key='tie', external_name='MAFFT_Tie', presentation_name='Tie (Model2 and MAFFT)',
#                                           correlation_direction=0)],
#                         True, ['mafft', 'model2'], True)
#
# pl6a = PlotLayout(16, 7, 1, 2, output_path, 'FigS6A')
# pl6a.triple_plot(fig_s_5a.subplot, [fig_s_6a.subplot], ['Different Samples Size'], [1, 4])
#
# fig_s_6b = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/subsampling_summary.csv', 'series_name': 'Samples'}],
#                         [Measure(key='20', external_name='', presentation_name='20 Samples', correlation_direction=0),
#                                          Measure(key='50', external_name='', presentation_name='50 Samples', correlation_direction=0),
#                                          Measure(key='100', external_name='', presentation_name='100 Samples', correlation_direction=0),
#                                          Measure(key='200', external_name='', presentation_name='200 Samples', correlation_direction=0),
#                                          Measure(key='400', external_name='', presentation_name='400 Samples', correlation_direction=0),
#                                          Measure(key='800', external_name='', presentation_name='800 Samples', correlation_direction=0),
#                                          Measure(key='1600', external_name='', presentation_name='1600 Samples', correlation_direction=0)],
#                         [Measure(key='prank', external_name='PRANK_Default', presentation_name='PRANK', correlation_direction=0),
#                                   Measure(key='model2', external_name='PRANK_Predicted', presentation_name='Model2',
#                                           correlation_direction=0),
#                                   Measure(key='tie', external_name='PRANK_Tie', presentation_name='Tie (Model2 and PRANK)',
#                                           correlation_direction=0)],
#                         True, ['prank', 'model2'], True)
#
# pl6b = PlotLayout(16, 7, 1, 2, output_path, 'FigS6B')
# pl6b.triple_plot(fig_s_5a.subplot, [fig_s_6b.subplot], ['Different Samples Size'], [1, 4])
#
# fig_s_6c = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/subsampling_summary.csv', 'series_name': 'Samples'}],
#                         [Measure(key='20', external_name='', presentation_name='20 Samples', correlation_direction=0),
#                                          Measure(key='50', external_name='', presentation_name='50 Samples', correlation_direction=0),
#                                          Measure(key='100', external_name='', presentation_name='100 Samples', correlation_direction=0),
#                                          Measure(key='200', external_name='', presentation_name='200 Samples', correlation_direction=0),
#                                          Measure(key='400', external_name='', presentation_name='400 Samples', correlation_direction=0),
#                                          Measure(key='800', external_name='', presentation_name='800 Samples', correlation_direction=0),
#                                          Measure(key='1600', external_name='', presentation_name='1600 Samples', correlation_direction=0)],
#                         [Measure(key='muscle', external_name='Muscle_Default', presentation_name='Muscle', correlation_direction=0),
#                                   Measure(key='model2', external_name='Muscle_Predicted', presentation_name='Model2',
#                                           correlation_direction=0),
#                                   Measure(key='tie', external_name='Muscle_Tie', presentation_name='Tie (Model2 and Muscle)',
#                                           correlation_direction=0)],
#                         True, ['muscle', 'model2'], True)
#
# pl6c = PlotLayout(16, 7, 1, 2, output_path, 'FigS6C')
# pl6c.triple_plot(fig_s_5a.subplot, [fig_s_6c.subplot], ['Different Samples Size'], [1, 4])
#
# fig_s_6d = StackedColGraph(data_dir,
#                         [{'relative_file_path': 'input/OrthoMaM/subsampling_summary.csv', 'series_name': 'Samples'}],
#                         [Measure(key='20', external_name='', presentation_name='20 Samples', correlation_direction=0),
#                                          Measure(key='50', external_name='', presentation_name='50 Samples', correlation_direction=0),
#                                          Measure(key='100', external_name='', presentation_name='100 Samples', correlation_direction=0),
#                                          Measure(key='200', external_name='', presentation_name='200 Samples', correlation_direction=0),
#                                          Measure(key='400', external_name='', presentation_name='400 Samples', correlation_direction=0),
#                                          Measure(key='800', external_name='', presentation_name='800 Samples', correlation_direction=0),
#                                          Measure(key='1600', external_name='', presentation_name='1600 Samples', correlation_direction=0)],
#                         [Measure(key='baliphy', external_name='BAli-Phy_Default', presentation_name='BAli-Phy', correlation_direction=0),
#                                   Measure(key='model2', external_name='BAli-Phy_Predicted', presentation_name='Model2',
#                                           correlation_direction=0),
#                                   Measure(key='tie', external_name='BAli-Phy_Tie', presentation_name='Tie (Model2 and BAli-Phy)',
#                                           correlation_direction=0)],
#                         True, ['baliphy', 'model2'], True)
#
# pl6d = PlotLayout(16, 7, 1, 2, output_path, 'FigS6D')
# pl6d.triple_plot(fig_s_5a.subplot, [fig_s_6d.subplot], ['Different Samples Size'], [1, 4])

# # fig s 7:
# fig_s_7_1k = CorrelationAnalyzer(data_dir,
#                               ['input/OrthoMaM/model_2/1k/pretrained_1000_0.5_features_w_predictions.csv'],
#                               'dseq_from_true',
#                               [
#                                     Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
#                                             correlation_direction=-1),
#                                     Measure(key='model2', external_name='predicted_score', presentation_name='Model2',
#                                             correlation_direction=1),
#                                         ],
#                               '1000L1_0.5', False, 'Correlation with Distance from True',
#                               401, 0.75, continue_missing_code=True)
#
# # fig 5 d:
# fig_s_7_5k = CorrelationAnalyzer(data_dir,
#                               ['input/OrthoMaM/model_2/5k/pretrained_5000_0.5_features_w_predictions.csv'],
#                               'dseq_from_true',
#                               [
#                                     Measure(key='sop', external_name='sp_BLOSUM62_GO_-10_GE_-0.5', presentation_name='SoP',
#                                             correlation_direction=-1),
#                                     Measure(key='model2', external_name='predicted_score', presentation_name='Model2',
#                                             correlation_direction=1),
#                                         ],
#                               '5000S1_0.5', False, 'Correlation with Distance from True',
#                               401, 0.75, continue_missing_code=True)
#
#
# pls5 = PlotLayout(14, 6, 1, 2, output_path, 'FigS7')
# pls5.double_col_plot([fig_s_7_1k.get_example_rank_scatter(), fig_s_7_5k.get_example_rank_scatter()],
#                      ['a', 'b'],
#                      {})
