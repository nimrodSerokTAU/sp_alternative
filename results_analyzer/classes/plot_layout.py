import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

from results_analyzer.classes.col_plot import ColPlot
from results_analyzer.classes.pdf_plot import PDFPlot
from results_analyzer.classes.scatter_plot import ScatterPlot
import matplotlib.patches as patches

from results_analyzer.classes.stacked_col_graphs import StackedColSubPlot, StackedColGraphData


class PlotLayout:
    size_x: int
    size_y: int
    row_count: int
    col_count: int
    dir_path: str
    identifier: str

    def __init__(self, size_x: int, size_y: int, row_count: int, col_count: int, dir_path: str, identifier: str) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.row_count = row_count
        self.col_count = col_count
        self.dir_path = dir_path
        self.identifier = identifier

    def triple_plot(self, single_data: StackedColSubPlot, data_by_labels: list[StackedColSubPlot], titles: list[str]):
        fig = plt.figure(figsize=(self.size_x, self.size_y))  # len(data_by_labels)
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 2, 2],
                      )
        plt.rcParams['hatch.linewidth'] = 0.5
        plt.rcParams['hatch.color'] = '#404245'

        names = ['a', 'b', 'c']
        title_nudge = 0.08

        ax0 = fig.add_subplot(gs[0])
        for i in range(len(single_data.data)):
            stacked_col_data: StackedColGraphData = single_data.data[i]
            label = stacked_col_data.label
            ax0.bar(list(stacked_col_data.x), stacked_col_data.data_by_labels, width=0.8,
                   bottom=list(stacked_col_data.bottom),
                   label=label, color=stacked_col_data.color, hatch=stacked_col_data.hatch)
            self.add_comp_labels(ax0, single_data.labels_list[i]['x'], single_data.labels_list[i]['bottom'],
                                 single_data.labels_list[i]['val'])
            ax0.set_ylabel(single_data.ylabel, fontsize=12)
            ax0.set_xticks(stacked_col_data.x, single_data.categories, fontsize=12)
            ax0.grid(False)
        ax0.text(-title_nudge, 0.98 + title_nudge, names[0], transform=ax0.transAxes, fontsize=14, fontweight='bold',
                va='top')
        h, l = ax0.get_legend_handles_labels()
        handles_m, handles_m = [], []
        handles_m.extend(h)
        handles_m.extend(l)


        handles, labels = [], []
        h, l = None, None
        for j in range(len(data_by_labels)):
            ax = fig.add_subplot(gs[j + 1])
            plot_data = data_by_labels[j]
            for i in range(len(plot_data.data)):
                stacked_col_data: StackedColGraphData = plot_data.data[i]
                label = stacked_col_data.label
                ax.bar(list(stacked_col_data.x), stacked_col_data.data_by_labels, width=0.6,
                           bottom=list(stacked_col_data.bottom),
                           label=label, color=stacked_col_data.color, hatch=stacked_col_data.hatch)
                self.add_comp_labels(ax, plot_data.labels_list[i]['x'], plot_data.labels_list[i]['bottom'],
                                plot_data.labels_list[i]['val'])
                ax.set_title(titles[j], fontsize=12)
                ax.set_ylabel(plot_data.ylabel, fontsize=12)
                ax.set_xticks(stacked_col_data.x, plot_data.categories, fontsize=12)
                ax.text(-title_nudge, 0.98 + title_nudge, names[j + 1], transform=ax.transAxes, fontsize=14,
                         fontweight='bold', va='top')
                h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

        fig.legend(handles_m, handles_m, ncol=2, loc='upper left', fontsize=11,
                   bbox_to_anchor=(0.01, 0.1), frameon=False)
        fig.legend(handles, labels, ncol=3, loc='upper center', fontsize=11,
                   bbox_to_anchor=(0.65, 0.1), frameon=False)
        fig.tight_layout(rect=[0, 0.1, 1, 1], w_pad=4)


        p_val = max(single_data.p_value_per_ds) * 1000

        plt.savefig(f'{self.dir_path}/{self.identifier}_counts_{data_by_labels[0].samples_num}-{data_by_labels[1].samples_num}_millPVal_{p_val:.2f}.tiff')
        plt.savefig(f'{self.dir_path}/{self.identifier}_counts_{data_by_labels[0].samples_num}-{data_by_labels[1].samples_num}_millPVal_{p_val:.2f}.pdf')
        plt.show()
        # axs.clf()
        plt.close()

    def plot_with_inset(self, code: str, main: PDFPlot, insets: list[ScatterPlot]):
        fig, axs = plt.subplots(figsize=(self.size_x, self.size_y))  # len(data_by_labels)
        custom_linestyle = (0, (8, 4))

        sns.kdeplot(main.true_scores, color='black', linewidth=1.5, clip=(0, None))
        number_of_seq = len(main.true_scores)

        for i in range(len(main.labels)):
            plt.scatter(main.x[i], main.y[i], marker=main.markers[i], s=main.s,
                        color=main.colors[i], label=main.labels[i])

        axs.set_xlabel(main.xlabel, fontsize=12)
        axs.set_ylabel(main.ylabel, fontsize=12)
        axs.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.28, 0.2))
        axs.text(-0.04, 1.04, 'a', transform=axs.transAxes, fontsize=14,
                      fontweight='bold', va='top')

        insets_loc = [[0.44, 0.35, 0.25, 0.55], [0.74, 0.35, 0.25, 0.55]]
        inset_titles = ['b', 'c']
        for i in range(len(insets)):
            ax_inset = axs.inset_axes(insets_loc[i])
            ax_inset.scatter(insets[i].x, insets[i].y[0], marker=insets[i].markers[0], linewidths=insets[i].line_widths, color=insets[i].color[0],
                               label=f'{insets[i].names[0]} (r={insets[i].r_val[0]:.2f})', alpha=insets[i].alpha, s=insets[i].s,
                               )
            m, b = np.polyfit(insets[i].x, insets[i].y[0], 1)
            x_line = np.linspace(0, 1, 100)  # 100 points for a smooth line
            y_line = m * x_line + b
            ax_inset.plot(x_line, y_line, color=insets[i].color[0], linewidth=0.5, linestyle=custom_linestyle)

            ax_inset.axvline(min(insets[i].x), color=insets[i].ds_line_color, linestyle=custom_linestyle, linewidth=insets[i].ds_line_linewidth)
            if insets[i].horizontal_line:
                ax_inset.axhline(insets[i].horizontal_line, color=insets[i].ds_line_color, linestyle=custom_linestyle, linewidth=insets[i].ds_line_linewidth)
            ax_inset.set_xlim(insets[i].xlim_min, insets[i].xlim_max)
            ax_inset.set_ylim(insets[i].ylim_min, insets[i].ylim_max)
            ax_inset.set_xlabel(insets[i].xlabel, fontsize=11)
            ax_inset.set_ylabel(insets[i].ylabel, fontsize=11)
            if insets[i].legend_loc:
                ax_inset.legend(loc=insets[i].legend_loc)
            else:
                ax_inset.legend(frameon=False)
            ax_inset.text(-0.1, 1.05, inset_titles[i], transform=ax_inset.transAxes, fontsize=14,
                          fontweight='bold', va='top')

        fig.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f'{self.dir_path}/{self.identifier}_{code}_seq_num_{number_of_seq}.tiff')
        plt.savefig(f'{self.dir_path}/{self.identifier}_{code}_seq_num_{number_of_seq}.pdf')
        plt.show()
        fig.clf()
        plt.close()

    def double_col_plot(self, all_data: list, titles: list[str], patches_dict: dict[int,list[dict]]):
        fig = plt.figure(figsize=(self.size_x, self.size_y))

        gs = GridSpec(len(titles) // 2, 2, figure=fig)
        plt.rcParams['hatch.linewidth'] = 0.5
        plt.rcParams['hatch.color'] = 'white'#,'#404245'
        custom_linestyle = (0, (8, 4))
        title_nudge = 0.06


        for j, data in enumerate(all_data):
            row = j // 2
            col = j % 2
            ax = fig.add_subplot(gs[row, col])

            if isinstance(data, ScatterPlot):
                for i in range(len(data.names)):
                    ax.scatter(data.x, data.y[i], marker=data.markers[i], linewidths=data.line_widths, color=data.color[i], label=f'{data.names[i]} (r={data.r_val[i]:.2f})',
                                alpha=data.alpha, facecolor='None', edgecolor=data.color[i], s=data.s)
                    m, b = np.polyfit(data.x, data.y[i], 1)
                    x_line = np.linspace(0, 1, 100)  # 100 points for a smooth line
                    y_line = m * x_line + b
                    ax.plot(x_line, y_line, color=data.color[i], linewidth=0.5, linestyle=custom_linestyle)

                ax.axvline(min(data.x), color=data.ds_line_color, linestyle=custom_linestyle, linewidth=data.ds_line_linewidth)
                if data.horizontal_line:
                    ax.axhline(data.horizontal_line, color=data.ds_line_color, linestyle=custom_linestyle, linewidth=data.ds_line_linewidth)
                ax.set_xlim(data.xlim_min, data.xlim_max)
                ax.set_ylim(data.ylim_min, data.ylim_max)
                ax.set_xlabel(data.xlabel, fontsize=11)
                ax.set_ylabel(data.ylabel, fontsize=11)
                if data.legend_loc:
                    ax.legend(loc=data.legend_loc)
                else:
                    ax.legend()
                ax.text(-title_nudge, 0.98 + title_nudge, titles[j], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
                if j in patches_dict:
                    for patch in patches_dict[j]:
                        ellipse = patches.Ellipse(xy=patch['center'], width=patch['width'], height=patch['height'], angle=patch['angle'], edgecolor=patch['color'], facecolor='none')
                        ax.add_patch(ellipse)


            if isinstance(data, ColPlot):
                hatches = []
                for h in data.hatch:
                    for col_i in range(data.df.shape[0]):
                        hatches.append(h)
                data.df.plot(kind='bar', ax=ax, color=data.color, alpha=data.alpha, fontsize=10, width=0.7)
                for i, patch in enumerate(ax.patches):
                    patch.set_hatch(hatches[i % len(hatches)])  # Assign hatches cyclically
                plt.xticks(rotation='horizontal')
                ax.set_xlabel(data.xlabel, fontsize=11)
                ax.set_ylabel(data.ylabel, fontsize=11)
                ax.legend(labels=data.names, fontsize=12)
                ax.text(-title_nudge, 0.98 + title_nudge, titles[j], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

        fig.tight_layout(rect=[0, 0, 1, 1])



        data_count_string = '_'.join([str(data.data_count) for data in all_data])
        plt.savefig(
            f'{self.dir_path}/{self.identifier}_{data_count_string}.tiff')
        plt.savefig(
            f'{self.dir_path}/{self.identifier}_{data_count_string}.pdf')
        plt.show()
        plt.clf()
        plt.close()

    @staticmethod
    def add_comp_labels(fig, x, h: list[float], values: list[float]):
        for i in range(len(x)):
            this_h = h[i] + values[i] * 0.75
            fig.text(i, this_h, f'{values[i]:.1f}%', ha='center', va='top', fontsize=12)

