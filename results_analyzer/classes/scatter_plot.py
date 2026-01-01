import numpy as np

class ScatterPlot:
    x: np.array
    y: list[list[float]]
    names: list[str]
    markers: list[str]
    color: list[str]
    edge_color: list[str]
    label: list[str]
    r_val: list[float]
    line_widths: float
    alpha: float
    s: float
    ds_line_color: str
    ds_line_linestyle: str
    ds_line_linewidth: float
    xlim_min: float
    xlim_max: float
    ylim_min: float
    ylim_max: float
    xlabel: str
    ylabel: str
    horizontal_line: float
    data_count: int

    def __init__(self, x: np.array, markers: list[str],
                 line_widths: float, alpha: float, s: float, axvline_color: str, axvline_linestyle: str,
                 axvline_linewidth: float, xlabel: str, ylabel: str, legend_loc: str):
        self.x = x
        self.markers = markers
        self.line_widths = line_widths
        self.alpha = alpha
        self.s = s
        self.ds_line_color = axvline_color
        self.ds_line_linestyle = axvline_linestyle
        self.ds_line_linewidth = axvline_linewidth
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend_loc = legend_loc

    def set_data(self, y: list[list[float]], names: list[str], color: list[str], label: list[str],
                        r_val: list[float], xlim_min: float, xlim_max: float, ylim_min: float, ylim_max: float,
                        horizontal_line: float | None, data_count: int):
        self.y = y
        self.names = names
        self.color = color
        self.edge_color = color
        self.r_val = r_val
        self.label = label
        self.xlim_min = xlim_min
        self.xlim_max = xlim_max
        self.ylim_min = ylim_min
        self.ylim_max = ylim_max
        self.data_count = data_count
        self.horizontal_line = horizontal_line

