class KdesPlot:
    values: list[list[float]]
    colors: list[str]
    names: list[str]

    def __init__(self, values: list[list[float]], colors: list[str], names: list[str]):
        self.values = values
        self.colors = colors
        self.names = names

