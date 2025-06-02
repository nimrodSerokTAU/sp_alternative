class BasicStats:
    code: str
    taxa_num: int
    msa_len: int
    ordered_col_names: list[str]

    def __init__(self, code: str, taxa_num: int, msa_len: int, ordered_col_names: list[str]):
        self.code = code
        self.taxa_num = taxa_num
        self.msa_len = msa_len
        self.ordered_col_names = ordered_col_names

    def get_my_features_as_list(self) -> list:
        values: list = []
        attrs = vars(self)
        for col_name in self.ordered_col_names:
            value = attrs[col_name]
            if type(value) == float:
                value = round(value, 3)
            values.append(value)
        return values

    def get_ordered_col_names(self) -> list[str]:
        return self.ordered_col_names
