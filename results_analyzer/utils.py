def normalize(v: list[float]):
    list_min: float = min(v)
    list_max: float = max(v)
    val_range = list_max - list_min
    return [(x - list_min) / val_range  for x in v]

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False