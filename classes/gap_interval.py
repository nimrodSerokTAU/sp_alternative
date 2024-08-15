class GapInterval:
    start: int
    end: int

    def __init__(self):
        pass

    def set_start(self, start: int):
        self.start = start

    def set_end(self, end: int):
        self.end = end

    def get_len(self) -> int:
        return self.end - self.start + 1

    def g_cost(self, gs_cost: float, ge_cost: float) -> float:
        return self.get_len() * ge_cost + gs_cost

    def is_empty(self) -> bool:
        return not hasattr(self, 'start')

    def copy_me(self) -> 'GapInterval':
        new_one = GapInterval()
        new_one.set_start(self.start)
        new_one.set_end(self.end)
        return new_one

