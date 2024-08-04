class GapInterval:
    start: int
    end: int
    gs_cost: int
    ge_cost: int

    def __init__(self, gs_cost: int, ge_cost: int):
        self.gs_cost = gs_cost
        self.ge_cost = ge_cost

    def set_start(self, start: int):
        self.start = start

    def set_end(self, end: int):
        self.end = end

    def get_len(self) -> int:
        return self.end - self.start + 1

    def g_cost(self) -> int:
        return self.get_len() * self.ge_cost + self.gs_cost

    def is_empty(self) -> bool:
        return not hasattr(self, 'start')

    def copy_me(self) -> 'GapInterval':
        new_one = GapInterval(self.gs_cost, self.ge_cost)
        new_one.set_start(self.start)
        new_one.set_end(self.end)
        return new_one

    def is_equal_to(self, other_interval: 'GapInterval') -> bool:
        return self.start == other_interval.start and self.end == other_interval.end

    def is_included_in(self, other_interval: 'GapInterval') -> bool:
        return self.start >= other_interval.start and self.end <= other_interval.end

    def intersection_with(self, other_interval: 'GapInterval') -> int:
        return min(self.end, other_interval.end) - max(self.start, other_interval.start)
