class Split:
    def __init__(self, name, split_cfg):
        self.name = name
        self.n_sims = split_cfg.n_sims
        self.ps_fidu_fixed = split_cfg.get("ps_fidu_fixed", None)

    def __str__(self):
        return self.name

    def iter_sims(self):
        return SplIterator(self)


class SplIterator:
    def __init__(self, split):
        self.split = split
        self.current_sim = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_sim < self.split.n_sims:
            result = self.current_sim
            self.current_sim += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return self.split.n_sims
