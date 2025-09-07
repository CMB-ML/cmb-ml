from omegaconf.errors import InterpolationKeyError


class Split:
    def __init__(self, name, split_cfg):
        self.name = name
        # if a cap is specified, only get that many sims
        try:
            self.n_sims = split_cfg.get("n_sims_cap", split_cfg.n_sims)
        except InterpolationKeyError:
            # We may not specify a cap, e.g. for simulations, because there's no inference
            self.n_sims = split_cfg.n_sims
        if self.n_sims is None:
            # This happens when n_sims_cap is set to null
            self.n_sims = split_cfg.n_sims
        self.ps_fidu_fixed = split_cfg.get("ps_fidu_fixed", None)
        self.ps_fidu_planck = split_cfg.get("ps_fidu_planck", None)
        if self.ps_fidu_planck and self.ps_fidu_fixed is None:
            self.ps_fidu_fixed = True
        if self.ps_fidu_planck and not self.ps_fidu_fixed:
            raise ValueError("Split cannot have ps_fidu_fixed=False and ps_fidu_planck=True.")

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

    def __next__(self) -> int:
        if self.current_sim < self.split.n_sims:
            result = self.current_sim
            self.current_sim += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.split.n_sims
