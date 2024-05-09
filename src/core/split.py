class Split:
    def __init__(self, name, cfg) -> None:
        self.name = name
        self.n_sims = cfg.n_sims
        self.ps_fidu_fixed = cfg.get("ps_fidu_fixed", None)

    def __str__(self) -> str:
        return self.name

    def iter_sims(self):
        for i in range(self.n_sims):
            yield i
