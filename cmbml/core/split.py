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
        self.ps_fidu_fixed = split_cfg.get("ps_fidu_fixed", False)
        self.ps_fidu_planck = split_cfg.get("ps_fidu_planck", False)
        self.noise_fixed = split_cfg.get("noise_fixed", False)
        self.fgs_fixed = split_cfg.get("fgs_fixed", False)
        self.resume_at = split_cfg.get("resume_at", 0)  # For use when, e.g., power goes out. 
                                                        #   User needs to set all complete splits
                                                        #   to "resume_at" the number of simulations
                                                        #   and can then finish unfinished splits.
                                                        #   Run only the interrupted stage!
        if self.ps_fidu_planck and self.ps_fidu_fixed is None:
            self.ps_fidu_fixed = True
        if self.ps_fidu_planck and not self.ps_fidu_fixed:
            raise ValueError("Split cannot have ps_fidu_fixed=False and ps_fidu_planck=True.")

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Split(name={self.name})"

    def iter_sims(self):
        return SplIterator(self)


class SplIterator:
    def __init__(self, split):
        self.split = split
        self.current_sim = split.resume_at

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
        return self.split.n_sims - self.split.resume_at


class Splits:
    def __init__(self, splits, max_display: int = 10):
        self._splits = list(splits)
        self._by_name = {s.name.lower(): s for s in self._splits}
        self._max_display = max_display

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._splits[key]
        elif isinstance(key, str):
            try:
                return self._by_name[key.lower()]
            except KeyError:
                raise KeyError(f"Split with name '{key}' not found.")
        else:
            raise TypeError("Key must be an integer index or a string name.")
    
    def __len__(self):
        return len(self._splits)
    
    def __iter__(self):
        return iter(self._splits)
    
    def __repr__(self):
        names = [s.name for s in self._splits]
        total = len(names)
        if total > self._max_display:
            display  = names[:self._max_display] + [f"... {total - self._max_display} more ..."]
        else:
            display = names
        return f"Splits({display})"
    
    __str__ = __repr__
    
    def names(self):
        return list(self._by_name.keys())
    
    def get(self, name, default=None):
        return self._by_name.get(name.lower(), default)