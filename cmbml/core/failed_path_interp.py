class FailedPathInterpolationSentinel:
    """
    Indicates that path interpolation failed while reading an asset path template.
    Stores the original exception so later failures are explainable.
    """

    def __init__(self, *, asset_name=None, source_stage=None, error=None):
        self.asset_name = asset_name
        self.source_stage = source_stage
        self.error = error

    def __repr__(self):
        return (
            "FailedPathInterpolationSentinel("
            f"asset_name={self.asset_name!r}, "
            f"source_stage={self.source_stage!r}, "
            f"error={self.error!r})"
        )

    def __str__(self):
        return (
            f"<failed path interpolation for asset={self.asset_name!r}, "
            f"stage={self.source_stage!r}: {self.error}>"
        )

    def format(self, *args, **kwargs):
        raise RuntimeError(
            "Attempted to format a path template whose interpolation already failed.\n"
            f"asset_name={self.asset_name!r}\n"
            f"source_stage={self.source_stage!r}\n"
            f"original_error={self.error!r}"
        )
