_NOISE_REGISTRY = {}


def register_noise(label: str):
    """Decorator to register a noise generator class under a label."""
    def decorator(cls):
        if label in _NOISE_REGISTRY:
            raise ValueError(f"Noise label already registered: {label}")
        _NOISE_REGISTRY[label] = cls
        return cls
    return decorator


def get_noise_class(label: str):
    """Retrieve a registered noise generator class by label."""
    try:
        return _NOISE_REGISTRY[label]
    except KeyError:
        raise ValueError(f"Unsupported noise type: {label}")


def list_noise_types():
    # Expose registry contents for debugging/inspection
    return sorted(_NOISE_REGISTRY.keys())
