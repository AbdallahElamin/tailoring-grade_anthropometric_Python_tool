# tailorvision/fit/__init__.py
# Lazy imports — torch is only loaded when these classes are instantiated,
# not at collection time.  This allows test modules that don't use the
# fitting engine to import cleanly without torch installed.

def __getattr__(name: str):
    _MAP = {
        "BodyModelAdapter":        ("tailorvision.fit.body_model_adapter", "BodyModelAdapter"),
        "SMPLX_CANONICAL_HEIGHT_M":("tailorvision.fit.body_model_adapter", "SMPLX_CANONICAL_HEIGHT_M"),
        "PoseFitEngine":           ("tailorvision.fit.pose_fit_engine",    "PoseFitEngine"),
        "FitResult":               ("tailorvision.fit.pose_fit_engine",    "FitResult"),
        "AnthropometricPrior":     ("tailorvision.fit.anthropometric_prior","AnthropometricPrior"),
    }
    if name in _MAP:
        import importlib
        mod_name, attr = _MAP[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'tailorvision.fit' has no attribute {name!r}")

__all__ = [
    "BodyModelAdapter", "SMPLX_CANONICAL_HEIGHT_M",
    "PoseFitEngine", "FitResult", "AnthropometricPrior",
]
