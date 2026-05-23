import numpy as np
from app.domains.xai.utils import normalize_attribution

try:
    from ensemblexai.ensemble import NormEnsembleXAI
    from ensemblexai.normalize import SecondMomentScaling
    ENSEMBLEXAI_AVAILABLE = True
except ImportError:
    ENSEMBLEXAI_AVAILABLE = False


def compute_ensemble(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute ensemble XAI maps (mean and max) from a dict of attribution maps.
    Uses ensemblexai library if available, otherwise manual implementation.
    """
    if not maps:
        return {}

    attr_list = [normalize_attribution(v) for v in maps.values()]

    if ENSEMBLEXAI_AVAILABLE:
        try:
            norm_mean = NormEnsembleXAI(normalization=SecondMomentScaling(), aggregation="mean")
            norm_max = NormEnsembleXAI(normalization=SecondMomentScaling(), aggregation="max")
            ens_mean = norm_mean.ensemble(attr_list)
            ens_max = norm_max.ensemble(attr_list)
            return {
                "ensemble_mean": normalize_attribution(ens_mean),
                "ensemble_max": normalize_attribution(ens_max),
            }
        except Exception:
            pass

    # Manual fallback
    stack = np.stack(attr_list, axis=0)  # (N, H, W)
    # Second-moment normalization
    for i in range(stack.shape[0]):
        sm = np.sqrt(np.mean(stack[i] ** 2) + 1e-8)
        if sm > 0:
            stack[i] /= sm

    return {
        "ensemble_mean": normalize_attribution(np.mean(stack, axis=0)),
        "ensemble_max": normalize_attribution(np.max(stack, axis=0)),
    }
