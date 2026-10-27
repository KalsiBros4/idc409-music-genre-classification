"""
model_utils.py
--------------
Model evaluation and selection utilities for the IDC409 Music Genre Classification project.
"""

from typing import Dict, Any, Sequence


def select_best_model(
    cv_results: Dict[str, Any],
    test_results: Dict[str, Any] = None,
    metric: str = "mean_f1",
    exclude: Sequence[str] = ("Baseline (Majority)",),
) -> str:
    """
    Select the winning model name based strictly on mean cross-validation macro F1
    from the training set.

    Methodological Rule:
    -------------------
    The winning model must be chosen strictly using cross-validation on the
    training data. The held-out test set is reserved exclusively for final
    unbiased evaluation of generalization performance. Therefore, even if
    `test_results` is provided, it is intentionally never consulted for model
    selection. Baseline models are excluded from eligibility.

    Parameters
    ----------
    cv_results : dict
        Mapping of model_name -> dict with validation metrics (must contain `metric` key).
    test_results : dict, optional
        Held-out test set evaluation results. Ignored during model selection.
    metric : str, default="mean_f1"
        The cross-validation metric key to maximize.
    exclude : tuple of str, default=("Baseline (Majority)",)
        Names of trivial/baseline models not eligible to be chosen as the winner.

    Returns
    -------
    str
        The name of the winning model.
    """
    eligible = {name: metrics for name, metrics in cv_results.items() if name not in exclude}
    if not eligible:
        raise ValueError("No eligible candidate models available for selection.")

    def _get_cv_score(val):
        if isinstance(val, dict):
            return val.get(metric, val.get("CV macro-F1 (mean)", 0.0))
        return float(val)

    return max(eligible, key=lambda name: _get_cv_score(eligible[name]))
