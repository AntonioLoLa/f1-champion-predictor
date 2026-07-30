"""SHAP and permutation-importance explainability for the deployed model."""

from f1cp.explain.permutation import PermutationResult
from f1cp.explain.permutation import explain as explain_permutation
from f1cp.explain.plots import permutation_importance_figure, shap_importance_figure
from f1cp.explain.shap_explainer import ShapResult
from f1cp.explain.shap_explainer import explain as explain_shap

__all__ = [
    "PermutationResult",
    "ShapResult",
    "explain_permutation",
    "explain_shap",
    "permutation_importance_figure",
    "shap_importance_figure",
]
