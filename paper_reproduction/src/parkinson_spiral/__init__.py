"""parkinson_spiral — paper reproduction package.

Cross-dataset robustness of spiral classifiers for Parkinson's disease diagnosis.
Four pipelines (DINOv2 frozen probe, Huang VGG19, Dogan triple, Folador HOG)
benchmarked across three datasets (Kaggle, Folador, Hemato) under the paper's
augmentation recipes (AUG-None, AUG-Huang, AUG-Dogan).

The naming here matches the paper exactly (datasets kaggle/folador/hemato,
recipes AUG-*), not the internal development codenames.
"""

__version__ = "1.0.0"

# The seed every experiment in the paper was run with (locked).
DEFAULT_SEED = 20260516
