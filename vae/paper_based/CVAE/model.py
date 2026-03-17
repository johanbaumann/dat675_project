"""Compatibility wrapper for legacy imports.

The project standardizes on `model_labels.CVAE` as the single maintained model
implementation. Importing `model.CVAE` remains supported for older scripts and
checkpoints.
"""

from model_labels import CVAE as _UnifiedCVAE


class CVAE(_UnifiedCVAE):
    """Backwards-compatible alias of the unified CVAE implementation."""

    pass
