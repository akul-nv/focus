# SPDX-License-Identifier: Apache-2.0
"""VLM integration: pipelines, context tracking, spatial attention
analysis, attention feedback, and value-cache focus calibration.

.. rubric:: Extensibility

Subclass :class:`BaseVLMPipeline` to integrate a different VLM backend
(for example, another locally hosted model).
"""

from .attention import AttentionMapExtractor
from .base import BaseVLMPipeline
from .context import VLMContextManager
from .focus import SpatialFocusCalibrator
from .inference import BurstState, VLMInferenceQueue, VLMResponse
from .model_pool import load_vlm_pipeline
from .spatial import SpatialAttentionAnalyzer

__all__ = [
    "AttentionMapExtractor",
    "BaseVLMPipeline",
    "BurstState",
    "SpatialAttentionAnalyzer",
    "SpatialFocusCalibrator",
    "VLMContextManager",
    "VLMInferenceQueue",
    "VLMResponse",
    "load_vlm_pipeline",
]
