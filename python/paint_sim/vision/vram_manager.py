"""Dual-Profile VRAM Lifecycle Manager (8 GB Local RTX 3070 vs. 24 GB Cloud L4)."""
from __future__ import annotations

import contextlib
import gc
import logging
import os
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class VRAMManager:
    """
    Implements ADR-005 / SRS Section 6.1:
    - When `low_vram_mode=True` (Local RTX 3070 8 GB):
      Enforces sequential FP16 stage execution (Stage 1: SAM 2.1 ~1.8 GB -> offload to CPU RAM
      & flush CUDA cache -> Stage 2: Intrinsics + Normals ~4.3 GB), keeping peak VRAM <= 5.1 GB
      (strictly under the 7.0 GB ceiling).
    - When `low_vram_mode=False` (GCP L4 24 GB / T4 16 GB):
      Keeps all models resident in VRAM simultaneously (~6.9 GB) for zero swap latency.
    """

    STAGE_VRAM_ESTIMATES_GB: Dict[str, float] = {
        "cuda_context": 0.8,
        "sam_segmentation": 1.8,
        "intrinsics_and_normals": 4.3,
    }

    def __init__(self, low_vram_mode: Optional[bool] = None, safety_ceiling_gb: float = 7.0) -> None:
        if low_vram_mode is None:
            low_vram_mode = os.environ.get("LOW_VRAM_MODE", "true").lower() in ("1", "true", "yes")
        self.low_vram_mode: bool = low_vram_mode
        self.safety_ceiling_gb: float = safety_ceiling_gb
        self.loaded_models: Dict[str, Any] = {}
        self.active_vram_models: Dict[str, float] = {}
        self.peak_estimated_vram_gb: float = self.STAGE_VRAM_ESTIMATES_GB["cuda_context"]

    def current_allocated_gb(self) -> float:
        """Return actual PyTorch CUDA allocated GB plus tracked stage models."""
        cuda_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                cuda_gb = torch.cuda.memory_allocated() / (1024.0 ** 3)
        except Exception:
            pass
        tracked_gb = self.STAGE_VRAM_ESTIMATES_GB["cuda_context"] + sum(self.active_vram_models.values())
        return max(cuda_gb, tracked_gb)

    def flush_cuda(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def offload_to_cpu(self, model_key: str) -> None:
        model = self.loaded_models.get(model_key)
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
        self.active_vram_models.pop(model_key, None)
        self.flush_cuda()

    @contextlib.contextmanager
    def acquire_stage(self, stage_name: str, vram_footprint_gb: Optional[float] = None) -> Iterator[None]:
        if vram_footprint_gb is None:
            vram_footprint_gb = self.STAGE_VRAM_ESTIMATES_GB.get(stage_name, 1.5)

        if self.low_vram_mode:
            # Offload any previously active stage before loading the new stage
            for existing_key in list(self.active_vram_models.keys()):
                if existing_key != stage_name:
                    self.offload_to_cpu(existing_key)

        self.active_vram_models[stage_name] = vram_footprint_gb
        current_gb = self.current_allocated_gb()
        self.peak_estimated_vram_gb = max(self.peak_estimated_vram_gb, current_gb)

        if current_gb > self.safety_ceiling_gb:
            raise MemoryError(
                f"Stage '{stage_name}' exceeded VRAM safety ceiling: {current_gb:.2f} GB > {self.safety_ceiling_gb:.2f} GB"
            )

        try:
            yield
        finally:
            if self.low_vram_mode:
                self.offload_to_cpu(stage_name)
