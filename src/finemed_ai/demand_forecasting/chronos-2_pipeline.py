from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

import torch
import transformers
import accelerate
import huggingface_hub

from chronos import Chronos2Pipeline

logger = logging.getLogger(__name__)

class ChronosPipelineLoader:
    def __init__(self,model_name: str = "amazon/chronos-2") -> None:
        self.model_name = model_name

        self.device: str | None = None

        self.pipeline: Chronos2Pipeline | None = None

    def check_environment(self) -> None:
        logger.info("=" * 80)
    logger.info("Chronos Environment Check")
    logger.info("=" * 80)

    logger.info("Python           : %s", platform.python_version())
    logger.info("PyTorch          : %s", torch.__version__)
    logger.info("Transformers     : %s", transformers.__version__)
    logger.info("Accelerate       : %s", accelerate.__version__)
    logger.info("HF Hub           : %s", huggingface_hub.__version__)

    # Basic Validation

    if torch.__version__ is None:
        raise RuntimeError("PyTorch is not installed.")

    if transformers.__version__ is None:
        raise RuntimeError("Transformers is not installed.")

    if accelerate.__version__ is None:
        raise RuntimeError("Accelerate is not installed.")

    if huggingface_hub.__version__ is None:
        raise RuntimeError("HuggingFace Hub is not installed.")

    logger.info("Environment Validation Passed")

    def get_device(self) -> str:

        # CUDA GPU

        if torch.cuda.is_available():
        
                self.device = "cuda"
        
                logger.info(
                    "CUDA Available : %s",
                    torch.cuda.get_device_name(0)
                )
        
                return self.device
        
        # Apple Silicon
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.device = "mps"

            logger.info("Apple MPS Available")

            return self.device

        #CPU
        self.device = "cpu"

        logger.warning("CUDA not available. Running on CPU.")

        return self.device

    def load_model(self) -> Chronos2Pipeline:
        logger.info("=" * 80)
    logger.info("Loading Chronos-2 Model")
    logger.info("=" * 80)

    # Make sure device has been selected
    if self.device is None:
        self.get_device()

    try:

        # Select datatype
        if self.device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        logger.info("Model Name : %s", self.model_name)
        logger.info("Device     : %s", self.device)
        logger.info("Torch Dtype: %s", dtype)

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=dtype,
        )

        logger.info("Chronos-2 Loaded Successfully")

        return self.pipeline

    except Exception as e:

        logger.exception("Chronos-2 Loading Failed")

        raise RuntimeError(
            f"Unable to load Chronos-2 model : {e}"
        ) from e

    def model_summary(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Chronos-2 model has not been loaded. Run load_model() first.")

        logger.info("=" * 80)
        logger.info("Chronos-2 Model Summary")
        logger.info("=" * 80)

        logger.info("Model Name : %s", self.model_name)
        logger.info("Device     : %s", self.device)

        try:
            model = self.pipeline.model

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters()if p.requires_grad)

            dtype = next(model.parameters()).dtype
            logger.info("Torch Dtype       : %s", dtype)
            logger.info("Total Parameters  : %s", f"{total_params:,}")
            logger.info("Trainable Params  : %s", f"{trainable_params:,}")

        except AttributeError:
            logger.warning("Unable to retrieve model parameter information.")

            logger.info("=" * 80)

    def get_pipeline(self) -> Chronos2Pipeline:
        if self.pipeline is None:
            logger.info("Chronos Pipeline not loaded. Loading now...")

            self.load_model()

        return self.pipeline