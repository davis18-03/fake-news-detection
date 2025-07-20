import logging
import os
from typing import Any

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


def get_classifier(
    finetuned_model_dir: str = "./finetuned_model",
    default_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
) -> Any:
    """Load a fine-tuned model if available, else load the default model. Returns a HuggingFace pipeline."""
    if os.path.isdir(finetuned_model_dir) and os.path.isfile(
        os.path.join(finetuned_model_dir, "config.json")
    ):
        try:
            classifier = pipeline(
                "text-classification",
                model=finetuned_model_dir,
                framework="pt",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info(f"Using fine-tuned model from {finetuned_model_dir}")
            return classifier
        except Exception as e:
            logger.warning(
                f"Failed to load fine-tuned model: {e}\nFalling back to default model."
            )
    classifier = pipeline(
        "text-classification",
        model=default_model,
        framework="pt",
        device=0 if torch.cuda.is_available() else -1,
    )
    logger.info(f"Using default model: {default_model}")
    return classifier


def predict_proba(classifier: Any, texts: list[str]) -> Any:
    """Predict class probabilities for a list of texts using the provided classifier pipeline."""
    import numpy as np

    results = classifier(texts)
    out = []
    for r in results:
        if r["label"] == "NEGATIVE":
            out.append([r["score"], 1 - r["score"]])
        else:
            out.append([1 - r["score"], r["score"]])
    return np.array(out)


def get_device_info() -> str:
    """Return a string describing the current device (GPU or CPU)."""
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU"
