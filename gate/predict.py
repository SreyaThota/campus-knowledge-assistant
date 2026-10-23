"""Inference interface for loading trained ScopeGate checkpoints and scoring queries."""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from transformers import AutoTokenizer

from gate.data import LABEL_IN_SCOPE, LABEL_OOS
from gate.model import ScopeGateClassifier, DEFAULT_ENCODER


class ScopeGatePredictor:
    """
    Inference wrapper for a trained ScopeGateClassifier checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        max_length: int = 36,
    ):
        self.device = device or torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.max_length = max_length

        self.model, self.metadata = ScopeGateClassifier.load_from_checkpoint(
            self.checkpoint_path,
            device=self.device,
        )
        encoder_name = self.metadata.get("encoder_name", DEFAULT_ENCODER)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        # Default decision threshold tuned strictly on validation split
        self.threshold = float(self.metadata.get("val_threshold", 0.5))

    def predict_oos_score(
        self,
        texts: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute out-of-scope likelihood scores for a batch of query texts.
        Returns 1D numpy array of probabilities in [0.0, 1.0].
        """
        self.model.eval()
        scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                oos_probs = self.model.get_oos_probabilities(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                scores.extend(oos_probs.cpu().numpy().tolist())

        return np.array(scores, dtype=np.float32)

    def predict(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Classifies texts into LABEL_IN_SCOPE (1) or LABEL_OOS (0).
        """
        th = self.threshold if threshold is None else threshold
        scores = self.predict_oos_score(texts)
        return np.where(scores >= th, LABEL_OOS, LABEL_IN_SCOPE)

    def predict_single(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Classify a single query text.
        """
        th = self.threshold if threshold is None else threshold
        score = float(self.predict_oos_score([text])[0])
        is_oos = score >= th
        return {
            "text": text,
            "oos_score": score,
            "decision_threshold": th,
            "is_in_scope": not is_oos,
            "predicted_label": LABEL_OOS if is_oos else LABEL_IN_SCOPE,
        }
