"""PyTorch classification model for Learned Out-of-Scope (OOS) Scope Gate.

Backbone: sentence-transformers/all-MiniLM-L6-v2 with mean pooling.
Head: Dropout(0.2) + Linear(hidden_dim, 2).
Output classes:
  - Class 0: LABEL_OOS (out-of-scope)
  - Class 1: LABEL_IN_SCOPE (in-scope)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from gate.data import LABEL_IN_SCOPE, LABEL_OOS

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class ScopeGateClassifier(nn.Module):
    """
    Scope Gate neural classifier built on a transformer encoder backbone.
    """

    def __init__(
        self,
        encoder_name: str = DEFAULT_ENCODER,
        dropout_prob: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean pooling across non-padding tokens.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass producing unnormalized logits of shape (batch_size, 2).
        Logits index 0: LABEL_OOS
        Logits index 1: LABEL_IN_SCOPE
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs[0]  # First element is last_hidden_state
        pooled = self.mean_pooling(token_embeddings, attention_mask)
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits

    def get_oos_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute P(OOS) for each query in the batch.
        Returns tensor of shape (batch_size,) with values in [0.0, 1.0].
        """
        logits = self.forward(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        # Index 0 corresponds to LABEL_OOS
        return probs[:, LABEL_OOS]

    def save_checkpoint(
        self,
        save_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model weights and metadata dictionary.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "encoder_name": self.encoder_name,
            "hidden_size": self.hidden_size,
            "metadata": metadata or {},
        }
        torch.save(checkpoint, save_path)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple["ScopeGateClassifier", Dict[str, Any]]:
        """
        Load model weights and metadata from checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(encoder_name=checkpoint.get("encoder_name", DEFAULT_ENCODER))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        metadata = checkpoint.get("metadata", {})
        return model, metadata
