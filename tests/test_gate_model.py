"""Unit tests for gate.model and gate.predict modules."""

import tempfile
from pathlib import Path
import torch
import pytest

from gate.model import ScopeGateClassifier
from gate.predict import ScopeGatePredictor
from gate.data import LABEL_IN_SCOPE, LABEL_OOS


def test_mean_pooling_accuracy():
    """Verify mean pooling ignores masked padding tokens."""
    model = ScopeGateClassifier.__new__(ScopeGateClassifier)
    # 2 tokens of dimension 4
    token_embs = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    ])
    # Attention mask: only 1st token is real, 2nd is padding
    mask = torch.tensor([[1, 0]])

    pooled = model.mean_pooling(token_embs, mask)
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.allclose(pooled, expected)

    # Both tokens unmasked -> average
    mask_both = torch.tensor([[1, 1]])
    pooled_both = model.mean_pooling(token_embs, mask_both)
    expected_both = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    assert torch.allclose(pooled_both, expected_both)


def test_model_forward_shape():
    """Verify forward outputs logits with shape (batch_size, 2)."""
    model = ScopeGateClassifier()
    model.eval()

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, 2)
        oos_probs = model.get_oos_probabilities(input_ids, attention_mask)
        assert oos_probs.shape == (batch_size,)
        assert (oos_probs >= 0.0).all() and (oos_probs <= 1.0).all()


def test_checkpoint_save_and_predict_roundtrip():
    """Verify saving checkpoint and loading via ScopeGatePredictor."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.pt"
        model = ScopeGateClassifier()
        metadata = {
            "val_threshold": 0.45,
            "seed": 42,
        }
        model.save_checkpoint(ckpt_path, metadata=metadata)
        assert ckpt_path.exists()

        predictor = ScopeGatePredictor(checkpoint_path=ckpt_path)
        assert predictor.threshold == 0.45

        res = predictor.predict_single("test query text")
        assert "oos_score" in res
        assert "is_in_scope" in res
        assert res["predicted_label"] in [LABEL_IN_SCOPE, LABEL_OOS]
