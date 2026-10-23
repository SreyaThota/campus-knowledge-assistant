"""Baseline Out-of-Scope (OOS) detection models.

Baselines implemented:
1. ExemplarCosineBaseline: Max cosine similarity to individual in-scope exemplars (all-MiniLM-L6-v2).
2. CentroidCosineBaseline: Cosine similarity to mean in-scope embedding (all-MiniLM-L6-v2).
3. TfidfLogisticRegressionBaseline: TF-IDF n-grams + class-weighted Logistic Regression.
4. FrozenEmbeddingLogisticRegressionBaseline: Frozen all-MiniLM-L6-v2 embeddings + class-weighted LogReg.

All thresholds and hyperparameters are tuned strictly on the validation split.
"""

from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from gate.data import LABEL_IN_SCOPE, LABEL_OOS

logger = logging.getLogger(__name__)


def compute_binary_metrics_at_threshold(
    oos_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute binary classification metrics at a given OOS decision threshold.

    Decision rule:
        If oos_score >= threshold -> Classified as OUT-OF-SCOPE (refused)
        If oos_score < threshold  -> Classified as IN-SCOPE (accepted)

    Args:
        oos_scores: Array of float scores in [0, 1] where higher indicates OOS.
        labels: Ground truth binary labels (LABEL_IN_SCOPE=1, LABEL_OOS=0).
        threshold: Decision boundary for OOS classification.
    """
    is_true_oos = (labels == LABEL_OOS)
    is_true_in = (labels == LABEL_IN_SCOPE)

    pred_oos = (oos_scores >= threshold)
    pred_in = (oos_scores < threshold)

    tp = np.sum(pred_oos & is_true_oos)  # True OOS correctly rejected
    fp = np.sum(pred_oos & is_true_in)   # True in-scope incorrectly rejected (false reject)
    fn = np.sum(pred_in & is_true_oos)   # True OOS incorrectly accepted (false accept)
    tn = np.sum(pred_in & is_true_in)    # True in-scope correctly accepted

    oos_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    oos_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * oos_precision * oos_recall / (oos_precision + oos_recall)) if (oos_precision + oos_recall) > 0 else 0.0
    in_scope_acceptance = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "oos_recall": oos_recall,
        "oos_precision": oos_precision,
        "oos_f1": f1,
        "in_scope_acceptance": in_scope_acceptance,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(
    val_oos_scores: np.ndarray,
    val_labels: np.ndarray,
    criterion: str = "max_f1",
    target_in_scope_retention: float = 0.95,
) -> float:
    """
    Find optimal decision threshold strictly on the validation split.

    Criteria supported:
    - 'max_f1': Maximizes F1 score for detecting out-of-scope queries.
    - 'target_retention': Finds highest OOS recall subject to in_scope_acceptance >= target.
    """
    # Evaluate over fine candidate thresholds
    threshold_candidates = np.linspace(0.01, 0.99, 400)
    best_threshold = 0.5
    best_val = -1.0

    if criterion == "target_retention":
        best_recall = -1.0
        for th in threshold_candidates:
            m = compute_binary_metrics_at_threshold(val_oos_scores, val_labels, th)
            if m["in_scope_acceptance"] >= target_in_scope_retention:
                if m["oos_recall"] > best_recall:
                    best_recall = m["oos_recall"]
                    best_threshold = th
        if best_recall >= 0:
            return float(best_threshold)
        # Fallback to max_f1 if target retention was never met

    for th in threshold_candidates:
        m = compute_binary_metrics_at_threshold(val_oos_scores, val_labels, th)
        if m["oos_f1"] > best_val:
            best_val = m["oos_f1"]
            best_threshold = th

    return float(best_threshold)


class BaseGateBaseline(ABC):
    """Abstract interface for all scope gate baseline models."""

    def __init__(self, name: str):
        self.name = name
        self.val_threshold: Optional[float] = None

    @abstractmethod
    def fit(self, train_texts: List[str], train_labels: List[int]) -> "BaseGateBaseline":
        pass

    @abstractmethod
    def predict_oos_score(self, texts: List[str]) -> np.ndarray:
        """Returns 1D array of floats in [0, 1] representing OOS likelihood."""
        pass

    def tune_threshold_on_validation(
        self,
        val_texts: List[str],
        val_labels: List[int],
        criterion: str = "max_f1",
    ) -> float:
        """Selects and sets decision threshold using validation split only."""
        scores = self.predict_oos_score(val_texts)
        labels_arr = np.array(val_labels, dtype=int)
        self.val_threshold = find_optimal_threshold(scores, labels_arr, criterion=criterion)
        return self.val_threshold

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """
        Classifies texts into LABEL_IN_SCOPE (1) or LABEL_OOS (0).
        """
        th = self.val_threshold if threshold is None else threshold
        if th is None:
            raise ValueError(f"Threshold not tuned for {self.name}. Call tune_threshold_on_validation first.")
        scores = self.predict_oos_score(texts)
        return np.where(scores >= th, LABEL_OOS, LABEL_IN_SCOPE)


class ExemplarCosineBaseline(BaseGateBaseline):
    """
    Max cosine similarity to individual in-scope exemplars using all-MiniLM-L6-v2.
    Mirrors how the assistant's retrieval refusal guardrail functions.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        super().__init__(name="Exemplar Cosine Similarity (Nearest Neighbor)")
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder: Optional[SentenceTransformer] = None
        self.in_scope_embeddings: Optional[np.ndarray] = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "ExemplarCosineBaseline":
        in_scope_texts = [
            t for t, label in zip(train_texts, train_labels)
            if label == LABEL_IN_SCOPE
        ]
        logger.info(f"Encoding {len(in_scope_texts)} in-scope exemplars for ExemplarCosineBaseline...")
        raw_embs = self.encoder.encode(
            in_scope_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.in_scope_embeddings = np.array(raw_embs, dtype=np.float32)
        return self

    def predict_oos_score(self, texts: List[str]) -> np.ndarray:
        if self.in_scope_embeddings is None:
            raise ValueError("Model must be fitted before predict_oos_score.")

        query_embs = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Cosine similarity matrix: Q x E^T (both L2 normalized, so dot product = cosine similarity)
        sim_matrix = np.dot(query_embs, self.in_scope_embeddings.T)
        max_sims = np.max(sim_matrix, axis=1)

        # Invert similarity to get OOS score: lower max similarity -> higher OOS likelihood
        oos_scores = 1.0 - max_sims
        return np.clip(oos_scores, 0.0, 1.0)


class CentroidCosineBaseline(BaseGateBaseline):
    """
    Cosine similarity to the mean centroid of in-scope training exemplars.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        super().__init__(name="Centroid Cosine Similarity")
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder: Optional[SentenceTransformer] = None
        self.centroid: Optional[np.ndarray] = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "CentroidCosineBaseline":
        in_scope_texts = [
            t for t, label in zip(train_texts, train_labels)
            if label == LABEL_IN_SCOPE
        ]
        logger.info(f"Encoding {len(in_scope_texts)} texts to compute Centroid...")
        embs = self.encoder.encode(
            in_scope_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        mean_vector = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_vector)
        self.centroid = mean_vector / (norm + 1e-12)
        return self

    def predict_oos_score(self, texts: List[str]) -> np.ndarray:
        if self.centroid is None:
            raise ValueError("Model must be fitted before predict_oos_score.")

        query_embs = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims = np.dot(query_embs, self.centroid)
        oos_scores = 1.0 - sims
        return np.clip(oos_scores, 0.0, 1.0)


class TfidfLogisticRegressionBaseline(BaseGateBaseline):
    """
    TF-IDF (word n-grams 1-2) with class-weighted Logistic Regression.
    """

    def __init__(self, seed: int = 42, max_features: int = 10000):
        super().__init__(name="TF-IDF + Logistic Regression")
        self.seed = seed
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        self.is_fitted = False

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "TfidfLogisticRegressionBaseline":
        X = self.vectorizer.fit_transform(train_texts)
        y = np.array(train_labels, dtype=int)
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_oos_score(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_oos_score.")
        X = self.vectorizer.transform(texts)
        proba = self.classifier.predict_proba(X)
        # Find index for LABEL_OOS (0)
        oos_col_idx = list(self.classifier.classes_).index(LABEL_OOS)
        return proba[:, oos_col_idx]


class FrozenEmbeddingLogisticRegressionBaseline(BaseGateBaseline):
    """
    Frozen all-MiniLM-L6-v2 embeddings with class-weighted Logistic Regression.
    Direct ablation to determine whether end-to-end fine-tuning adds value over frozen representations.
    """

    def __init__(
        self,
        seed: int = 42,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        super().__init__(name="Frozen Embeddings + Logistic Regression")
        self.seed = seed
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder: Optional[SentenceTransformer] = None
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
            solver="lbfgs",
        )
        self.is_fitted = False

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def fit(self, train_texts: List[str], train_labels: List[int]) -> "FrozenEmbeddingLogisticRegressionBaseline":
        logger.info(f"Extracting frozen embeddings for {len(train_texts)} training queries...")
        X = self.encoder.encode(
            train_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        y = np.array(train_labels, dtype=int)
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_oos_score(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_oos_score.")
        X = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        proba = self.classifier.predict_proba(X)
        oos_col_idx = list(self.classifier.classes_).index(LABEL_OOS)
        return proba[:, oos_col_idx]
