"""Hierarchical inference scaffolding for the chitinase classifier.

This module provides a clean orchestration layer that accepts a raw amino-acid
sequence, extracts the hybrid features, and sequentially routes the sample
through the three prediction levels.

The concrete neural network loaders and architecture-specific wrappers will be
filled in ``hierarchy.py`` in the next refactor step. For now, this file keeps
the public API, routing logic, and calibration hooks isolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .features import HybridFeatureBundle, HybridFeatureExtractor


@dataclass(frozen=True)
class HierarchyThresholds:
    """Decision thresholds for routing between hierarchy levels."""

    level1_positive_probability: float = 0.50
    level2_min_confidence: float = 0.50
    level3_min_confidence: float = 0.50


@dataclass(frozen=True)
class RawPredictionOutput:
    """Uncalibrated output returned by a predictor backend."""

    probabilities: np.ndarray
    logits: np.ndarray | None = None
    uncertainties: Any = field(default_factory=dict)


@dataclass(frozen=True)
class LevelPrediction:
    """Prediction summary for one hierarchy level."""

    probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    logits: np.ndarray | None = None
    uncertainties: Any = field(default_factory=dict)
    predicted_index: int | None = None

    @property
    def predicted_probability(self) -> float:
        """Return the calibrated probability of the predicted class."""

        if self.predicted_index is None or self.calibrated_probabilities.size == 0:
            return float("nan")
        flat = np.asarray(self.calibrated_probabilities).reshape(-1)
        index = min(self.predicted_index, flat.size - 1)
        return float(flat[index])


@dataclass(frozen=True)
class HierarchicalPrediction:
    """End-to-end result for a single sequence."""

    sequence: str
    features: HybridFeatureBundle
    level1: LevelPrediction
    level2: LevelPrediction | None = None
    level3: LevelPrediction | None = None
    final_label: str | int | None = None


@runtime_checkable
class PredictorProtocol(Protocol):
    """Minimal interface expected from a model wrapper."""

    def predict_with_uncertainty(self, inputs: Any, **kwargs: Any) -> Any:
        """Return probabilities/logits/uncertainties for ``inputs``."""


Calibrator = Callable[[np.ndarray, np.ndarray | None], np.ndarray]


def _identity_calibrator(probabilities: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
    return np.asarray(probabilities)


def _coerce_prediction_output(output: Any) -> RawPredictionOutput:
    """Normalize predictor outputs to ``RawPredictionOutput``."""

    if isinstance(output, RawPredictionOutput):
        return output

    if isinstance(output, tuple):
        if len(output) == 3:
            probabilities, logits, uncertainties = output
        elif len(output) == 2:
            probabilities, logits = output
            uncertainties = {}
        elif len(output) == 1:
            probabilities = output[0]
            logits = None
            uncertainties = {}
        else:
            raise ValueError("Unexpected predictor output tuple length")
        return RawPredictionOutput(
            probabilities=np.squeeze(np.asarray(probabilities)),
            logits=None if logits is None else np.squeeze(np.asarray(logits)),
            uncertainties=uncertainties,
        )

    if hasattr(output, "probabilities"):
        probabilities = np.squeeze(np.asarray(getattr(output, "probabilities")))
        logits = getattr(output, "logits", None)
        uncertainties = getattr(output, "uncertainties", {}) or {}
        return RawPredictionOutput(
            probabilities=probabilities,
            logits=None if logits is None else np.squeeze(np.asarray(logits)),
            uncertainties=uncertainties,
        )

    raise TypeError(
        "Predictor outputs must be a RawPredictionOutput, a tuple, or an object "
        "with a `probabilities` attribute."
    )


def _call_predictor(
    predictor: PredictorProtocol | Callable[..., Any] | None,
    inputs: Any,
    **kwargs: Any,
) -> RawPredictionOutput:
    """Call a predictor backend and normalize its output."""

    if predictor is None:
        raise NotImplementedError(
            "No predictor has been attached yet. Wire the architecture-specific "
            "model wrappers in hierarchy.py and pass them into the inference "
            "pipeline."
        )

    if hasattr(predictor, "predict_with_uncertainty"):
        output = predictor.predict_with_uncertainty(inputs, **kwargs)
    elif callable(predictor):
        output = predictor(inputs, **kwargs)
    else:
        raise TypeError("predictor must be callable or expose predict_with_uncertainty()")

    return _coerce_prediction_output(output)


def _apply_calibrator(
    probabilities: np.ndarray,
    logits: np.ndarray | None,
    calibrator: Calibrator | None,
) -> np.ndarray:
    if calibrator is None:
        return np.squeeze(np.asarray(probabilities))
    return np.squeeze(np.asarray(calibrator(np.asarray(probabilities), logits)))


def _select_prediction_index(probabilities: np.ndarray) -> int | None:
    flat = np.asarray(probabilities).reshape(-1)
    if flat.size == 0:
        return None
    return int(np.argmax(flat))


def _binary_positive_probability(probabilities: np.ndarray) -> float:
    flat = np.asarray(probabilities).reshape(-1)
    if flat.size == 0:
        return float("nan")
    if flat.size == 1:
        return float(flat[0])
    return float(flat[-1])


class HierarchicalChitinaseInference:
    """Orchestrate the three-level chitinase inference workflow."""

    def __init__(
        self,
        feature_extractor: HybridFeatureExtractor | None = None,
        level1_predictor: PredictorProtocol | Callable[..., Any] | None = None,
        level2_predictor: PredictorProtocol | Callable[..., Any] | None = None,
        level3_predictor: PredictorProtocol | Callable[..., Any] | None = None,
        *,
        level1_calibrator: Calibrator | None = None,
        level2_calibrator: Calibrator | None = None,
        level3_calibrator: Calibrator | None = None,
        thresholds: HierarchyThresholds | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor or HybridFeatureExtractor()
        self.level1_predictor = level1_predictor
        self.level2_predictor = level2_predictor
        self.level3_predictor = level3_predictor
        self.level1_calibrator = level1_calibrator or _identity_calibrator
        self.level2_calibrator = level2_calibrator or _identity_calibrator
        self.level3_calibrator = level3_calibrator or _identity_calibrator
        self.thresholds = thresholds or HierarchyThresholds()

    def _predict_level(
        self,
        predictor: PredictorProtocol | Callable[..., Any] | None,
        inputs: Any,
        calibrator: Calibrator | None,
        **kwargs: Any,
    ) -> LevelPrediction:
        raw = _call_predictor(predictor, inputs, **kwargs)
        calibrated = _apply_calibrator(raw.probabilities, raw.logits, calibrator)
        predicted_index = _select_prediction_index(calibrated)
        return LevelPrediction(
            probabilities=np.squeeze(np.asarray(raw.probabilities)),
            calibrated_probabilities=np.squeeze(np.asarray(calibrated)),
            logits=raw.logits,
            uncertainties=raw.uncertainties,
            predicted_index=predicted_index,
        )

    def predict(
        self,
        sequence: str,
        *,
        mc_passes: int = 20,
        latent_samples: int = 20,
    ) -> HierarchicalPrediction:
        """Predict the hierarchy for a single amino-acid sequence."""

        features = self.feature_extractor.transform_one(sequence)
        model_inputs = features.as_model_inputs()

        level1 = self._predict_level(
            self.level1_predictor,
            model_inputs,
            self.level1_calibrator,
            T=mc_passes,
        )

        final_label: str | int | None = None
        positive_probability = _binary_positive_probability(level1.calibrated_probabilities)
        if positive_probability < self.thresholds.level1_positive_probability:
            final_label = 0
            return HierarchicalPrediction(
                sequence=features.sequence,
                features=features,
                level1=level1,
                final_label=final_label,
            )

        level2 = self._predict_level(
            self.level2_predictor,
            model_inputs,
            self.level2_calibrator,
            T=mc_passes,
        )

        level3 = self._predict_level(
            self.level3_predictor,
            model_inputs,
            self.level3_calibrator,
            T=mc_passes,
            S=latent_samples,
        )

        final_label = level3.predicted_index
        return HierarchicalPrediction(
            sequence=features.sequence,
            features=features,
            level1=level1,
            level2=level2,
            level3=level3,
            final_label=final_label,
        )

    def predict_many(
        self,
        sequences: Sequence[str],
        *,
        mc_passes: int = 20,
        latent_samples: int = 20,
    ) -> list[HierarchicalPrediction]:
        """Predict a list of sequences one by one."""

        return [
            self.predict(
                sequence,
                mc_passes=mc_passes,
                latent_samples=latent_samples,
            )
            for sequence in sequences
        ]


__all__ = [
    "HierarchicalChitinaseInference",
    "HierarchicalPrediction",
    "HierarchyThresholds",
    "LevelPrediction",
    "PredictorProtocol",
    "RawPredictionOutput",
]
