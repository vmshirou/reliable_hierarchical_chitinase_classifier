"""Model wrappers and calibration utilities for the hierarchical classifier.

This module turns the research notebooks into reusable, production-friendly
predictor classes. Each wrapper accepts file paths to trained artifacts and
exposes a uniform ``predict_with_uncertainty`` interface that the inference
pipeline can orchestrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression

from .artifacts import (
    LEVEL1_MODEL_SPEC,
    LEVEL2_MODEL_SPEC,
    LEVEL3_CLASSIFIER_SPEC,
    LEVEL3_ENCODER_SPEC,
    resolve_drive_artifact_path,
)
from .inference import RawPredictionOutput


def _to_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _load_keras_model(
    path: str | Path,
    *,
    custom_objects: Mapping[str, Any] | None = None,
) -> keras.Model:
    """Load a Keras model for inference without requiring compilation metadata."""

    return keras.models.load_model(
        _to_path(path),
        custom_objects=custom_objects,
        compile=False,
    )


def _as_numpy(array: Any) -> np.ndarray:
    return np.asarray(array)


def _squeeze_keep_batch(array: Any) -> np.ndarray:
    return np.squeeze(_as_numpy(array))


@tf.keras.utils.register_keras_serializable(package="chitinase_clf")
class MatrixScaler(tf.keras.Model):
    """Matrix scaling calibrator used by the second hierarchical level."""

    def __init__(self, num_classes: int = 7, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.W = self.add_weight(
            name="weights",
            shape=(num_classes, num_classes),
            initializer=tf.keras.initializers.Identity(),
            trainable=True,
            dtype=tf.float64,
        )
        self.b = self.add_weight(
            name="bias",
            shape=(num_classes,),
            initializer=tf.zeros_initializer(),
            trainable=True,
            dtype=tf.float64,
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.loss_tracker]

    def call(self, logits: Any) -> tf.Tensor:
        logits = tf.cast(logits, tf.float64)
        return tf.matmul(logits, self.W) + self.b

    def _compute_loss(self, logits: Any, labels: Any) -> tf.Tensor:
        scaled_logits = self(logits)
        return self.loss_fn(labels, scaled_logits)

    def train_step(self, data: Any) -> Mapping[str, tf.Tensor]:
        logit_batch, y_batch = data
        logit_batch = tf.cast(logit_batch, tf.float64)
        y_batch = tf.cast(y_batch, tf.float64)

        with tf.GradientTape() as tape:
            loss = self._compute_loss(logit_batch, y_batch)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: Any) -> Mapping[str, tf.Tensor]:
        logit_batch, y_batch = data
        logit_batch = tf.cast(logit_batch, tf.float64)
        y_batch = tf.cast(y_batch, tf.float64)
        loss = self._compute_loss(logit_batch, y_batch)
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.function
def categorical_entropy(probs: tf.Tensor, axis: int = -1, eps: float = 1e-8, from_logits: bool = False) -> tf.Tensor:
    x = tf.convert_to_tensor(probs, dtype=tf.float32)
    if from_logits:
        x = tf.nn.softmax(x, axis=axis)
    x = tf.clip_by_value(x, eps, 1.0)
    return -tf.reduce_sum(x * tf.math.log(x), axis=axis)


@tf.function
def binary_entropy(p: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    return -(p * tf.math.log(p) + (1 - p) * tf.math.log(1 - p))


@tf.function
def sampling_reparameterization(distribution_params: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    mean, log_var = distribution_params
    epsilon = tf.random.normal(shape=tf.shape(mean), mean=0.0, stddev=1.0)
    return mean + tf.exp(log_var / 2.0) * epsilon


class _LayerwiseMCModel:
    """Shared layer-by-layer forward pass with MC-Dropout enabled."""

    def __init__(self, model: keras.Model) -> None:
        self.model = model

    @staticmethod
    def _tensor_key(tensor: Any) -> str:
        return getattr(tensor, "name", str(tensor))

    def _layerwise_forward(self, inputs: Any) -> dict[str, tf.Tensor]:
        layer_outputs: dict[str, tf.Tensor] = {}
        model_inputs = inputs if isinstance(inputs, (tuple, list)) else (inputs,)

        for tensor, value in zip(self.model.inputs, model_inputs):
            layer_outputs[self._tensor_key(tensor)] = tf.convert_to_tensor(value)

        for layer in self.model.layers:
            if isinstance(layer, layers.InputLayer):
                continue

            inbound = layer.input
            if isinstance(inbound, (list, tuple)):
                layer_input = [layer_outputs[self._tensor_key(t)] for t in inbound]
            else:
                layer_input = layer_outputs[self._tensor_key(inbound)]

            if isinstance(layer, layers.Dropout):
                layer_output = layer(layer_input, training=True)
            elif isinstance(layer, (layers.BatchNormalization, layers.GaussianNoise)):
                layer_output = layer(layer_input, training=False)
            else:
                layer_output = layer(layer_input)

            outbound = layer.output
            if isinstance(outbound, (list, tuple)):
                for tensor, value in zip(outbound, layer_output):
                    layer_outputs[self._tensor_key(tensor)] = value
            else:
                layer_outputs[self._tensor_key(outbound)] = layer_output

        return layer_outputs


class BinaryMCDropoutPredictor(_LayerwiseMCModel):
    """MC-Dropout predictor for the binary Level 1 model."""

    def __init__(self, model_path: str | Path | None = None, *, custom_objects: Mapping[str, Any] | None = None) -> None:
        resolved_path = resolve_drive_artifact_path(model_path, LEVEL1_MODEL_SPEC)
        model = _load_keras_model(resolved_path, custom_objects=custom_objects)
        super().__init__(model)

    def _single_pass(self, inputs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        outputs = self._layerwise_forward(inputs)
        probabilities = tf.cast(outputs[self._tensor_key(self.model.output)], tf.float32)
        probabilities = tf.reshape(probabilities, (-1,))
        probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0 - 1e-7)
        logits = tf.math.log(probabilities) - tf.math.log1p(-probabilities)
        return probabilities, logits

    def predict_with_uncertainty(self, inputs: Any, T: int = 20) -> RawPredictionOutput:
        batch_probabilities: list[tf.Tensor] = []
        batch_logits: list[tf.Tensor] = []

        for _ in range(T):
            p_t, logits_t = self._single_pass(inputs)
            batch_probabilities.append(p_t)
            batch_logits.append(logits_t)

        stacked_probabilities = tf.stack(batch_probabilities, axis=0)
        stacked_logits = tf.stack(batch_logits, axis=0)

        aleatoric = tf.reduce_mean(binary_entropy(stacked_probabilities), axis=0)
        marginal = tf.reduce_mean(stacked_probabilities, axis=0)
        logits_mean = tf.reduce_mean(stacked_logits, axis=0)
        total = binary_entropy(marginal)
        epistemic = tf.maximum(total - aleatoric, 0.0)

        return RawPredictionOutput(
            probabilities=_squeeze_keep_batch(marginal),
            logits=_squeeze_keep_batch(logits_mean),
            uncertainties=(
                _squeeze_keep_batch(total),
                _squeeze_keep_batch(aleatoric),
                _squeeze_keep_batch(epistemic),
            ),
        )


class MulticlassMCDropoutPredictor(_LayerwiseMCModel):
    """MC-Dropout predictor for the multiclass Level 2 model."""

    def __init__(self, model_path: str | Path | None = None, *, custom_objects: Mapping[str, Any] | None = None) -> None:
        resolved_path = resolve_drive_artifact_path(model_path, LEVEL2_MODEL_SPEC)
        model = _load_keras_model(resolved_path, custom_objects=custom_objects)
        if len(model.layers) < 2:
            raise ValueError(
                "The Level 2 model must expose at least two layers so logits can be read from the penultimate layer."
            )
        super().__init__(model)
        self._logits_tensor = self.model.layers[-2].output
        self._logits_key = self._tensor_key(self._logits_tensor)

    def _single_pass(self, inputs: Any) -> tuple[tf.Tensor, tf.Tensor]:
        outputs = self._layerwise_forward(inputs)
        logits = tf.cast(outputs[self._logits_key], tf.float32)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return probabilities, logits

    def predict_with_uncertainty(self, inputs: Any, T: int = 50) -> RawPredictionOutput:
        batch_probabilities: list[tf.Tensor] = []
        batch_logits: list[tf.Tensor] = []

        for _ in range(T):
            p_t, logits_t = self._single_pass(inputs)
            batch_probabilities.append(p_t)
            batch_logits.append(logits_t)

        stacked_probabilities = tf.stack(batch_probabilities, axis=0)
        stacked_logits = tf.stack(batch_logits, axis=0)

        aleatoric = tf.reduce_mean(categorical_entropy(stacked_probabilities), axis=0)
        marginal = tf.reduce_mean(stacked_probabilities, axis=0)
        logits_mean = tf.reduce_mean(stacked_logits, axis=0)
        total = categorical_entropy(marginal)
        epistemic = tf.maximum(total - aleatoric, 0.0)

        return RawPredictionOutput(
            probabilities=_squeeze_keep_batch(marginal),
            logits=_squeeze_keep_batch(logits_mean),
            uncertainties=(
                _squeeze_keep_batch(total),
                _squeeze_keep_batch(aleatoric),
                _squeeze_keep_batch(epistemic),
            ),
        )


class MultimodalVAEPredictor:
    """MC-Dropout + latent sampling predictor for the Level 3 MVAE."""

    def __init__(
        self,
        encoder_path: str | Path | None = None,
        classifier_path: str | Path | None = None,
        *,
        latent_dim: int = 128,
        custom_objects: Mapping[str, Any] | None = None,
    ) -> None:
        resolved_encoder_path = resolve_drive_artifact_path(encoder_path, LEVEL3_ENCODER_SPEC)
        resolved_classifier_path = resolve_drive_artifact_path(classifier_path, LEVEL3_CLASSIFIER_SPEC)
        self.encoder = _load_keras_model(resolved_encoder_path, custom_objects=custom_objects)
        self.classifier = _load_keras_model(resolved_classifier_path, custom_objects=custom_objects)
        self.latent_dim = latent_dim
        if len(self.classifier.layers) < 2:
            raise ValueError(
                "The Level 3 classifier must expose at least two layers so logits and probabilities can be read."
            )
        self._logits_model = keras.Model(
            inputs=self.classifier.input,
            outputs=self.classifier.layers[-2].output,
        )
        self._probability_model = keras.Model(
            inputs=self.classifier.input,
            outputs=self.classifier.layers[-1].output,
        )

    def _single_pass(self, inputs: Any, S: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        mean, log_var = self.encoder(inputs, training=True)
        latent_dim = int(mean.shape[-1]) if mean.shape[-1] is not None else self.latent_dim
        mean = tf.expand_dims(mean, axis=1)
        log_var = tf.expand_dims(log_var, axis=1)
        mean = tf.repeat(mean, repeats=S, axis=1)
        log_var = tf.repeat(log_var, repeats=S, axis=1)
        z = sampling_reparameterization((mean, log_var))

        batch_size = tf.shape(z)[0]
        z_flat = tf.reshape(z, (-1, latent_dim))
        p, logits = self._probability_model(z_flat, training=True), self._logits_model(z_flat, training=True)
        p = tf.reshape(tf.cast(p, tf.float32), (batch_size, S))
        logits = tf.reshape(tf.cast(logits, tf.float32), (batch_size, S))
        p_mean = tf.reduce_mean(p, axis=1)
        logits_mean = tf.reduce_mean(logits, axis=1)
        u_aleatoric = tf.reduce_mean(binary_entropy(p), axis=1)
        return p_mean, logits_mean, u_aleatoric

    def predict_with_uncertainty(self, inputs: Any, T: int = 20, S: int = 20) -> RawPredictionOutput:
        batch_probabilities: list[tf.Tensor] = []
        batch_logits: list[tf.Tensor] = []
        batch_u_a: list[tf.Tensor] = []

        for _ in range(T):
            p_t, logits_t, u_a_t = self._single_pass(inputs, S)
            batch_probabilities.append(p_t)
            batch_logits.append(logits_t)
            batch_u_a.append(u_a_t)

        stacked_probabilities = tf.stack(batch_probabilities, axis=0)
        stacked_logits = tf.stack(batch_logits, axis=0)
        stacked_u_a = tf.stack(batch_u_a, axis=0)

        aleatoric = tf.reduce_mean(stacked_u_a, axis=0)
        marginal = tf.reduce_mean(stacked_probabilities, axis=0)
        logits_mean = tf.reduce_mean(stacked_logits, axis=0)
        total = binary_entropy(marginal)
        epistemic = tf.maximum(total - aleatoric, 0.0)

        return RawPredictionOutput(
            probabilities=_squeeze_keep_batch(marginal),
            logits=_squeeze_keep_batch(logits_mean),
            uncertainties=(
                _squeeze_keep_batch(total),
                _squeeze_keep_batch(aleatoric),
                _squeeze_keep_batch(epistemic),
            ),
        )


class BinaryLogisticCalibrator:
    """Binary calibration wrapper based on sklearn LogisticRegression."""

    def __init__(self, estimator: LogisticRegression) -> None:
        self.estimator = estimator

    @classmethod
    def from_path(cls, path: str | Path) -> "BinaryLogisticCalibrator":
        estimator = joblib.load(_to_path(path))
        if not isinstance(estimator, LogisticRegression):
            raise TypeError("The loaded object is not a LogisticRegression instance.")
        return cls(estimator)

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        labels: np.ndarray,
        *,
        C: float = 1e10,
        solver: str = "lbfgs",
    ) -> "BinaryLogisticCalibrator":
        estimator = LogisticRegression(solver=solver, C=C)
        estimator.fit(np.asarray(logits).reshape(-1, 1), np.asarray(labels).reshape(-1))
        return cls(estimator)

    def __call__(self, probabilities: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if logits is None:
            raise ValueError("BinaryLogisticCalibrator requires logits.")
        logits_2d = np.asarray(logits).reshape(-1, 1)
        calibrated = self.estimator.predict_proba(logits_2d)[:, 1]
        return calibrated.reshape(-1, 1)


class MatrixSoftmaxCalibrator:
    """Apply a learned MatrixScaler followed by softmax."""

    def __init__(self, scaler: MatrixScaler) -> None:
        self.scaler = scaler

    @classmethod
    def from_path(cls, path: str | Path, *, custom_objects: Mapping[str, Any] | None = None) -> "MatrixSoftmaxCalibrator":
        custom_objects = dict(custom_objects or {})
        custom_objects.setdefault("MatrixScaler", MatrixScaler)
        scaler = keras.models.load_model(_to_path(path), custom_objects=custom_objects, compile=False)
        if not isinstance(scaler, MatrixScaler):
            raise TypeError("The loaded object is not a MatrixScaler instance.")
        return cls(scaler)

    def __call__(self, probabilities: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        if logits is None:
            raise ValueError("MatrixSoftmaxCalibrator requires logits.")
        logits_tensor = tf.convert_to_tensor(np.asarray(logits), dtype=tf.float32)
        scaled = tf.nn.softmax(self.scaler(logits_tensor), axis=-1)
        return scaled.numpy()


@dataclass(frozen=True)
class HierarchyArtifacts:
    """Convenience container for the three level predictors and calibrators."""

    level1: BinaryMCDropoutPredictor
    level2: MulticlassMCDropoutPredictor
    level3: MultimodalVAEPredictor
    level1_calibrator: BinaryLogisticCalibrator | None = None
    level2_calibrator: MatrixSoftmaxCalibrator | None = None
    level3_calibrator: BinaryLogisticCalibrator | None = None

    def as_inference_kwargs(self) -> dict[str, Any]:
        return {
            "level1_predictor": self.level1,
            "level2_predictor": self.level2,
            "level3_predictor": self.level3,
            "level1_calibrator": self.level1_calibrator,
            "level2_calibrator": self.level2_calibrator,
            "level3_calibrator": self.level3_calibrator,
        }


def build_hierarchical_inference(
    *,
    level1_model_path: str | Path | None = None,
    level2_model_path: str | Path | None = None,
    level3_encoder_path: str | Path | None = None,
    level3_classifier_path: str | Path | None = None,
    level1_calibrator_path: str | Path | None = None,
    level2_calibrator_path: str | Path | None = None,
    level3_calibrator_path: str | Path | None = None,
    feature_extractor: Any | None = None,
    thresholds: Any | None = None,
    latent_dim: int = 128,
    custom_objects: Mapping[str, Any] | None = None,
) -> Any:
    """Build a ready-to-use hierarchical inference pipeline from artifact paths."""

    from .inference import HierarchicalChitinaseInference

    level1_predictor = BinaryMCDropoutPredictor(level1_model_path, custom_objects=custom_objects)
    level2_predictor = MulticlassMCDropoutPredictor(level2_model_path, custom_objects=custom_objects)
    level3_predictor = MultimodalVAEPredictor(
        level3_encoder_path,
        level3_classifier_path,
        latent_dim=latent_dim,
        custom_objects=custom_objects,
    )

    level1_calibrator = (
        BinaryLogisticCalibrator.from_path(level1_calibrator_path)
        if level1_calibrator_path is not None
        else None
    )
    level2_calibrator = (
        MatrixSoftmaxCalibrator.from_path(level2_calibrator_path, custom_objects=custom_objects)
        if level2_calibrator_path is not None
        else None
    )
    level3_calibrator = (
        BinaryLogisticCalibrator.from_path(level3_calibrator_path)
        if level3_calibrator_path is not None
        else None
    )

    return HierarchicalChitinaseInference(
        feature_extractor=feature_extractor,
        level1_predictor=level1_predictor,
        level2_predictor=level2_predictor,
        level3_predictor=level3_predictor,
        level1_calibrator=level1_calibrator,
        level2_calibrator=level2_calibrator,
        level3_calibrator=level3_calibrator,
        thresholds=thresholds,
    )


__all__ = [
    "BinaryLogisticCalibrator",
    "BinaryMCDropoutPredictor",
    "HierarchyArtifacts",
    "MatrixScaler",
    "MatrixSoftmaxCalibrator",
    "MulticlassMCDropoutPredictor",
    "MultimodalVAEPredictor",
    "binary_entropy",
    "build_hierarchical_inference",
    "categorical_entropy",
    "sampling_reparameterization",
]
