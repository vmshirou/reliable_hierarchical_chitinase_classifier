"""Explainability utilities for the multimodal chitinase models.

The notebooks use SHAP on the concatenated multimodal model. This module keeps
that logic but removes ad-hoc plotting so it can be used programmatically from
scripts, APIs, or notebooks alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from .artifacts import load_protbert_components as _load_protbert_components
from .features import (
    DescriptorFeatureExtractor,
    clean_protein_sequence,
    format_sequence_for_protbert,
)

try:  # pragma: no cover - optional dependency path
    import shap
except Exception as exc:  # pragma: no cover - optional dependency path
    shap = None  # type: ignore[assignment]
    _SHAP_IMPORT_ERROR = exc
else:  # pragma: no cover - optional dependency path
    _SHAP_IMPORT_ERROR = None


def _require_shap() -> Any:
    if shap is None:  # pragma: no cover - import-time failure path
        raise ImportError(
            "shap is required for explainability utilities."
        ) from _SHAP_IMPORT_ERROR
    return shap


def _as_numpy(array: Any) -> np.ndarray:
    return np.asarray(array)


def _to_float_tensor(value: Any) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float32)


def _squeeze_batch(array: Any) -> np.ndarray:
    return np.squeeze(_as_numpy(array))


def _token_shap_to_vector(values: Any) -> np.ndarray:
    """Reduce token SHAP outputs to a per-token attribution vector."""

    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr.sum(axis=-1)
    return np.asarray(arr).reshape(-1)


def _descriptor_shap_to_vector(values: Any) -> np.ndarray:
    """Reduce descriptor SHAP outputs to a 1D attribution vector."""

    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    return np.asarray(arr).reshape(-1)


def _aligned_token_and_score_vectors(
    sequence: str,
    shap_token_values: Any,
    tokenizer: Any | None = None,
    *,
    max_length: int = 512,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Align ProtBERT token names with their SHAP scores.

    SHAP values are produced for the full ProtBERT input, including special
    tokens and padding. The public-facing attribution tables should only show
    biological residues, so this helper drops special tokens while keeping the
    scores aligned with the surviving positions.
    """

    token_scores = _token_shap_to_vector(shap_token_values)[:max_length]
    if tokenizer is None:
        tokens = tokenize_protein_sequence(sequence, tokenizer=None)
        n = min(len(tokens), len(token_scores))
        return tuple(tokens[:n]), np.asarray(token_scores[:n], dtype=np.float32)

    cleaned = clean_protein_sequence(sequence)
    formatted = format_sequence_for_protbert(cleaned)
    encoded = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    token_ids = encoded["input_ids"]
    if isinstance(token_ids, np.ndarray):
        if token_ids.ndim > 1:
            token_ids = token_ids[0]
        token_ids = token_ids.tolist()
    elif isinstance(token_ids, (list, tuple)) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple, np.ndarray)):
        token_ids = list(token_ids[0])
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    special_tokens = set(getattr(tokenizer, "all_special_tokens", ()))
    paired = [
        (token, float(score))
        for token, score in zip(tokens, token_scores)
        if token not in special_tokens
    ]
    if not paired:
        return tuple(), np.zeros((0,), dtype=np.float32)
    token_names, scores = zip(*paired)
    return tuple(token_names), np.asarray(scores, dtype=np.float32)


def load_protbert_components(
    model_name: str = "Rostlab/prot_bert",
    *,
    cache_dir: str | Path | None = None,
) -> tuple[Any, tf.keras.Model]:
    """Load ProtBERT tokenizer and TensorFlow model."""

    return _load_protbert_components(model_name=model_name, cache_dir=cache_dir)


def _protbert_embedding_layer(protbert_model: tf.keras.Model) -> tf.keras.layers.Layer:
    backbone = getattr(protbert_model, "bert", protbert_model)
    if not hasattr(backbone, "embeddings"):
        raise AttributeError("The provided model does not expose a BERT embeddings layer.")
    return backbone.embeddings


def protbert_pad_embedding_vector(
    protbert_model: tf.keras.Model,
    tokenizer: Any,
) -> np.ndarray:
    """Return the PAD embedding vector used by the SHAP wrapper."""

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("The tokenizer does not expose a pad token id.")
    embedding_layer = _protbert_embedding_layer(protbert_model)
    pad_embedding = embedding_layer(tf.constant([[pad_token_id]]))[0, 0, :]
    return pad_embedding.numpy().astype(np.float32)


def sequence_to_protbert_inputs(
    sequence: str,
    tokenizer: Any,
    protbert_model: tf.keras.Model,
    *,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raw sequence into ProtBERT token embeddings and attention mask."""

    formatted = format_sequence_for_protbert(clean_protein_sequence(sequence))
    encoded = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )
    embedding_layer = _protbert_embedding_layer(protbert_model)
    token_embeddings = embedding_layer(encoded["input_ids"]).numpy().astype(np.float32)
    attention_mask = encoded["attention_mask"].numpy().astype(np.int32)
    return token_embeddings, attention_mask


def sequences_to_shap_background(
    sequences: Sequence[str],
    tokenizer: Any,
    protbert_model: tf.keras.Model,
    *,
    max_length: int = 512,
    descriptor_extractor: DescriptorFeatureExtractor | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    """Build SHAP background tensors from raw sequences."""

    cleaned = [clean_protein_sequence(seq) for seq in sequences]
    if not cleaned:
        raise ValueError("At least one background sequence is required.")

    descriptor_extractor = descriptor_extractor or DescriptorFeatureExtractor()
    descriptors = descriptor_extractor.transform_many(cleaned)
    embeddings = []
    for seq in cleaned:
        token_embeddings, _ = sequence_to_protbert_inputs(
            seq,
            tokenizer,
            protbert_model,
            max_length=max_length,
        )
        embeddings.append(token_embeddings)

    background_embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    background_descriptors = np.asarray(descriptors, dtype=np.float32)
    feature_names = descriptor_extractor.feature_names
    pad_vector = protbert_pad_embedding_vector(protbert_model, tokenizer)
    return background_embeddings, background_descriptors, feature_names, pad_vector


@dataclass(frozen=True)
class ShapExplanationBundle:
    """Programmatic container for SHAP outputs."""

    token_attributions: np.ndarray
    descriptor_attributions: np.ndarray
    tokens: tuple[str, ...]
    feature_names: tuple[str, ...]
    expected_value: Any


class ShapMultimodalWrapper(tf.keras.Model):
    """Keras wrapper matching the notebook SHAP setup."""

    def __init__(
        self,
        protbert_model: tf.keras.Model,
        encoder_model: tf.keras.Model,
        classifier_model: tf.keras.Model,
        pad_embedding_vector: np.ndarray | tf.Tensor,
        standardize_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.protbert_model = protbert_model
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model
        self.pad_embedding_vector = _to_float_tensor(pad_embedding_vector)
        self.standardize_fn = standardize_fn or (lambda x: x)

    @tf.function
    def call(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        inputs_embeds = inputs[0]
        descriptors = inputs[1]
        pad_vector = tf.cast(self.pad_embedding_vector, tf.float32)

        delta = tf.reduce_sum(tf.abs(inputs_embeds - pad_vector), axis=-1)
        attention_mask = tf.cast(delta > 1e-6, tf.int32)

        bert_output = self.protbert_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            training=False,
        )

        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        standardized = tf.convert_to_tensor(self.standardize_fn(cls_embedding))
        latent_mean, _ = self.encoder_model([standardized, descriptors], training=False)
        final_output = self.classifier_model(latent_mean, training=False)
        return final_output


def build_shap_model(
    protbert_model: tf.keras.Model,
    encoder_model: tf.keras.Model,
    classifier_model: tf.keras.Model,
    pad_embedding_vector: np.ndarray | tf.Tensor,
    standardize_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    *,
    embedding_shape: tuple[int | None, int] | None = None,
    descriptor_dim: int | None = None,
) -> tf.keras.Model:
    """Build the functional model used by SHAP GradientExplainer."""

    if embedding_shape is None or descriptor_dim is None:
        raise ValueError(
            "embedding_shape and descriptor_dim are required to build the SHAP model."
        )

    pad_vector = _as_numpy(pad_embedding_vector)
    inferred_embedding_shape = embedding_shape or (None, int(pad_vector.shape[-1]))
    inferred_descriptor_dim = descriptor_dim

    inputs_embeds = tf.keras.Input(
        shape=inferred_embedding_shape,
        dtype=tf.float32,
        name="inputs_embeds",
    )
    descriptors = tf.keras.Input(
        shape=(inferred_descriptor_dim,),
        dtype=tf.float32,
        name="descriptors",
    )
    wrapper = ShapMultimodalWrapper(
        protbert_model=protbert_model,
        encoder_model=encoder_model,
        classifier_model=classifier_model,
        pad_embedding_vector=pad_embedding_vector,
        standardize_fn=standardize_fn,
    )
    outputs = wrapper([inputs_embeds, descriptors])
    return tf.keras.Model(
        inputs=[inputs_embeds, descriptors],
        outputs=outputs,
        name="ShapExplainerModel",
    )


def create_gradient_explainer(
    shap_model: tf.keras.Model,
    background_embeddings: np.ndarray,
    background_descriptors: np.ndarray,
    *,
    batch_size: int = 4,
) -> Any:
    """Create a SHAP GradientExplainer for the multimodal model."""

    shap_module = _require_shap()
    return shap_module.GradientExplainer(
        shap_model,
        data=[background_embeddings, background_descriptors],
        batch_size=batch_size,
    )


def expected_value_from_background(
    shap_model: tf.keras.Model,
    background_embeddings: np.ndarray,
    background_descriptors: np.ndarray,
) -> np.ndarray:
    """Compute the expected output value over the SHAP background set."""

    predictions = shap_model.predict([background_embeddings, background_descriptors], verbose=0)
    return np.mean(np.asarray(predictions), axis=0)


def explain_sample(
    explainer: Any,
    sample_embeddings: np.ndarray,
    sample_descriptors: np.ndarray,
) -> list[np.ndarray]:
    """Generate SHAP values for a single multimodal sample."""

    return explainer.shap_values([sample_embeddings, sample_descriptors])


def tokenize_protein_sequence(sequence: str, tokenizer: Any | None = None) -> tuple[str, ...]:
    """Return the tokens used by ProtBERT-style tokenization."""

    cleaned = clean_protein_sequence(sequence)
    if tokenizer is None:
        return tuple(format_sequence_for_protbert(cleaned).split())

    token_ids = tokenizer(format_sequence_for_protbert(cleaned), truncation=True, max_length=512)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tuple(tok for tok in tokens if tok not in getattr(tokenizer, "all_special_tokens", ()))


def token_attributions_from_shap(
    shap_token_values: np.ndarray,
    sequence: str,
    tokenizer: Any | None = None,
    *,
    max_length: int = 512,
) -> pd.DataFrame:
    """Convert token-level SHAP values into a tidy DataFrame."""

    tokens, token_scores = _aligned_token_and_score_vectors(
        sequence,
        shap_token_values,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    n = min(len(tokens), len(token_scores))
    return pd.DataFrame(
        {
            "token": list(tokens[:n]),
            "shap_value": token_scores[:n],
            "abs_shap": np.abs(token_scores[:n]),
        }
    )


def descriptor_attributions_frame(
    shap_descriptor_values: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return per-descriptor SHAP values as a DataFrame."""

    values = _descriptor_shap_to_vector(shap_descriptor_values)
    n = min(len(values), len(feature_names))
    return pd.DataFrame(
        {
            "feature": list(feature_names)[:n],
            "shap_value": values[:n],
            "abs_shap": np.abs(values[:n]),
        }
    )


def top_descriptor_attributions(
    shap_descriptor_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the top descriptors ranked by mean absolute SHAP value."""

    frame = descriptor_attributions_frame(shap_descriptor_values, feature_names)
    return frame.sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)


def top_token_attributions(
    shap_token_values: np.ndarray,
    sequence: str,
    tokenizer: Any | None = None,
    *,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the top sequence tokens ranked by absolute SHAP value."""

    frame = token_attributions_from_shap(shap_token_values, sequence, tokenizer=tokenizer)
    return frame.sort_values("abs_shap", ascending=False).head(top_k).reset_index(drop=True)


class MultimodalShapExplainer:
    """High-level orchestration for multimodal SHAP analysis."""

    def __init__(
        self,
        shap_model: tf.keras.Model,
        explainer: Any,
        *,
        tokenizer: Any | None = None,
        feature_names: Sequence[str] | None = None,
        background_embeddings: np.ndarray | None = None,
        background_descriptors: np.ndarray | None = None,
    ) -> None:
        self.shap_model = shap_model
        self.explainer = explainer
        self.tokenizer = tokenizer
        self.feature_names = tuple(feature_names) if feature_names is not None else None
        self.background_embeddings = background_embeddings
        self.background_descriptors = background_descriptors

    @classmethod
    def from_models(
        cls,
        protbert_model: tf.keras.Model,
        encoder_model: tf.keras.Model,
        classifier_model: tf.keras.Model,
        pad_embedding_vector: np.ndarray | tf.Tensor,
        background_embeddings: np.ndarray,
        background_descriptors: np.ndarray,
        *,
        standardize_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        tokenizer: Any | None = None,
        feature_names: Sequence[str] | None = None,
        batch_size: int = 4,
    ) -> "MultimodalShapExplainer":
        background_embeddings = np.asarray(background_embeddings)
        background_descriptors = np.asarray(background_descriptors)
        if background_embeddings.ndim == 2:
            background_embeddings = background_embeddings[np.newaxis, ...]
        if background_descriptors.ndim == 1:
            background_descriptors = background_descriptors[np.newaxis, ...]
        shap_model = build_shap_model(
            protbert_model=protbert_model,
            encoder_model=encoder_model,
            classifier_model=classifier_model,
            pad_embedding_vector=pad_embedding_vector,
            standardize_fn=standardize_fn,
            embedding_shape=tuple(background_embeddings.shape[1:]) if background_embeddings.ndim >= 2 else None,
            descriptor_dim=int(background_descriptors.shape[1]) if background_descriptors.ndim >= 2 else None,
        )
        explainer = create_gradient_explainer(
            shap_model,
            background_embeddings=background_embeddings,
            background_descriptors=background_descriptors,
            batch_size=batch_size,
        )
        return cls(
            shap_model,
            explainer,
            tokenizer=tokenizer,
            feature_names=feature_names,
            background_embeddings=background_embeddings,
            background_descriptors=background_descriptors,
        )

    def explain(
        self,
        sequence: str,
        sample_embeddings: np.ndarray,
        sample_descriptors: np.ndarray,
        *,
        max_length: int = 512,
    ) -> ShapExplanationBundle:
        """Explain a single sample and return a tidy attribution bundle."""

        shap_values = explain_sample(self.explainer, sample_embeddings, sample_descriptors)
        tokens, token_values = _aligned_token_and_score_vectors(
            sequence,
            shap_values[0],
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        descriptor_values = _descriptor_shap_to_vector(shap_values[1])
        feature_names = self.feature_names or tuple(f"feature_{i}" for i in range(descriptor_values.shape[0]))
        return ShapExplanationBundle(
            token_attributions=token_values,
            descriptor_attributions=descriptor_values,
            tokens=tokens,
            feature_names=feature_names,
            expected_value=expected_value_from_background(
                self.shap_model,
                self.background_embeddings if self.background_embeddings is not None else sample_embeddings,
                self.background_descriptors if self.background_descriptors is not None else sample_descriptors,
            ),
        )

    def token_frame(self, sequence: str, shap_token_values: np.ndarray) -> pd.DataFrame:
        """Return a token attribution table for a sequence."""

        return token_attributions_from_shap(
            shap_token_values,
            sequence,
            tokenizer=self.tokenizer,
        )

    def descriptor_frame(self, shap_descriptor_values: np.ndarray) -> pd.DataFrame:
        """Return a descriptor attribution table."""

        if self.feature_names is None:
            raise ValueError("feature_names are required to build descriptor frames.")
        return descriptor_attributions_frame(
            shap_descriptor_values,
            self.feature_names,
        )


__all__ = [
    "MultimodalShapExplainer",
    "ShapExplanationBundle",
    "ShapMultimodalWrapper",
    "build_shap_model",
    "create_gradient_explainer",
    "descriptor_attributions_frame",
    "expected_value_from_background",
    "explain_sample",
    "load_protbert_components",
    "protbert_pad_embedding_vector",
    "sequence_to_protbert_inputs",
    "sequences_to_shap_background",
    "token_attributions_from_shap",
    "tokenize_protein_sequence",
    "top_descriptor_attributions",
    "top_token_attributions",
]
