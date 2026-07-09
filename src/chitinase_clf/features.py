"""Feature extraction utilities for chitinase classification.

This module converts raw amino-acid sequences into the hybrid representation
used by the research notebooks:

* Physicochemical descriptors via ``propy3``
* ProtBERT embeddings via ``transformers``

The implementation is intentionally notebook-compatible, but reorganized into
reusable classes and functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import re

import numpy as np
import pandas as pd

from .artifacts import load_protbert_components

AMINO_ACIDS_STANDARD: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
_PROTBERT_RARE_TOKENS = {"U", "Z", "O", "B"}

def clean_protein_sequence(sequence: str) -> str:
    """Normalize a raw protein sequence.

    The function removes whitespace, uppercases the text, and keeps the
    sequence in a compact form suitable for descriptor and embedding extraction.
    """

    if sequence is None:
        raise ValueError("sequence cannot be None")

    cleaned = re.sub(r"\s+", "", str(sequence)).upper().strip()
    if not cleaned:
        raise ValueError("sequence cannot be empty")
    return cleaned


def find_non_standard_amino_acids(sequence: str) -> set[str]:
    """Return the set of non-standard amino acids present in ``sequence``."""

    cleaned = clean_protein_sequence(sequence)
    return set(cleaned) - AMINO_ACIDS_STANDARD


def replace_non_standard_with_most_common(
    sequence: str,
    non_standard_amino_acids: set[str] | None = None,
    fallback_residue: str = "A",
) -> str:
    """Replace non-standard residues with the most common standard residue.

    This mirrors the original notebook logic and keeps the input compatible with
    ``propy3`` descriptor calculation.
    """

    cleaned = clean_protein_sequence(sequence)
    non_standard = (
        non_standard_amino_acids
        if non_standard_amino_acids is not None
        else find_non_standard_amino_acids(cleaned)
    )
    if not non_standard:
        return cleaned

    standard_counts: dict[str, int] = {}
    for residue in cleaned:
        if residue in AMINO_ACIDS_STANDARD:
            standard_counts[residue] = standard_counts.get(residue, 0) + 1

    if standard_counts:
        replacement = max(standard_counts.items(), key=lambda item: item[1])[0]
    else:
        replacement = fallback_residue

    if replacement not in AMINO_ACIDS_STANDARD:
        raise ValueError(
            "fallback_residue must be a standard amino acid if the sequence "
            "contains no standard residues"
        )

    for residue in non_standard:
        cleaned = cleaned.replace(residue, replacement)
    return cleaned


def format_sequence_for_protbert(sequence: str) -> str:
    """Prepare a sequence for ProtBERT tokenization.

    ProtBERT expects amino acids separated by spaces and uses ``X`` for rare or
    unknown residues. The function also removes whitespace and keeps the input
    uppercase.
    """

    cleaned = clean_protein_sequence(sequence)
    normalized_chars: list[str] = []
    for residue in cleaned:
        if residue in AMINO_ACIDS_STANDARD:
            normalized_chars.append(residue)
        elif residue in _PROTBERT_RARE_TOKENS:
            normalized_chars.append("X")
        else:
            normalized_chars.append("X")
    return " ".join(normalized_chars)


def _import_pypro() -> Any:
    """Import the ``propy3`` PyPro module lazily."""

    try:
        import propy.PyPro as PyPro  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time failure path
        raise ImportError(
            "propy3 is required for descriptor extraction. Install it with "
            "`pip install propy3`."
        ) from exc
    return PyPro


def calculate_protein_descriptors(sequence: str) -> dict[str, float]:
    """Compute the descriptor dictionary used by the original notebooks."""

    PyPro = _import_pypro()
    normalized = replace_non_standard_with_most_common(sequence)
    protein = PyPro.GetProDes(normalized)

    descriptors = {
        "AAC": protein.GetAAComp(),
        "MBauto": protein.GetMoreauBrotoAuto(),
        "Moranauto": protein.GetMoranAuto(),
        "Gearyauto": protein.GetGearyAuto(),
        "CTD": protein.GetCTD(),
    }

    combined_descriptors: dict[str, float] = {}
    for group_name in ("AAC", "MBauto", "Moranauto", "Gearyauto", "CTD"):
        group = descriptors[group_name]
        for key, value in group.items():
            combined_descriptors[str(key)] = float(value)
    return combined_descriptors


def _align_descriptor_rows(
    rows: list[dict[str, float]],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(feature_names)}
    for row_idx, row in enumerate(rows):
        for name, value in row.items():
            col_idx = index.get(name)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(value)
    return matrix


@dataclass(frozen=True)
class HybridFeatureBundle:
    """Container for the hybrid representation of a single sequence."""

    sequence: str
    descriptors: np.ndarray
    embeddings: np.ndarray
    descriptor_names: tuple[str, ...]

    def as_model_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return batched model inputs in the notebook order: embeddings, descriptors."""

        return (
            np.expand_dims(self.embeddings, axis=0),
            np.expand_dims(self.descriptors, axis=0),
        )


@dataclass(frozen=True)
class HybridFeatureBatch:
    """Container for a batch of hybrid features."""

    sequences: tuple[str, ...]
    descriptors: np.ndarray
    embeddings: np.ndarray
    descriptor_names: tuple[str, ...]

    def as_model_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Return batched model inputs in the notebook order: embeddings, descriptors."""

        return self.embeddings, self.descriptors


class DescriptorFeatureExtractor:
    """Compute propy3 descriptors for protein sequences."""

    def __init__(
        self,
        normalizer: Callable[[str], str] | None = None,
    ) -> None:
        self._normalizer = normalizer or replace_non_standard_with_most_common
        self._feature_names: tuple[str, ...] | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names inferred from the first extracted sequence."""

        if self._feature_names is None:
            raise RuntimeError(
                "feature_names are not available yet. Call transform_one() or "
                "transform_many() first."
            )
        return self._feature_names

    def _feature_names_or_empty(self) -> tuple[str, ...]:
        """Return known feature names, or an empty tuple when nothing was fitted yet."""

        return self._feature_names or tuple()

    def transform_one(self, sequence: str) -> np.ndarray:
        """Return a single descriptor vector."""

        descriptor_dict = calculate_protein_descriptors(self._normalizer(sequence))
        if self._feature_names is None:
            self._feature_names = tuple(descriptor_dict.keys())
        return np.array(
            [float(descriptor_dict.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float32,
        )

    def transform_many(self, sequences: Sequence[str]) -> np.ndarray:
        """Return a descriptor matrix for multiple sequences."""

        rows = [calculate_protein_descriptors(self._normalizer(seq)) for seq in sequences]
        if not rows:
            return np.zeros((0, len(self._feature_names_or_empty())), dtype=np.float32)
        if self._feature_names is None:
            self._feature_names = tuple(rows[0].keys())
        return _align_descriptor_rows(rows, self.feature_names)

    def transform_dataframe(
        self,
        frame: pd.DataFrame,
        sequence_column: str,
        *,
        drop_sequence_column: bool = False,
    ) -> pd.DataFrame:
        """Append descriptor columns to a DataFrame."""

        if drop_sequence_column:
            base = frame.drop(columns=[sequence_column])
        else:
            base = frame.copy()
        if frame.empty:
            if self._feature_names is None:
                return base
            descriptor_df = pd.DataFrame(
                np.zeros((0, len(self.feature_names)), dtype=np.float32),
                columns=list(self.feature_names),
                index=frame.index,
            )
            return pd.concat([base, descriptor_df], axis=1)

        descriptor_matrix = self.transform_many(frame[sequence_column].tolist())
        descriptor_df = pd.DataFrame(
            descriptor_matrix,
            columns=list(self.feature_names),
            index=frame.index,
        )
        return pd.concat([base, descriptor_df], axis=1)


class ProtBertEmbeddingExtractor:
    """Generate ProtBERT CLS embeddings for protein sequences."""

    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        batch_size: int = 96,
        max_length: int = 512,
        cache_dir: str | None = None,
        load_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self._tokenizer = None
        self._model = None
        if load_model:
            self.load()

    @property
    def is_loaded(self) -> bool:
        """Whether the tokenizer and model have already been loaded."""

        return self._tokenizer is not None and self._model is not None

    def load(self) -> "ProtBertEmbeddingExtractor":
        """Load tokenizer and TFBertModel lazily."""

        if self.is_loaded:
            return self

        self._tokenizer, self._model = load_protbert_components(
            self.model_name,
            cache_dir=self.cache_dir,
        )
        return self

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def _prepare_batch(self, sequences: Sequence[str]) -> list[str]:
        return [format_sequence_for_protbert(seq) for seq in sequences]

    def transform_one(self, sequence: str) -> np.ndarray:
        """Return a single ProtBERT CLS embedding."""

        batch = self.transform_many([sequence])
        return batch[0]

    def transform_many(self, sequences: Sequence[str]) -> np.ndarray:
        """Return CLS embeddings for multiple sequences."""

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        if len(sequences) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            end = min(start + self.batch_size, len(sequences))
            prepared_batch = self._prepare_batch(sequences[start:end])
            encoded = self._tokenizer(
                prepared_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="tf",
            )
            outputs = self._model(encoded, training=False)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)
            embeddings.append(cls_embeddings)

        return np.vstack(embeddings)


class HybridFeatureExtractor:
    """Compose descriptor and ProtBERT embedding extraction."""

    def __init__(
        self,
        descriptor_extractor: DescriptorFeatureExtractor | None = None,
        embedding_extractor: ProtBertEmbeddingExtractor | None = None,
    ) -> None:
        self.descriptor_extractor = descriptor_extractor or DescriptorFeatureExtractor()
        self.embedding_extractor = embedding_extractor or ProtBertEmbeddingExtractor()

    def transform_one(self, sequence: str) -> HybridFeatureBundle:
        """Extract both representations for a single sequence."""

        cleaned = clean_protein_sequence(sequence)
        descriptors = self.descriptor_extractor.transform_one(cleaned)
        embeddings = self.embedding_extractor.transform_one(cleaned)
        return HybridFeatureBundle(
            sequence=cleaned,
            descriptors=descriptors,
            embeddings=embeddings,
            descriptor_names=self.descriptor_extractor.feature_names,
        )

    def transform_many(self, sequences: Sequence[str]) -> HybridFeatureBatch:
        """Extract both representations for multiple sequences."""

        cleaned_sequences = tuple(clean_protein_sequence(seq) for seq in sequences)
        if not cleaned_sequences:
            descriptor_names = (
                self.descriptor_extractor._feature_names
                if self.descriptor_extractor._feature_names is not None
                else tuple()
            )
            return HybridFeatureBatch(
                sequences=tuple(),
                descriptors=np.zeros((0, len(descriptor_names)), dtype=np.float32),
                embeddings=np.zeros((0, 0), dtype=np.float32),
                descriptor_names=descriptor_names,
            )
        descriptors = self.descriptor_extractor.transform_many(cleaned_sequences)
        embeddings = self.embedding_extractor.transform_many(cleaned_sequences)
        return HybridFeatureBatch(
            sequences=cleaned_sequences,
            descriptors=descriptors,
            embeddings=embeddings,
            descriptor_names=self.descriptor_extractor.feature_names,
        )


__all__ = [
    "AMINO_ACIDS_STANDARD",
    "DescriptorFeatureExtractor",
    "HybridFeatureBatch",
    "HybridFeatureBundle",
    "HybridFeatureExtractor",
    "ProtBertEmbeddingExtractor",
    "calculate_protein_descriptors",
    "clean_protein_sequence",
    "find_non_standard_amino_acids",
    "format_sequence_for_protbert",
    "replace_non_standard_with_most_common",
]
