"""Command-line interface for the chitinase classifier."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from .hierarchy import HierarchyThresholds, build_hierarchical_inference
from .artifacts import (
    LEVEL3_CLASSIFIER_SPEC,
    LEVEL3_ENCODER_SPEC,
    resolve_drive_artifact_path,
)
from .explainability import (
    MultimodalShapExplainer,
    load_protbert_components,
    protbert_pad_embedding_vector,
    sequence_to_protbert_inputs,
    sequences_to_shap_background,
)
from .features import DescriptorFeatureExtractor, clean_protein_sequence


def _jsonable(value: Any) -> Any:
    """Convert nested dataclasses and numpy values into JSON-serializable data."""

    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _read_sequence(args: argparse.Namespace) -> str:
    if args.sequence:
        return args.sequence.strip()
    if args.sequence_file:
        return Path(args.sequence_file).read_text(encoding="utf-8").strip()
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    raise SystemExit("Provide --sequence, --sequence-file, or pipe a sequence through stdin.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chitinase-clf",
        description="Run hierarchical chitinase inference from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run hierarchical inference for one sequence.")
    infer.add_argument("--sequence", help="Raw amino-acid sequence to classify.")
    infer.add_argument("--sequence-file", help="Path to a text file containing one sequence.")
    infer.add_argument("--level1-model", help="Path to the Level 1 Keras model, or omit to use the default cache/download flow.")
    infer.add_argument("--level2-model", help="Path to the Level 2 Keras model, or omit to use the default cache/download flow.")
    infer.add_argument("--level3-encoder", help="Path to the Level 3 encoder model, or omit to use the default cache/download flow.")
    infer.add_argument("--level3-classifier", help="Path to the Level 3 classifier model, or omit to use the default cache/download flow.")
    infer.add_argument("--level1-calibrator", help="Optional path to the Level 1 calibrator.")
    infer.add_argument("--level2-calibrator", help="Optional path to the Level 2 calibrator.")
    infer.add_argument("--level3-calibrator", help="Optional path to the Level 3 calibrator.")
    infer.add_argument("--mc-passes", type=int, default=20, help="MC-Dropout passes per level.")
    infer.add_argument("--latent-samples", type=int, default=20, help="Latent samples for the MVAE level.")
    infer.add_argument("--l1-threshold", type=float, default=0.50, help="Binary threshold for Level 1 routing.")
    infer.add_argument("--json", action="store_true", help="Print the result as JSON.")

    explain = subparsers.add_parser("explain", help="Generate SHAP attributions for one sequence.")
    explain.add_argument("--sequence", help="Raw amino-acid sequence to explain.")
    explain.add_argument("--sequence-file", help="Path to a text file containing one sequence.")
    explain.add_argument("--protbert-model", default="Rostlab/prot_bert", help="ProtBERT model name or path.")
    explain.add_argument("--cache-dir", help="Optional Hugging Face cache directory.")
    explain.add_argument("--encoder-model", help="Path to the multimodal encoder model, or omit to use the default cache/download flow.")
    explain.add_argument("--classifier-model", help="Path to the multimodal classifier model, or omit to use the default cache/download flow.")
    explain.add_argument(
        "--background-sequences-file",
        help="Text file with one background sequence per line. Preferred option.",
    )
    explain.add_argument(
        "--background-embeddings",
        help="Optional .npy file with precomputed background token embeddings.",
    )
    explain.add_argument(
        "--background-descriptors",
        help="Optional .npy file with precomputed background descriptors.",
    )
    explain.add_argument(
        "--feature-names-file",
        help="Optional text/JSON file with descriptor names when using precomputed arrays.",
    )
    explain.add_argument("--top-k", type=int, default=20, help="Number of top attributions to report.")
    explain.add_argument("--max-length", type=int, default=512, help="Maximum ProtBERT token length.")
    explain.add_argument("--json", action="store_true", help="Print the explanation as JSON.")

    return parser


def _load_numpy_file(path: str) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def _load_feature_names(path: str | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    file_path = Path(path)
    if file_path.suffix.lower() == ".json":
        return tuple(json.loads(file_path.read_text(encoding="utf-8")))
    return tuple(
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _sequence_for_explain(args: argparse.Namespace) -> str:
    return clean_protein_sequence(_read_sequence(args))


def _run_infer(args: argparse.Namespace) -> int:
    sequence = _read_sequence(args)
    pipeline = build_hierarchical_inference(
        level1_model_path=args.level1_model,
        level2_model_path=args.level2_model,
        level3_encoder_path=args.level3_encoder,
        level3_classifier_path=args.level3_classifier,
        level1_calibrator_path=args.level1_calibrator,
        level2_calibrator_path=args.level2_calibrator,
        level3_calibrator_path=args.level3_calibrator,
        thresholds=HierarchyThresholds(level1_positive_probability=args.l1_threshold),
    )
    result = pipeline.predict(
        sequence,
        mc_passes=args.mc_passes,
        latent_samples=args.latent_samples,
    )

    payload = _jsonable(result)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Sequence: {payload['sequence']}")
        print(f"Final label: {payload['final_label']}")
        print(f"Level 1 calibrated probabilities: {payload['level1']['calibrated_probabilities']}")
        if payload.get("level2") is not None:
            print(f"Level 2 calibrated probabilities: {payload['level2']['calibrated_probabilities']}")
        if payload.get("level3") is not None:
            print(f"Level 3 calibrated probabilities: {payload['level3']['calibrated_probabilities']}")
    return 0


def _run_explain(args: argparse.Namespace) -> int:
    sequence = _sequence_for_explain(args)
    tokenizer, protbert_model = load_protbert_components(
        model_name=args.protbert_model,
        cache_dir=args.cache_dir,
    )
    descriptor_extractor = DescriptorFeatureExtractor()

    if args.background_sequences_file:
        background_sequences = [
            clean_protein_sequence(line)
            for line in Path(args.background_sequences_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        background_embeddings, background_descriptors, feature_names, pad_vector = sequences_to_shap_background(
            background_sequences,
            tokenizer,
            protbert_model,
            max_length=args.max_length,
            descriptor_extractor=descriptor_extractor,
        )
    else:
        if not args.background_embeddings or not args.background_descriptors:
            raise SystemExit(
                "Provide --background-sequences-file or both --background-embeddings and --background-descriptors."
            )
        background_embeddings = _load_numpy_file(args.background_embeddings)
        background_descriptors = _load_numpy_file(args.background_descriptors)
        feature_names = _load_feature_names(args.feature_names_file)
        if feature_names is None:
            raise SystemExit(
                "When using precomputed background arrays, provide --feature-names-file for descriptor labels."
            )
        pad_vector = protbert_pad_embedding_vector(protbert_model, tokenizer)

    encoder_path = resolve_drive_artifact_path(args.encoder_model, LEVEL3_ENCODER_SPEC)
    classifier_path = resolve_drive_artifact_path(args.classifier_model, LEVEL3_CLASSIFIER_SPEC)
    encoder_model = tf.keras.models.load_model(encoder_path, compile=False)
    classifier_model = tf.keras.models.load_model(classifier_path, compile=False)

    explainer = MultimodalShapExplainer.from_models(
        protbert_model=protbert_model,
        encoder_model=encoder_model,
        classifier_model=classifier_model,
        pad_embedding_vector=pad_vector,
        background_embeddings=background_embeddings,
        background_descriptors=background_descriptors,
        tokenizer=tokenizer,
        feature_names=feature_names,
    )

    sample_embeddings, _ = sequence_to_protbert_inputs(
        sequence,
        tokenizer,
        protbert_model,
        max_length=args.max_length,
    )
    sample_descriptors = descriptor_extractor.transform_one(sequence)
    bundle = explainer.explain(
        sequence,
        sample_embeddings,
        sample_descriptors,
        max_length=args.max_length,
    )

    token_n = min(len(bundle.tokens), len(bundle.token_attributions))
    desc_n = min(len(bundle.feature_names), len(bundle.descriptor_attributions))

    token_df = pd.DataFrame(
        {
            "token": list(bundle.tokens[:token_n]),
            "shap_value": bundle.token_attributions[:token_n],
            "abs_shap": np.abs(bundle.token_attributions[:token_n]),
        }
    ).sort_values("abs_shap", ascending=False).head(args.top_k)
    desc_df = pd.DataFrame(
        {
            "feature": list(bundle.feature_names[:desc_n]),
            "shap_value": bundle.descriptor_attributions[:desc_n],
            "abs_shap": np.abs(bundle.descriptor_attributions[:desc_n]),
        }
    ).sort_values("abs_shap", ascending=False).head(args.top_k)

    payload = {
        "sequence": sequence,
        "expected_value": _jsonable(bundle.expected_value),
        "top_tokens": token_df.to_dict(orient="records"),
        "top_descriptors": desc_df.to_dict(orient="records"),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Sequence: {sequence}")
        print("Top tokens:")
        print(token_df.to_string(index=False))
        print("Top descriptors:")
        print(desc_df.to_string(index=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        return _run_infer(args)

    if args.command == "explain":
        return _run_explain(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
