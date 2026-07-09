"""Artifact resolution helpers for downloadable model components.

This module centralizes the local fallback paths and Google Drive download
logic used by the hierarchical classifiers and the multimodal explainability
stack. The rest of the package calls into these helpers so the download policy
stays consistent across the library and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_CACHE_ROOT = Path("models") / "downloaded"
DEFAULT_PROTBERT_CACHE_DIR = DEFAULT_MODEL_CACHE_ROOT / "protbert"


@dataclass(frozen=True)
class DriveArtifactSpec:
    """Describe a downloadable artifact and its default local location."""

    default_path: Path
    drive_url: str
    description: str


LEVEL1_MODEL_SPEC = DriveArtifactSpec(
    default_path=DEFAULT_MODEL_CACHE_ROOT / "hierarchy" / "level1_classifier.keras",
    drive_url="https://drive.google.com/uc?id=REPLACE_ME_LEVEL1_MODEL_ID",
    description="Level 1 hierarchical classifier",
)

LEVEL2_MODEL_SPEC = DriveArtifactSpec(
    default_path=DEFAULT_MODEL_CACHE_ROOT / "hierarchy" / "level2_classifier.keras",
    drive_url="https://drive.google.com/uc?id=REPLACE_ME_LEVEL2_MODEL_ID",
    description="Level 2 hierarchical classifier",
)

LEVEL3_ENCODER_SPEC = DriveArtifactSpec(
    default_path=DEFAULT_MODEL_CACHE_ROOT / "hierarchy" / "level3_encoder.keras",
    drive_url="https://drive.google.com/uc?id=REPLACE_ME_LEVEL3_ENCODER_ID",
    description="Level 3 encoder",
)

LEVEL3_CLASSIFIER_SPEC = DriveArtifactSpec(
    default_path=DEFAULT_MODEL_CACHE_ROOT / "hierarchy" / "level3_classifier.keras",
    drive_url="https://drive.google.com/uc?id=REPLACE_ME_LEVEL3_CLASSIFIER_ID",
    description="Level 3 classifier head",
)


def _expand_path(path: str | Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path
    return Path(path).expanduser()


def _is_placeholder_drive_url(url: str) -> bool:
    return "REPLACE_ME" in url or not url.strip()


def download_gdrive_artifact(url: str, output_path: Path, *, description: str) -> Path:
    """Download a Google Drive artifact to ``output_path`` using ``gdown``."""

    if _is_placeholder_drive_url(url):
        raise FileNotFoundError(
            f"No Google Drive URL has been configured yet for {description}. "
            "Replace the placeholder in src/chitinase_clf/artifacts.py."
        )

    try:
        import gdown
    except Exception as exc:  # pragma: no cover - import-time failure path
        raise ImportError(
            "gdown is required to download missing model artifacts."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(
        url=url,
        output=str(output_path),
        quiet=False,
        fuzzy=True,
    )
    if not downloaded or not output_path.exists():
        raise RuntimeError(f"Failed to download {description} from Google Drive.")
    return output_path


def resolve_drive_artifact_path(
    path: str | Path | None,
    spec: DriveArtifactSpec,
) -> Path:
    """Return an existing artifact path or download the configured default cache.

    Explicit paths are treated as user-provided inputs and must already exist.
    Only the built-in default cache path is allowed to fall back to Google Drive
    so the library does not unexpectedly try to download artifacts when the
    caller clearly asked for a local file.
    """

    if path is None:
        candidate = spec.default_path
        if candidate.exists():
            return candidate
        return download_gdrive_artifact(
            spec.drive_url,
            candidate,
            description=spec.description,
        )

    candidate = _expand_path(path, spec.default_path)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Artifact not found at explicit path: {candidate}")


def _cache_dir_str(cache_dir: str | Path | None) -> str:
    if cache_dir is None:
        return str(DEFAULT_PROTBERT_CACHE_DIR)
    return str(Path(cache_dir).expanduser())


def _protbert_available_locally(model_name: str, cache_dir: str | Path | None) -> bool:
    try:
        from transformers import BertTokenizer, TFBertModel
    except Exception:  # pragma: no cover - import-time failure path
        return False

    cache_dir_str = _cache_dir_str(cache_dir)
    try:
        BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            cache_dir=cache_dir_str,
            local_files_only=True,
        )
        TFBertModel.from_pretrained(
            model_name,
            from_pt=True,
            cache_dir=cache_dir_str,
            local_files_only=True,
        )
    except Exception:
        return False
    return True


def load_protbert_components(
    model_name: str = "Rostlab/prot_bert",
    *,
    cache_dir: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load ProtBERT components, downloading them if they are not cached."""

    try:
        from transformers import BertTokenizer, TFBertModel
    except Exception as exc:  # pragma: no cover - import-time failure path
        raise ImportError(
            "transformers<5 and TensorFlow are required to load ProtBERT."
        ) from exc

    cache_dir_str = _cache_dir_str(cache_dir)
    model_path = Path(model_name).expanduser()
    model_reference = str(model_path) if model_path.exists() else model_name

    if not Path(model_reference).exists():
        _ = _protbert_available_locally(model_reference, cache_dir_str)

    tokenizer = BertTokenizer.from_pretrained(
        model_reference,
        do_lower_case=False,
        cache_dir=cache_dir_str,
    )
    model = TFBertModel.from_pretrained(
        model_reference,
        from_pt=True,
        cache_dir=cache_dir_str,
    )
    return tokenizer, model


__all__ = [
    "DEFAULT_MODEL_CACHE_ROOT",
    "DEFAULT_PROTBERT_CACHE_DIR",
    "DriveArtifactSpec",
    "LEVEL1_MODEL_SPEC",
    "LEVEL2_MODEL_SPEC",
    "LEVEL3_CLASSIFIER_SPEC",
    "LEVEL3_ENCODER_SPEC",
    "download_gdrive_artifact",
    "load_protbert_components",
    "resolve_drive_artifact_path",
]
