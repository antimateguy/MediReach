"""ModelStore: save and load versioned model bundles."""
from __future__ import annotations

import json
import os

import joblib

from medireach.domain import ModelMetadata
from medireach.errors import ModelBundleError
from medireach.features.builder import FEATURE_COLUMNS
from medireach.ml.models import RiskModel

MODEL_FILE = "model.joblib"
META_FILE = "meta.json"
LATEST_FILE = "LATEST"


class ModelStore:
    """A model bundle is a folder: the fitted model plus the metadata that describes it.

    Layout: models/<version>/model.joblib, models/<version>/meta.json, and
    models/LATEST holding the version to use by default.
    """

    def __init__(self, models_dir: str) -> None:
        self._models_dir = models_dir

    def save(self, model: RiskModel, metadata: ModelMetadata) -> str:
        """Write the bundle, point LATEST at it, and return its version."""
        folder = os.path.join(self._models_dir, metadata.version)
        os.makedirs(folder, exist_ok=True)
        joblib.dump(model, os.path.join(folder, MODEL_FILE))
        with open(os.path.join(folder, META_FILE), "w", encoding="utf-8") as handle:
            json.dump(metadata.to_dict(), handle, indent=2)
        with open(os.path.join(self._models_dir, LATEST_FILE), "w", encoding="utf-8") as handle:
            handle.write(metadata.version)
        return metadata.version

    def latest_version(self) -> str:
        path = os.path.join(self._models_dir, LATEST_FILE)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
        except OSError as exc:
            raise ModelBundleError(f"no trained model found ({path}); run the train pipeline first") from exc

    def load_metadata(self, version: str | None = None) -> ModelMetadata:
        path = os.path.join(self._models_dir, version or self.latest_version(), META_FILE)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return ModelMetadata.from_dict(json.load(handle))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelBundleError(f"cannot read model metadata {path}: {exc}") from exc

    def load(self, version: str | None = None) -> tuple[RiskModel, ModelMetadata]:
        """Load a bundle (LATEST by default) after checking it matches the current features."""
        metadata = self.load_metadata(version)
        # A model trained on a different feature list would score silently
        # wrong numbers (v1 defect D3), so refuse to load it.
        if list(metadata.feature_columns) != list(FEATURE_COLUMNS):
            raise ModelBundleError(
                f"model {metadata.version} was trained on a different feature list than this code builds"
            )
        path = os.path.join(self._models_dir, metadata.version, MODEL_FILE)
        try:
            model = joblib.load(path)
        except Exception as exc:  # a damaged pickle can raise almost any error type
            raise ModelBundleError(f"cannot read model file {path}: {exc}") from exc
        if not isinstance(model, RiskModel):
            raise ModelBundleError(f"{path} does not contain a RiskModel")
        return model, metadata
