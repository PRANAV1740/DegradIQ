"""Stable prediction-engine interface between the validated ML pipeline and apps.

`PredictionEngine` loads a first-stop model bundle ONCE, validates its compatibility
with the current feature schema and label definition, and exposes a single, typed
forecast method that the FastAPI service and the Streamlit dashboard can both call.
It reuses the committed hazard mathematics in ``pit_strategy.predict_first_stops``
(no re-implementation) and augments each forecast with the 10-90 interval WIDTH and a
provenance block.

Design contract:
  * Load once (no per-request retraining); the fitted pipeline is held in memory.
  * Compatibility is checked at load: the model's feature columns must equal the
    current ``FEATURE_COLUMNS``; an incompatible model raises ``IncompatibleModelError``
    rather than silently mis-predicting.
  * The label-definition version is recorded and classified ('current',
    'unversioned_legacy', or 'stale'); callers may require the current version.
  * The production logistic-regression model is loaded by default and is never
    modified by this class. A validated candidate can be loaded by passing its path;
    promotion remains an explicit, separate step.
  * ``context_mode`` ('historical' | 'replay' | 'live') labels how the forecast is
    being used; ``mode`` ('pre-race' | 'live') selects the feature regime. They are
    reported, never conflated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib

try:
    from .pit_strategy import (
        FEATURE_COLUMNS, LABEL_DEFINITION_VERSION, MODEL_PATH, METADATA_PATH,
        predict_first_stops,
    )
except ImportError:  # Vercel root set to the package dir.
    from pit_strategy import (  # type: ignore
        FEATURE_COLUMNS, LABEL_DEFINITION_VERSION, MODEL_PATH, METADATA_PATH,
        predict_first_stops,
    )

import json

CONTEXT_MODES = ("historical", "replay", "live")
FEATURE_MODES = ("pre-race", "live")


class IncompatibleModelError(RuntimeError):
    """Raised when a model's feature schema or label version is incompatible."""


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class PredictionEngine:
    """Load-once, compatibility-checked first-stop forecaster for the apps."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        metadata_path: Path | str | None = None,
        *,
        require_current_label_version: bool = False,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = (
            Path(metadata_path) if metadata_path is not None
            else self.model_path.with_name(self.model_path.stem + "_metadata.json")
        )
        if not self.model_path.exists():
            raise IncompatibleModelError(f"Model artifact not found: {self.model_path}")

        loaded = joblib.load(self.model_path)
        self._bundle = self._normalise_bundle(loaded)
        self._metadata = self._load_metadata()

        self.feature_columns = list(
            self._bundle.get("feature_columns")
            or self._metadata.get("feature_columns")
            or FEATURE_COLUMNS
        )
        self.label_version = (
            self._bundle.get("label_definition_version")
            or self._metadata.get("label_definition_version")
        )
        self.model_sha256 = _sha256(self.model_path)

        self._validate_feature_schema()
        self.label_version_status = self._classify_label_version()
        if require_current_label_version and self.label_version_status != "current":
            raise IncompatibleModelError(
                f"Model label version is '{self.label_version}' "
                f"({self.label_version_status}); current '{LABEL_DEFINITION_VERSION}' required.")

    # ---- loading / validation -------------------------------------------------
    @staticmethod
    def _normalise_bundle(loaded: Any) -> dict[str, Any]:
        """Accept both a production bundle dict and a bare fitted pipeline."""
        if isinstance(loaded, dict) and "pipeline" in loaded:
            return dict(loaded)
        if hasattr(loaded, "predict_proba"):
            return {"pipeline": loaded}
        raise IncompatibleModelError(
            "Unrecognised model artifact: expected a bundle dict with 'pipeline' or a "
            "fitted estimator exposing predict_proba.")

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _validate_feature_schema(self) -> None:
        if list(self.feature_columns) != list(FEATURE_COLUMNS):
            raise IncompatibleModelError(
                "Model feature schema is incompatible with the current pipeline.\n"
                f"  model:   {self.feature_columns}\n  current: {FEATURE_COLUMNS}")

    def _classify_label_version(self) -> str:
        if self.label_version is None:
            return "unversioned_legacy"
        return "current" if self.label_version == LABEL_DEFINITION_VERSION else "stale"

    # ---- public API -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._bundle.get("pipeline") is not None

    def provenance(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "feature_columns": self.feature_columns,
            "label_definition_version": self.label_version,
            "label_version_status": self.label_version_status,
            "trained_at_utc": self._metadata.get("trained_at_utc")
            or self._bundle.get("trained_at_utc"),
            "training_seasons": self._metadata.get("training_seasons")
            or self._bundle.get("training_seasons"),
            "architecture": self._metadata.get("architecture")
            or self._bundle.get("architecture", "unknown"),
        }

    def predict(
        self,
        event_name: str,
        race_laps: int,
        drivers: list[dict[str, Any]],
        mode: str = "pre-race",
        context_mode: str = "historical",
    ) -> dict[str, Any]:
        """Per-driver first-stop forecast with interval width + provenance.

        `mode` is the feature regime ('pre-race' | 'live'); `context_mode` labels how
        the result is used ('historical' | 'replay' | 'live'). Delegates the hazard
        mathematics to the committed ``predict_first_stops``.
        """
        if mode not in FEATURE_MODES:
            raise ValueError(f"mode must be one of {FEATURE_MODES}")
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")

        result = predict_first_stops(self._bundle, event_name, race_laps, drivers, mode)

        # Augment each forecast with the 10-90 interval width (laps), so callers get
        # the sharpness alongside the window — never coverage/width confusion.
        for pred in result.get("predictions", []):
            window = pred.get("likely_window_if_stops") or {}
            lo, hi = window.get("from_lap"), window.get("to_lap")
            pred["interval_width_if_stops"] = (
                int(hi - lo) if (lo is not None and hi is not None) else None)

        result["engine"] = {
            "context_mode": context_mode,
            "feature_mode": mode,
            "model": self.provenance(),
            "interval_note": (
                "likely_window_if_stops is a 10-90 interval; interval_width_if_stops is "
                "its width in laps. Width is reported so wide (uncertain) windows are "
                "not mistaken for precise timing."),
        }
        return result


    def score_rows(self, frame: "Any") -> "Any":
        """Per-row P(first stop on the next lap) for actual feature rows.

        Unlike ``predict`` (which forecasts synthetic future laps), this scores an
        existing frame of observed per-lap rows — used by point-in-time historical
        replay, where each row's features are exactly what was known at the end of
        that lap. The frame must contain the current ``FEATURE_COLUMNS``.
        """
        missing = [c for c in self.feature_columns if c not in getattr(frame, "columns", [])]
        if missing:
            raise ValueError(f"score_rows: frame is missing feature columns: {missing}")
        return self._bundle["pipeline"].predict_proba(frame[self.feature_columns])[:, 1]


def load_default_engine(require_current_label_version: bool = False) -> PredictionEngine | None:
    """Load the production engine, or None if the artifact is absent/incompatible.

    Mirrors the fail-soft behaviour the API expects (503 'model not trained').
    """
    try:
        return PredictionEngine(
            MODEL_PATH, METADATA_PATH,
            require_current_label_version=require_current_label_version)
    except IncompatibleModelError:
        return None
