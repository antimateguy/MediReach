"""OutreachService: fetch, score, apply policy, explain, save."""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Callable

import pandas as pd

from medireach.data.repositories import DATE_FORMAT, QUEUE_COLUMNS, AppointmentRepository, QueueRepository
from medireach.domain import QueueRun
from medireach.features.builder import FeatureBuilder
from medireach.ml.models import RiskModel
from medireach.outreach.policies import OutreachPolicy
from medireach.outreach.reasons import ReasonExplainer

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class OutreachService:
    """Produces the outreach list for a window of dates and stores it as a run.

    Depends only on abstractions (repositories, model, policy), which are
    handed in. It never decides which concrete class to use.
    """

    def __init__(self, appointments: AppointmentRepository, queue: QueueRepository,
                 feature_builder: FeatureBuilder, model: RiskModel, model_version: str,
                 policy: OutreachPolicy, explainer: ReasonExplainer | None = None,
                 clock: Callable[[], datetime] = _utc_now) -> None:
        self._appointments = appointments
        self._queue = queue
        self._feature_builder = feature_builder
        self._model = model
        self._model_version = model_version
        self._policy = policy
        self._explainer = explainer
        self._clock = clock

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def model_version(self) -> str:
        return self._model_version

    def score(self, start: date, end: date, policy: OutreachPolicy | None = None,
              source: AppointmentRepository | None = None) -> pd.DataFrame:
        """Scored and prioritised appointments for [start, end]. Nothing is saved."""
        return (policy or self._policy).assign(self.score_appointments(start, end, source))

    def score_appointments(self, start: date, end: date, source: AppointmentRepository | None = None) -> pd.DataFrame:
        """Risk score and reasons for each appointment in [start, end], without a policy.

        `source` replaces the service's own repository for this call, for example
        an as-of view of the data at the moment a plan is made.
        """
        # Everything up to `end` is fetched because history features need the
        # patient's earlier appointments, not only the ones in the window.
        fetched = (source or self._appointments).get_appointments(end=end)
        features, meta = self._feature_builder.build(fetched)
        in_window = meta["appointment_date"] >= pd.Timestamp(start)
        if not in_window.any():
            raise ValueError(f"no appointments between {start} and {end}")
        window_features = features[in_window]
        scores = pd.DataFrame({
            "appointment_id": meta.loc[in_window, "appointment_id"],
            "risk_score": self._model.predict_proba(window_features),
        })
        if self._explainer is not None:
            scores["reasons"] = self._explainer.explain(window_features)
        else:
            scores["reasons"] = ""
        # Scores travel with their appointment_id and are joined by key, never
        # by row position (v1 wrote scores to the wrong patients this way).
        return meta[in_window].merge(scores, on="appointment_id", how="inner", validate="one_to_one")

    def run(self, start: date, end: date, notes: str = "") -> QueueRun:
        """Score the window, apply the policy, and append the result as a new run."""
        prioritised = self.score(start, end)
        created = self._clock()
        run = QueueRun(
            run_id=created.strftime("run-%Y%m%dT%H%M%S%f"),
            created_at=created.isoformat(timespec="microseconds"),
            window_start=start.strftime(DATE_FORMAT),
            window_end=end.strftime(DATE_FORMAT),
            model_name=self._model.name,
            model_version=self._model_version,
            policy_name=self._policy.name,
            n_rows=len(prioritised),
            notes=notes,
        )
        self._queue.save_run(run, prioritised[list(QUEUE_COLUMNS)])
        logger.info("Saved %s: %d appointments, %s to %s", run.run_id, run.n_rows, run.window_start, run.window_end)
        return run
