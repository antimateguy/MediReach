"""Phase 6: reason codes and the outreach service, end to end with in-memory doubles."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from medireach.config import Settings
from medireach.data.database import SqliteDatabase
from medireach.data.repositories import QUEUE_COLUMNS, SqliteAppointmentRepository, SqliteQueueRepository
from medireach.features.builder import FEATURE_COLUMNS, FeatureBuilder
from medireach.ml.models import LogisticRiskModel
from medireach.outreach.policies import CapacityPolicy, ThresholdPolicy
from medireach.outreach.reasons import (
    HISTORY_TEXT_MISSED,
    HISTORY_TEXT_NEW,
    HISTORY_TEXT_OTHER,
    NO_REASON,
    REASON_TEXT,
    ReasonExplainer,
)
from medireach.outreach.service import OutreachService
from tests.doubles import (
    FixedContributionModel,
    InMemoryAppointmentRepository,
    InMemoryQueueRepository,
    ShuffledAppointmentRepository,
    StubRiskModel,
)

PRIOR_RATE = 0.2
WINDOW = (date(2016, 5, 23), date(2016, 5, 27))  # the last week of the synthetic data


class _Clock:
    """Hands out one fixed time per call, so run ids are predictable."""

    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> datetime:
        self._calls += 1
        return datetime(2016, 6, 1, 18, 0, self._calls, tzinfo=timezone.utc)


@pytest.fixture()
def repository(cleaned) -> InMemoryAppointmentRepository:
    patients, appointments, _ = cleaned
    return InMemoryAppointmentRepository(patients, appointments)


@pytest.fixture()
def builder(settings: Settings) -> FeatureBuilder:
    return FeatureBuilder(settings, PRIOR_RATE)


def _service(repository, queue, builder, **overrides) -> OutreachService:
    arguments = dict(
        appointments=repository, queue=queue, feature_builder=builder, model=StubRiskModel(),
        model_version="v-stub", policy=CapacityPolicy(0.10, 0.20), clock=_Clock(),
    )
    arguments.update(overrides)
    return OutreachService(**arguments)


# -------------------------------------------------------------------- service

def test_scores_stay_attached_to_the_right_appointment_after_a_shuffle(repository, builder) -> None:
    # Guards v1 defect D4: scores were matched to patients by row position.
    features, meta = builder.build(repository.get_appointments())
    expected = StubRiskModel.expected(features).set_axis(meta["appointment_id"])

    queue = InMemoryQueueRepository()
    run = _service(ShuffledAppointmentRepository(repository), queue, builder).run(*WINDOW)
    stored = queue.get_run(run.run_id).set_index("appointment_id")

    assert len(stored) > 10 and stored["risk_score"].nunique() > 3
    assert stored["risk_score"].to_numpy() == pytest.approx(expected.loc[stored.index].to_numpy())


def test_only_the_window_is_scored_but_earlier_visits_count_as_history(repository, builder) -> None:
    queue = InMemoryQueueRepository()
    run = _service(repository, queue, builder).run(*WINDOW)
    stored = queue.get_run(run.run_id)
    everything = repository.get_appointments()
    in_window = everything[everything["appointment_date"].between(pd.Timestamp(WINDOW[0]), pd.Timestamp(WINDOW[1]))]
    assert set(stored["appointment_id"]) == set(in_window["appointment_id"])
    assert run.n_rows == len(in_window)
    # The stub adds 0.30 for has_history. If the service had built features on
    # the window alone, almost nobody would have history.
    assert (stored["risk_score"] >= 0.35).mean() > 0.5


def test_run_is_stored_with_its_metadata_and_without_outcomes(repository, builder) -> None:
    queue = InMemoryQueueRepository()
    run = _service(repository, queue, builder).run(*WINDOW, notes="evening run")
    assert run.run_id == "run-20160601T180001000000"
    assert (run.window_start, run.window_end) == ("2016-05-23", "2016-05-27")
    assert (run.model_name, run.model_version, run.policy_name, run.notes) == ("stub", "v-stub", "capacity", "evening run")
    stored = queue.get_run(run.run_id)
    assert tuple(stored.columns) == QUEUE_COLUMNS and "no_show" not in stored.columns
    assert set(stored["priority"]) == {"HIGH", "MEDIUM", "LOW"}
    assert (stored["reasons"] == "").all()  # the stub model cannot explain itself


def test_a_second_run_is_appended_and_the_first_is_kept(repository, builder) -> None:
    queue = InMemoryQueueRepository()
    service = _service(repository, queue, builder)
    first = service.run(*WINDOW)
    second = service.run(date(2016, 5, 26), date(2016, 5, 27))
    assert first.run_id != second.run_id
    assert len(queue.get_run(first.run_id)) == first.n_rows > second.n_rows == len(queue.get_run(second.run_id))
    assert queue.latest_run_id() == second.run_id
    assert list(queue.list_runs()["run_id"]) == [second.run_id, first.run_id]


def test_score_can_try_another_policy_without_saving(repository, builder) -> None:
    queue = InMemoryQueueRepository()
    service = _service(repository, queue, builder)
    by_capacity = service.score(*WINDOW)
    by_threshold = service.score(*WINDOW, policy=ThresholdPolicy(high=0.50, medium=0.35))
    assert queue.latest_run_id() is None
    assert by_capacity["risk_score"].tolist() == by_threshold["risk_score"].tolist()
    assert (by_threshold.loc[by_threshold["risk_score"] >= 0.5, "priority"] == "HIGH").all()


def test_an_empty_window_is_an_error(repository, builder) -> None:
    with pytest.raises(ValueError, match="no appointments"):
        _service(repository, InMemoryQueueRepository(), builder).run(date(2017, 1, 1), date(2017, 1, 2))


def test_end_to_end_with_sqlite_and_a_real_model(database: SqliteDatabase, builder) -> None:
    appointments = SqliteAppointmentRepository(database)
    features, meta = builder.build(appointments.get_appointments())
    model = LogisticRiskModel().fit(features, meta["no_show"])
    queue = SqliteQueueRepository(database)
    service = _service(appointments, queue, builder, model=model, explainer=ReasonExplainer(model))
    run = service.run(*WINDOW)

    stored = queue.get_run(run.run_id)
    assert queue.latest_run_id() == run.run_id and len(stored) == run.n_rows
    assert stored["risk_score"].between(0, 1).all()
    assert (stored["reasons"].str.len() > 0).all()
    truth = meta.assign(score=model.predict_proba(features)).set_index("appointment_id")["score"]
    assert stored["risk_score"].to_numpy() == pytest.approx(truth.loc[stored["appointment_id"]].to_numpy())


# -------------------------------------------------------------------- reasons

def _row(**values: float) -> dict[str, float]:
    return {**{column: 0.0 for column in FEATURE_COLUMNS}, **values}


def _explain(features: list[dict[str, float]], contributions: list[dict[str, float]], **options) -> list[str]:
    feature_frame = pd.DataFrame(features, columns=list(FEATURE_COLUMNS))
    contribution_frame = pd.DataFrame(contributions, columns=list(FEATURE_COLUMNS)).fillna(0.0)
    return list(ReasonExplainer(FixedContributionModel(contribution_frame), **options).explain(feature_frame))


def test_reasons_are_the_three_largest_positive_contributions_in_order() -> None:
    features = [_row(**{"lead_bucket_15-30": 1, "age_group_18-34": 1, "scholarship": 1, "alcoholism": 1, "gender_male": 1})]
    contributions = [{"lead_bucket_15-30": 0.6, "age_group_18-34": 0.2, "scholarship": 0.3, "alcoholism": 0.1, "gender_male": -0.4}]
    expected = [REASON_TEXT["lead_bucket_15-30"], REASON_TEXT["scholarship"], REASON_TEXT["age_group_18-34"]]
    assert _explain(features, contributions) == ["; ".join(expected)]


def test_negative_and_tiny_contributions_are_not_reasons() -> None:
    features = [_row(lead_bucket_0=1, gender_male=1), _row(gender_male=1)]
    contributions = [{"lead_bucket_0": -1.8, "gender_male": 0.04}, {"gender_male": 0.04}]
    assert _explain(features, contributions) == [NO_REASON, NO_REASON]
    assert _explain(features, contributions, min_contribution=0.01) == [REASON_TEXT["gender_male"]] * 2


def test_history_columns_are_judged_together() -> None:
    missed = _row(has_history=1, had_prior_no_show=1, prior_appt_count=2, prior_no_show_rate_smoothed=0.6)
    attended = _row(has_history=1, prior_appt_count=4, prior_no_show_rate_smoothed=0.07)
    new = _row(prior_no_show_rate_smoothed=0.2)
    contributions = [
        # One column is negative (as in the real model), but the block is positive overall.
        {"prior_no_show_rate_smoothed": 0.9, "had_prior_no_show": -0.6, "has_history": 0.1},
        # One column is positive, but the block is negative overall: not a reason.
        {"prior_no_show_rate_smoothed": -0.5, "has_history": 0.13, "prior_appt_count": 0.2},
        {"prior_appt_count": 0.3},
    ]
    assert _explain([missed, attended, new], contributions) == [HISTORY_TEXT_MISSED, NO_REASON, HISTORY_TEXT_NEW]
    positive_without_misses = [{"has_history": 0.2, "prior_appt_count": 0.2}]
    assert _explain([attended], positive_without_misses) == [HISTORY_TEXT_OTHER]


def test_every_non_history_feature_has_a_phrase() -> None:
    history = {"has_history", "prior_appt_count", "prior_no_show_rate_smoothed", "had_prior_no_show"}
    assert set(REASON_TEXT) == set(FEATURE_COLUMNS) - history


def test_reasons_keep_the_index_of_the_features() -> None:
    features = pd.DataFrame([_row(scholarship=1), _row()], index=[70, 30])
    contributions = pd.DataFrame([_row(scholarship=0.3), _row()], index=[70, 30])
    reasons = ReasonExplainer(FixedContributionModel(contributions)).explain(features)
    assert list(reasons.index) == [70, 30]
    assert list(reasons) == [REASON_TEXT["scholarship"], NO_REASON]
