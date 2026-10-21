"""Pipeline: backtest the outreach strategies on the test period under the default assumptions."""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from medireach import artifacts as files
from medireach.bootstrap import Container, build_container, configure_logging
from medireach.config import Settings
from medireach.domain import Action
from medireach.errors import ModelBundleError
from medireach.outreach.policies import CapacityPolicy
from medireach.pipelines.train import SCORE_COLUMNS

# Display name -> score column in test_scores.csv, simplest strategy first.
STRATEGIES: dict[str, str] = {
    "Lead-time baseline": "score_baseline",
    "Logistic regression": "score_lr",
    "Random forest": "score_rf",
}
EFFECTIVENESS_GRID: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)
CALL_SHARE_GRID: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30)


def selected_scores(container: Container) -> pd.DataFrame:
    """Test-period rows with the selected model's score as risk_score, plus the outcome."""
    if container.metadata is None:
        raise ModelBundleError("no trained model is loaded; run the train pipeline first")
    scores = container.artifacts.read_csv(files.TEST_SCORES)
    column = SCORE_COLUMNS[container.metadata.model_name]
    return scores[["appointment_id", "appointment_date", "no_show"]].assign(risk_score=scores[column])


def write_impact(container: Container) -> pd.DataFrame:
    """Compare strategies, find the break-even point, size the pilot, and save all three."""
    settings, simulator = container.settings, container.simulator
    policy = CapacityPolicy(settings.call_share, settings.sms_share)
    scores = container.artifacts.read_csv(files.TEST_SCORES)
    comparison = simulator.compare(scores, STRATEGIES, policy, settings.impact)

    selected = selected_scores(container)
    sensitivity = simulator.sensitivity(selected, settings.impact, EFFECTIVENESS_GRID, CALL_SHARE_GRID, settings.sms_share)
    assigned = policy.assign(selected)
    called = assigned[assigned["action"] == Action.CALL.value]
    # The pilot measures the effect of being put on the call list, answered or
    # not, so the reduction it must detect is reach x effectiveness.
    plan = container.pilot.plan(
        p_control=float(called["no_show"].mean()),
        relative_reduction=settings.impact.call_reach_rate * settings.impact.call_effectiveness,
        daily_high_volume=len(called) / assigned["appointment_date"].nunique(),
    )
    container.artifacts.write_csv(files.IMPACT_COMPARISON, comparison)
    container.artifacts.write_csv(files.IMPACT_SENSITIVITY, sensitivity)
    container.artifacts.write_json(files.IMPACT_SUMMARY, {
        "assumptions": asdict(settings.impact),
        "policy": {"name": policy.name, "call_share": policy.call_share, "sms_share": policy.sms_share},
        "selected_model": container.metadata.model_name if container.metadata else None,
        "break_even_call_effectiveness": simulator.break_even_effectiveness(selected, policy, settings.impact),
        "pilot_plan": asdict(plan),
    })
    return comparison


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings)
    comparison = write_impact(build_container(settings))
    columns = ["strategy", "n_calls", "n_sms", "precision_in_contacted", "capture_rate", "prevented", "cost", "net"]
    print(comparison[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
