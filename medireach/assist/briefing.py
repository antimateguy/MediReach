"""BriefingWriter: the admin's morning briefing. The code computes every number; AI only words them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from medireach.assist.llm import TextCache, TextGenerator, cache_key, unsupported_numbers
from medireach.domain import Plan
from medireach.errors import AIUnavailable
from medireach.live.status import LiveSnapshot
from medireach.ops.measurement import ExperimentSummary

FEATURE = "briefing"
SYSTEM = (
    "You write a short morning briefing for the operations manager of an outpatient clinic. "
    "Use only the facts you are given. Copy every number exactly as it is written in the facts and add no "
    "other numbers. Plain English, no jargon, no advice beyond what the facts support. "
    'Reply as JSON: {"headline": "<one sentence>", "bullets": ["<3 to 5 short sentences>"]}'
)


@dataclass(frozen=True)
class Briefing:
    headline: str
    bullets: list[str]
    source: str                     # "AI" or "template"
    model: str | None = None
    note: str = ""
    facts: dict[str, str] = field(default_factory=dict)


def briefing_facts(snapshot: LiveSnapshot, plan: Plan | None, numbers: Mapping[str, float] | None,
                   tasks_done: int, tasks_total: int, summary: ExperimentSummary) -> dict[str, str]:
    """The numbers the briefing may use, already formatted the way they should appear."""
    facts = {
        "today": f"{snapshot.moment.real_now:%A %d %B %Y}",
        "clinic_open_today": "yes" if snapshot.is_open else "no",
        "appointments_today_booked_so_far": f"{snapshot.appointments_today:,}",
        "same_day_bookings_so_far": f"{snapshot.same_day_bookings_so_far:,}",
        "next_clinic_day": f"{snapshot.next_clinic_day:%A %d %B}",
        "appointments_known_for_next_clinic_day": f"{snapshot.appointments_next_day_known:,}",
        "plan_for_next_clinic_day": plan.status.value.lower() if plan else "not made yet",
        "tasks_done_today": f"{tasks_done:,} of {tasks_total:,}",
        "days_measured": f"{summary.closed_days:,}",
        "outcomes_used": {"simulated": "a simulation with a planted effect", "historical": "real 2016 outcomes"}.get(summary.source, "none yet"),
    }
    if numbers:
        facts.update({
            "calls_planned": f"{int(numbers['calls']):,}", "sms_planned": f"{int(numbers['sms']):,}",
            "expected_no_shows_next_clinic_day": f"{numbers['expected_no_shows']:,.0f}",
            "share_of_expected_no_shows_on_call_list": f"{numbers['call_list_capture']:.1%}",
        })
    if summary.relative_reduction is not None and summary.reduction_ci is not None:
        low, high = summary.reduction_ci
        facts["measured_reduction_in_no_shows"] = f"{summary.relative_reduction:.1%} (95% CI {low:.1%} to {high:.1%})"
    return facts


class BriefingWriter:
    """Template briefing always; an AI-worded one on request, checked and cached."""

    def __init__(self, generator: TextGenerator | None, cache: TextCache) -> None:
        self._generator = generator
        self._cache = cache

    @property
    def ai_available(self) -> bool:
        return self._generator is not None

    def template(self, facts: Mapping[str, str], note: str = "") -> Briefing:
        bullets = [f"{facts['appointments_today_booked_so_far']} appointments are booked for today so far, "
                   f"{facts['same_day_bookings_so_far']} of them same-day bookings."]
        if "calls_planned" in facts:
            bullets.append(f"The plan for {facts['next_clinic_day']} is {facts['plan_for_next_clinic_day']}: "
                           f"{facts['calls_planned']} calls and {facts['sms_planned']} SMS, about "
                           f"{facts['expected_no_shows_next_clinic_day']} expected no-shows "
                           f"({facts['share_of_expected_no_shows_on_call_list']} of them on the call list).")
        else:
            bullets.append(f"The plan for {facts['next_clinic_day']} is not made yet.")
        bullets.append(f"Tasks done today: {facts['tasks_done_today']}.")
        if "measured_reduction_in_no_shows" in facts:
            bullets.append(f"Measured so far ({facts['outcomes_used']}): {facts['measured_reduction_in_no_shows']} fewer "
                           f"no-shows in the call arm than in the hold-out, over {facts['days_measured']} days.")
        headline = f"{facts['today']}: {facts['appointments_known_for_next_clinic_day']} appointments known for {facts['next_clinic_day']}."
        return Briefing(headline, bullets, "template", None, note, dict(facts))

    def cached(self, facts: Mapping[str, str]) -> Briefing | None:
        hit = self._cache.get(cache_key(FEATURE, facts))
        return None if hit is None else Briefing(hit[0]["headline"], list(hit[0]["bullets"]), "AI", hit[1], "", dict(facts))

    def write(self, facts: Mapping[str, str]) -> Briefing:
        """An AI briefing if one can be made and passes the checks; otherwise the template, with the reason."""
        if self._generator is None:
            return self.template(facts, "AI is not set up (no Gemini key).")
        cached = self.cached(facts)
        if cached is not None:
            return cached
        try:
            content, model = self._generator.generate_json(SYSTEM, _prompt(facts))
            headline, bullets = _validate(content, facts)
        except AIUnavailable as exc:
            return self.template(facts, f"AI text was not used: {exc}.")
        self._cache.put(cache_key(FEATURE, facts), FEATURE, model, {"headline": headline, "bullets": bullets})
        return Briefing(headline, bullets, "AI", model, "", dict(facts))


def _prompt(facts: Mapping[str, str]) -> str:
    return "Facts:\n" + "\n".join(f"- {name.replace('_', ' ')}: {value}" for name, value in facts.items())


def _validate(content: Mapping[str, Any], facts: Mapping[str, str]) -> tuple[str, list[str]]:
    headline, bullets = content.get("headline"), content.get("bullets")
    if not isinstance(headline, str) or not isinstance(bullets, list) or not 1 <= len(bullets) <= 6 \
            or not all(isinstance(b, str) for b in bullets):
        raise AIUnavailable("the reply did not have a headline and a list of bullets")
    extra = unsupported_numbers(" ".join([headline, *bullets]), facts)
    if extra:
        raise AIUnavailable(f"the reply contained numbers that are not in the facts ({', '.join(sorted(extra))})")
    return headline.strip(), [bullet.strip() for bullet in bullets]
