"""TemplateScripts: a short, polite call script and SMS for each task, built from the risk reasons."""
from __future__ import annotations

from datetime import date

from medireach.assist.llm import TextCache, TextGenerator, cache_key, unsupported_numbers
from medireach.errors import AIUnavailable

# Reason phrase (from outreach/reasons.py) -> one sentence the receptionist can say.
# Worded to help, never to blame: a reason explains why we call, not what the patient did wrong.
_REASON_LINES: dict[str, str] = {
    "Booked 4-7 days ahead": "Your appointment was booked a few days ago, so we are checking it still suits you.",
    "Booked 8-14 days ahead": "Your appointment was booked over a week ago, so we are checking it still suits you.",
    "Booked 15-30 days ahead": "Your appointment was booked a few weeks ago, so we are checking it still suits you.",
    "Booked more than a month ahead": "Your appointment was booked more than a month ago, so we are checking it still suits you.",
    "Missed a previous appointment": "We would like to make sure this time works well for you.",
    "No history yet (uncertain)": "As this may be one of your first visits, we wanted to make sure you have the details.",
    "Child (0-17), depends on a carer": "As the appointment is for a child, we wanted to check the time works for the family.",
    "Younger adult (18-34)": "We know schedules can change, so we are checking the time still works.",
    "Enrolled in welfare programme (may need support)": "If getting to the clinic is difficult, we can tell you about the support available.",
    "Handicap recorded (may need support)": "If you need any help getting to the clinic, just let us know.",
}
_DEFAULT_LINE = "We are calling to confirm your appointment."


class TemplateScripts:
    """Fixed wording, always available. Block I3 adds AI drafts that fall back to this."""

    name = "template"

    @staticmethod
    def _clinic(clinic: str) -> str:
        return clinic.title()

    def call_script(self, reasons: str, appointment_day: date, clinic: str, appointments: int = 1) -> str:
        line = next((_REASON_LINES[reason.strip()] for reason in reasons.split(";") if reason.strip() in _REASON_LINES),
                    _DEFAULT_LINE)
        which = "your appointments" if appointments > 1 else "your appointment"
        return (f"Hello, this is the {self._clinic(clinic)} health unit calling about {which} on "
                f"{appointment_day:%A %d %B}. {line} Can you still come? If not, we can move it to another day.")

    def sms_text(self, appointment_day: date, clinic: str, appointments: int = 1) -> str:
        which = "appointments" if appointments > 1 else "an appointment"
        return (f"Reminder: you have {which} at the {self._clinic(clinic)} health unit on {appointment_day:%a %d %b}. "
                "Reply 1 to confirm or 2 to change it.")


SCRIPT_FEATURE = "call_script"
SCRIPT_SYSTEM = (
    "You improve a short phone script for a clinic receptionist calling a patient to confirm an appointment. "
    "Keep it to two or three sentences, warm, polite and in plain English. Never mention risk, scores, "
    "predictions, or that the patient might not come. Give no medical advice. Use only the facts given, "
    "copy any number exactly, and add no other numbers. "
    'Reply as JSON: {"script": "<the script>"}'
)
MAX_SCRIPT_LENGTH = 600


class ScriptWriter:
    """The template script always; an AI-polished version on request, checked and cached.

    Only the reasons, the clinic and the appointment day are sent: no patient id, name or age.
    """

    def __init__(self, generator: TextGenerator | None, cache: TextCache, templates: TemplateScripts) -> None:
        self._generator, self._cache, self._templates = generator, cache, templates

    @property
    def ai_available(self) -> bool:
        return self._generator is not None

    def polished(self, reasons: str, appointment_day: date, clinic: str, appointments: int = 1) -> tuple[str, str, str]:
        """(script, source, note). Source is "AI" or "template"; the note explains a fallback."""
        draft = self._templates.call_script(reasons, appointment_day, clinic, appointments)
        facts = {"reasons": reasons, "clinic": clinic.title(), "appointment_day": f"{appointment_day:%A %d %B}",
                 "appointments": str(appointments), "draft_script": draft}
        if self._generator is None:
            return draft, "template", "AI is not set up (no Gemini key)."
        key = cache_key(SCRIPT_FEATURE, facts)
        hit = self._cache.get(key)
        if hit is not None:
            return str(hit[0]["script"]), "AI", ""
        try:
            content, model = self._generator.generate_json(SCRIPT_SYSTEM, "\n".join(f"{k}: {v}" for k, v in facts.items()))
            script = content.get("script")
            if not isinstance(script, str) or not script.strip() or len(script) > MAX_SCRIPT_LENGTH:
                raise AIUnavailable("the reply was not a short script")
            extra = unsupported_numbers(script, facts)
            if extra:
                raise AIUnavailable(f"the reply contained numbers that are not in the facts ({', '.join(sorted(extra))})")
        except AIUnavailable as exc:
            return draft, "template", f"AI text was not used: {exc}."
        self._cache.put(key, SCRIPT_FEATURE, model, {"script": script.strip()})
        return script.strip(), "AI", ""
