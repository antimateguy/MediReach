"""v3 I3: AI text (Gemini) with templates, number checks and a cache. No test touches the network."""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import date
from typing import Any, Iterator

import pytest

from medireach.assist.briefing import BriefingWriter
from medireach.assist.llm import (
    GeminiGenerator, TextCache, TextGenerator, cache_key, numbers_in, read_api_key, unsupported_numbers,
)
from medireach.assist.scripts import ScriptWriter, TemplateScripts
from medireach.data.database import IN_MEMORY, SqliteDatabase, read_sql_file
from medireach.errors import AIUnavailable

FACTS = {
    "today": "Wednesday 21 October 2026", "clinic_open_today": "yes", "appointments_today_booked_so_far": "3,590",
    "same_day_bookings_so_far": "648", "next_clinic_day": "Thursday 22 October",
    "appointments_known_for_next_clinic_day": "2,646", "plan_for_next_clinic_day": "approved",
    "tasks_done_today": "0 of 0", "days_measured": "0", "outcomes_used": "none yet",
    "calls_planned": "440", "sms_planned": "880", "expected_no_shows_next_clinic_day": "737",
    "share_of_expected_no_shows_on_call_list": "24.7%",
}


class StubGenerator(TextGenerator):
    """Returns a fixed reply (or raises) and records every prompt it was sent."""

    name = "stub"

    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply, self.prompts = reply, []

    def generate_json(self, system: str, prompt: str) -> tuple[dict[str, Any], str]:
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, "stub-model"


@pytest.fixture()
def cache(settings) -> Iterator[TextCache]:
    database = SqliteDatabase(IN_MEMORY)
    database.run_script("create_ops_schema", read_sql_file(settings.ops_schema_path, "read_ops_schema_file"))
    yield TextCache(database)
    database.close()


# ------------------------------------------------------------------ key

def test_key_comes_from_the_environment_first_then_the_secrets_file(tmp_path) -> None:
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('# comment\nGEMINI_API_KEY = "from-file"\n', encoding="utf-8")
    assert read_api_key({"GEMINI_API_KEY": "from-env"}, str(secrets)) == "from-env"
    assert read_api_key({}, str(secrets)) == "from-file"
    secrets.write_text('GEMINI_API_KEY = "paste-your-key-here"\n', encoding="utf-8")
    assert read_api_key({}, str(secrets)) is None
    assert read_api_key({}, str(tmp_path / "missing.toml")) is None


# --------------------------------------------------------- number check

def test_numbers_are_checked_against_the_facts() -> None:
    assert numbers_in("3,590 booked, 24.7% on the list") == {"3590", "24.7"}
    assert unsupported_numbers("440 calls and 880 SMS on Thursday 22 October; 24.7% on the list.", FACTS) == set()
    assert unsupported_numbers("About 450 calls.", FACTS) == {"450"}
    assert cache_key("x", {"a": 1, "b": 2}) == cache_key("x", {"b": 2, "a": 1}) != cache_key("y", {"a": 1, "b": 2})


# ------------------------------------------------------------- Gemini

class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _reply(text: str) -> _Response:
    return _Response(json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode())


def test_gemini_request_keeps_the_key_out_of_the_url_and_asks_for_json() -> None:
    seen = []

    def opener(request, timeout):
        seen.append((request, timeout))
        return _reply('{"headline": "h", "bullets": ["b"]}')

    content, model = GeminiGenerator("secret-key", ["model-a", "model-b"], 12.0, opener).generate_json("sys", "prompt")
    request, timeout = seen[0]
    assert content == {"headline": "h", "bullets": ["b"]} and model == "model-a" and timeout == 12.0
    assert "secret-key" not in request.full_url and request.get_header("X-goog-api-key") == "secret-key"
    body = json.loads(request.data)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_falls_back_to_the_second_model_then_gives_up() -> None:
    calls = []

    def busy_then_ok(request, timeout):
        calls.append(request.full_url)
        if "model-a" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, io.BytesIO(b""))
        return _reply('{"ok": true}')

    assert GeminiGenerator("k", ["model-a", "model-b"], 5, busy_then_ok).generate_json("s", "p")[1] == "model-b"
    assert len(calls) == 2

    def always_slow(request, timeout):
        raise TimeoutError("read timed out")

    with pytest.raises(AIUnavailable, match="TimeoutError"):
        GeminiGenerator("k", ["model-a", "model-b"], 5, always_slow).generate_json("s", "p")
    with pytest.raises(AIUnavailable, match="ValueError"):
        GeminiGenerator("k", ["model-a"], 5, lambda request, timeout: _reply('["not", "an", "object"]')).generate_json("s", "p")
    with pytest.raises(AIUnavailable):
        GeminiGenerator("", ["model-a"], 5)


# ------------------------------------------------------------- briefing

def test_template_briefing_uses_only_the_facts(cache: TextCache) -> None:
    briefing = BriefingWriter(None, cache).template(FACTS)
    assert briefing.source == "template"
    assert unsupported_numbers(" ".join([briefing.headline, *briefing.bullets]), FACTS) == set()
    assert BriefingWriter(None, cache).write(FACTS).note.startswith("AI is not set up")


def test_ai_briefing_is_checked_and_cached(cache: TextCache) -> None:
    stub = StubGenerator({"headline": "Thursday 22 October: 2,646 appointments known.", "bullets": ["440 calls planned."]})
    writer = BriefingWriter(stub, cache)
    first, second = writer.write(FACTS), writer.write(FACTS)
    assert first.source == second.source == "AI" and first.model == "stub-model"
    assert len(stub.prompts) == 1                      # the second came from the cache
    assert "3,590" in stub.prompts[0] and "patient" not in stub.prompts[0].lower()


def test_ai_briefing_with_an_invented_number_falls_back_to_the_template(cache: TextCache) -> None:
    invented = BriefingWriter(StubGenerator({"headline": "About 500 calls today.", "bullets": ["Fine."]}), cache).write(FACTS)
    assert invented.source == "template" and "500" in invented.note
    broken = BriefingWriter(StubGenerator({"headline": 3}), cache).write(FACTS)
    assert broken.source == "template" and "headline" in broken.note
    down = BriefingWriter(StubGenerator(AIUnavailable("model: HTTP 503")), cache).write(FACTS)
    assert down.source == "template" and "503" in down.note


# --------------------------------------------------------------- scripts

def test_script_polish_sends_no_patient_details_and_checks_numbers(cache: TextCache) -> None:
    good = StubGenerator({"script": "Hello, this is the Centro health unit calling about Thursday 22 October."})
    writer = ScriptWriter(good, cache, TemplateScripts())
    text, source, _ = writer.polished("Booked 15-30 days ahead", date(2026, 10, 22), "CENTRO")
    assert source == "AI" and "Centro" in text
    assert "patient_id" not in good.prompts[0] and "age" not in good.prompts[0].lower()
    invented = ScriptWriter(StubGenerator({"script": "Your 3 appointments..."}), cache, TemplateScripts())
    text, source, note = invented.polished("Booked 8-14 days ahead", date(2026, 10, 23), "CENTRO")
    assert source == "template" and "3" in note and text.startswith("Hello, this is the Centro health unit")


def test_template_scripts_are_polite_and_name_the_day() -> None:
    templates = TemplateScripts()
    script = templates.call_script("Missed a previous appointment; Booked 8-14 days ahead", date(2026, 10, 22), "JARDIM DA PENHA")
    assert "Thursday 22 October" in script and "Jardim Da Penha" in script and "missed" not in script.lower()
    assert "Reply 1 to confirm" in templates.sms_text(date(2026, 10, 22), "CENTRO", appointments=2)
