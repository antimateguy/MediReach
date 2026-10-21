"""TextGenerator: one small interface for AI text, a Gemini (free tier) implementation, a cache, number checks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from medireach.data.database import SqliteDatabase
from medireach.errors import AIUnavailable

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
KEY_NAME = "GEMINI_API_KEY"
PLACEHOLDER = "paste-your-key-here"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class TextGenerator(ABC):
    """Anything that turns a system instruction and a prompt into a JSON object."""

    name: str

    @abstractmethod
    def generate_json(self, system: str, prompt: str) -> tuple[dict[str, Any], str]:
        """(parsed JSON object, model that answered). Raises AIUnavailable on any failure."""


class GeminiGenerator(TextGenerator):
    """Gemini's generateContent over HTTPS, standard library only.

    The key travels in the x-goog-api-key header, never in the URL, so it cannot
    end up in a log line. If the first model is busy or fails, the fallback model
    is tried once.
    """

    name = "Gemini"

    def __init__(self, api_key: str, models: Sequence[str], timeout: float,
                 opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        if not api_key:
            raise AIUnavailable("no Gemini API key")
        self._key, self._models, self._timeout, self._open = api_key, tuple(m for m in models if m), timeout, opener

    def generate_json(self, system: str, prompt: str) -> tuple[dict[str, Any], str]:
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }).encode("utf-8")
        problems = []
        for model in self._models:
            request = urllib.request.Request(GEMINI_URL.format(model=model), data=body, method="POST",
                                             headers={"x-goog-api-key": self._key, "Content-Type": "application/json"})
            try:
                with self._open(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read())
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                return _parse_object(text), model
            except urllib.error.HTTPError as exc:
                problems.append(f"{model}: HTTP {exc.code}")
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, ValueError) as exc:
                problems.append(f"{model}: {type(exc).__name__}")
        raise AIUnavailable("; ".join(problems) or "no model configured")


def _parse_object(text: str) -> dict[str, Any]:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("the reply is not a JSON object")
    return parsed


def read_api_key(environment: Mapping[str, str], secrets_path: str) -> str | None:
    """The Gemini key from the environment, else from a `GEMINI_API_KEY = "..."` line in secrets.toml."""
    key = environment.get(KEY_NAME, "").strip()
    if not key and os.path.exists(secrets_path):
        with open(secrets_path, "r", encoding="utf-8") as handle:
            match = re.search(rf'^\s*{KEY_NAME}\s*=\s*"([^"]*)"', handle.read(), re.MULTILINE)
        key = match.group(1).strip() if match else ""
    return key if key and key != PLACEHOLDER else None


def numbers_in(text: str) -> set[str]:
    """Every number in a text, normalised (thousands separators removed)."""
    return {match.replace(",", "") for match in _NUMBER.findall(text)}


def unsupported_numbers(text: str, facts: Mapping[str, str]) -> set[str]:
    """Numbers in AI text that do not appear in the facts it was given. Empty means the check passed."""
    allowed = set()
    for value in facts.values():
        allowed |= numbers_in(str(value))
    return numbers_in(text) - allowed


def cache_key(feature: str, payload: Mapping[str, Any]) -> str:
    """Stable key for one request: same inputs, same key."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"{feature}:{digest[:32]}"


class TextCache:
    """AI replies kept in ops.db (table llm_outputs), so each distinct request is made once."""

    def __init__(self, database: SqliteDatabase) -> None:
        self._database = database

    def get(self, key: str) -> tuple[dict[str, Any], str] | None:
        frame = self._database.query("select_llm_output", "SELECT model, content FROM llm_outputs WHERE cache_key = ?", (key,))
        return None if frame.empty else (json.loads(frame.iloc[0]["content"]), str(frame.iloc[0]["model"]))

    def put(self, key: str, feature: str, model: str, content: dict[str, Any]) -> None:
        sql = """
            INSERT INTO llm_outputs (cache_key, feature, model, content, created_at) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (cache_key) DO UPDATE SET model = excluded.model, content = excluded.content
        """
        row = (key, feature, model, json.dumps(content, sort_keys=True), datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._database.write("upsert_llm_output", [(sql, [row])])
