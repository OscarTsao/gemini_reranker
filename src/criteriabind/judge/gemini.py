"""Gemini judge provider with JSON parsing and retries."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional

from ..schemas import JudgingJob, Judgment, Preference


class GeminiMissingDependencyError(ImportError):
    """Raised when google-generativeai dependency is unavailable."""


@dataclass
class _GeminiResponse:
    payload: dict[str, Any]
    latency_s: float
    token_usage: dict[str, int]


class GeminiJudge:
    """Wrapper around Gemini GenerativeModel enforcing JSON outputs."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        json_mode: bool,
        timeout_s: int,
        max_retries: int,
        retry_base: float,
    ) -> None:
        try:
            from google import generativeai as genai  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - only hit if dependency missing
            raise GeminiMissingDependencyError(
                "google-generativeai package is required for Gemini judge."
            ) from exc

        if not api_key:
            raise ValueError("GeminiJudge requires a non-empty API key.")
        genai.configure(api_key=api_key, client_options={"client_timeout": timeout_s})
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_output_tokens > 0:
            generation_config["max_output_tokens"] = max_output_tokens
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        self._genai = genai
        self._model = genai.GenerativeModel(model_name=model)
        self._generation_config = generation_config
        self._max_retries = max_retries
        self._retry_base = retry_base

    @staticmethod
    def _build_prompt(job: JudgingJob) -> str:
        cand_lines = [
            f"{idx}. {candidate.text.strip()}" for idx, candidate in enumerate(job.candidates)
        ]
        prompt = (
            "You are an expert clinical rater. "
            "Select the candidate text that best satisfies the diagnostic criterion. "
            "Return JSON with fields: "
            '`best_idx` (int index of the best candidate) and `preferences` (list of objects with '
            "`winner_idx` and `loser_idx`). Include a brief `rationale` string. "
            "If candidates tie, prefer the most specific supporting evidence.\n\n"
            f"Criterion: {job.criterion_text.strip()}\n"
            f"Note:\n{job.note_text.strip()}\n\n"
            "Candidates:\n" + "\n".join(cand_lines)
        )
        return prompt

    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, "text") and response.text:
            return response.text
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                chunks = []
                for part in content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        chunks.append(text)
                if chunks:
                    return "".join(chunks)
        raise ValueError("Gemini response did not contain text content.")

    @staticmethod
    def _parse_payload(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if not raw:
            raise ValueError("Empty response from Gemini.")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # Attempt to salvage JSON substring
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Failed to parse Gemini response as JSON: {raw}") from exc

    @staticmethod
    def _token_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return {}
        tokens: dict[str, int] = {}
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        if isinstance(prompt_tokens, int):
            tokens["input_tokens"] = prompt_tokens
        output_tokens = getattr(usage, "candidates_token_count", None)
        if isinstance(output_tokens, int):
            tokens["output_tokens"] = output_tokens
        return tokens

    def _call(self, prompt: str) -> _GeminiResponse:
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                start = time.perf_counter()
                response = self._model.generate_content(
                    prompt,
                    generation_config=self._generation_config,
                )
                latency_s = time.perf_counter() - start
                text = self._extract_text(response)
                payload = self._parse_payload(text)
                token_usage = self._token_usage(response)
                return _GeminiResponse(payload=payload, latency_s=latency_s, token_usage=token_usage)
            except Exception as exc:  # pragma: no cover - dependent on API behaviour
                last_error = exc
                if attempt >= self._max_retries:
                    break
                backoff = self._retry_base ** attempt
                time.sleep(min(backoff, 10.0))
        assert last_error is not None
        raise RuntimeError(f"Gemini request failed after retries: {last_error}") from last_error

    def score(self, job: JudgingJob) -> Judgment:
        prompt = self._build_prompt(job)
        response = self._call(prompt)

        payload = response.payload
        best_idx = int(payload.get("best_idx", 0))
        preferences_payload = payload.get("preferences") or []
        rationale = payload.get("rationale") or ""
        preferences: list[Preference] = []
        for pref in preferences_payload:
            try:
                preferences.append(
                    Preference(
                        winner_idx=int(pref["winner_idx"]),
                        loser_idx=int(pref["loser_idx"]),
                        weight=float(pref.get("weight", 1.0)),
                    )
                )
            except Exception:
                continue
        if not preferences and len(job.candidates) > 1:
            others = [idx for idx in range(len(job.candidates)) if idx != best_idx]
            for loser_idx in others:
                preferences.append(Preference(winner_idx=best_idx, loser_idx=loser_idx))

        model_name = getattr(self._model, "model_name", None) or getattr(
            self._model, "_model_name", "unknown"
        )
        return Judgment(
            job_id=job.job_id,
            note_id=job.note_id,
            criterion_id=job.criterion_id,
            criterion_text=job.criterion_text,
            note_text=job.note_text,
            candidates=job.candidates,
            best_idx=best_idx,
            preferences=preferences,
            rationale=rationale or "Gemini judgment rationale unavailable.",
            provider="gemini",
            model=model_name,
            latency_s=response.latency_s,
            token_usage=response.token_usage,
            meta={**job.meta, "raw_response": payload},
        )

    def batch(self, jobs: Iterable[JudgingJob]) -> list[Judgment]:
        return [self.score(job) for job in jobs]
