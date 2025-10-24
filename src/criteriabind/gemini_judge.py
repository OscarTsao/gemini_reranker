"""Gemini judging interface."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tyro

from google import genai
from google.genai import types

from .config import JudgeConfig
from .io_utils import read_jsonl, write_jsonl
from .logging_utils import get_logger
from .schemas import JudgeResult, JudgedItem, JudgingJob

LOGGER = get_logger(__name__)

RUBRIC = (
    "You are a clinical adjudicator. Given a criterion and N candidate snippets from the same note, "
    "pick the single best snippet that is (1) most faithful to the note, "
    "(2) most directly supports or refutes the criterion, (3) complete and clear, (4) safest. "
    "Output strictly JSON with fields: winner_index (int), rank (array of indices best→worst, "
    "a permutation of [0..N-1]), rationales (≤80 words), safety.flags (list), safety.notes (string). "
    "If two are close, pick the safer, more faithful one."
)

SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "winner_index": types.Schema(type=types.Type.INTEGER),
        "rank": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.INTEGER)),
        "rationales": types.Schema(type=types.Type.STRING),
        "safety": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "flags": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "notes": types.Schema(type=types.Type.STRING),
            },
            required=["flags", "notes"],
        ),
        "rubric_version": types.Schema(type=types.Type.STRING),
    },
    required=["winner_index", "rank", "rationales", "safety", "rubric_version"],
)


def _build_config(cfg: JudgeConfig) -> types.GenerateContentConfig:
    def _threshold(key: str) -> types.HarmBlockThreshold:
        name = cfg.safety.get(key, "BLOCK_MEDIUM_AND_ABOVE")
        return getattr(types.HarmBlockThreshold, name)

    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=_threshold("hate_speech"),
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=_threshold("harassment"),
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=_threshold("sexual"),
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=_threshold("danger"),
        ),
    ]
    return types.GenerateContentConfig(
        system_instruction=RUBRIC,
        response_mime_type=cfg.response_mime,
        response_schema=SCHEMA,
        safety_settings=safety_settings,
    )


def _format_prompt(job: JudgingJob, variant: int) -> str:
    header = f"Criterion: {job.criterion}\n"
    lines = [header, "Candidates:"]
    for idx, candidate in enumerate(job.candidates):
        lines.append(f"[{idx}] {candidate.text}")
    if variant == 1:
        lines.append("Select the best snippet. Provide a deterministic ranking.")
    elif variant == 2:
        lines.append("Re-evaluate carefully. If tied, choose the safer snippet.")
    else:
        lines.append("Final tie-breaker: choose the snippet with highest support and safety.")
    return "\n".join(lines)


class GeminiJudgeError(RuntimeError):
    """Raised when Gemini judging fails."""


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(GeminiJudgeError),
)
def _call_model(
    client: genai.Client,
    model: str,
    config: types.GenerateContentConfig,
    prompt: str,
) -> JudgeResult:
    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(prompt)])],
            config=config,
        )
    except Exception as exc:  # pragma: no cover - network path
        LOGGER.error("Gemini API call failed: %s", exc)
        raise GeminiJudgeError(str(exc)) from exc
    try:
        text = response.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        raise GeminiJudgeError("Malformed response from Gemini") from exc
    try:
        return JudgeResult.model_validate_json(text)
    except Exception as exc:
        LOGGER.error("Failed to parse JudgeResult: %s", text)
        raise GeminiJudgeError("Unable to parse JSON response") from exc


def judge_job(client: genai.Client, job: JudgingJob, cfg: JudgeConfig) -> Optional[JudgedItem]:
    if not job.candidates:
        LOGGER.warning("Skipping job %s with no candidates", job.id)
        return None
    config = _build_config(cfg)
    result_primary = _call_model(client, cfg.model, config, _format_prompt(job, 1))
    result_secondary = _call_model(client, cfg.model, config, _format_prompt(job, 2))
    result = result_primary
    if (result_primary.winner_index != result_secondary.winner_index or result_primary.rank != result_secondary.rank) and cfg.enable_two_pass:
        LOGGER.info("Two-pass disagreement for job %s, running tie-break", job.id)
        tie_break = _call_model(client, cfg.model, config, _format_prompt(job, 3))
        if (
            tie_break.winner_index == result_primary.winner_index
            and tie_break.rank == result_primary.rank
        ):
            result = result_primary
        elif (
            tie_break.winner_index == result_secondary.winner_index
            and tie_break.rank == result_secondary.rank
        ):
            result = result_secondary
        else:
            LOGGER.warning("Tie-break unresolved for job %s, dropping item", job.id)
            if cfg.drop_on_conflict:
                return None
            result = tie_break

    flags = result.safety.get("flags", [])
    if flags:
        LOGGER.warning("Dropping job %s due to safety flags: %s", job.id, flags)
        return None

    result = result.model_copy(update={"rubric_version": cfg.rubric_version})
    judged = JudgedItem(
        id=job.id,
        note_id=job.note_id,
        criterion=job.criterion,
        candidates=job.candidates,
        judge=result,
    )
    return judged


@dataclass
class JudgeArgs:
    """CLI arguments for Gemini judging."""

    in_path: Path
    out_path: Path
    model: str = "gemini-2.5-flash"
    rubric_version: str = "clinical_rubric_v1"
    drop_on_conflict: bool = True
    mock: bool = False


def main(args: JudgeArgs) -> None:
    jobs = [JudgingJob(**row) for row in read_jsonl(args.in_path)]
    LOGGER.info("Loaded %d judging jobs", len(jobs))

    if args.mock:
        LOGGER.info("Running in MOCK mode - using deterministic scoring")
        # Import mock_gemini module
        import sys
        from pathlib import Path as PathLib

        scripts_dir = PathLib(__file__).parent.parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from mock_gemini import score_candidates  # type: ignore[import-not-found]

        judged_items: List[JudgedItem] = []
        for job in jobs:
            if not job.candidates:
                LOGGER.warning("Skipping job %s with no candidates", job.id)
                continue
            try:
                result = score_candidates(job.criterion, job.candidates)
                result = result.model_copy(update={"rubric_version": args.rubric_version})
                judged = JudgedItem(
                    id=job.id,
                    note_id=job.note_id,
                    criterion=job.criterion,
                    candidates=job.candidates,
                    judge=result,
                )
                judged_items.append(judged)
            except Exception as exc:
                LOGGER.error("Failed to judge job %s: %s", job.id, exc)
                continue
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY must be set")

        client = genai.Client()
        judge_cfg = JudgeConfig(model=args.model, rubric_version=args.rubric_version, drop_on_conflict=args.drop_on_conflict)
        judged_items = []
        for job in jobs:
            try:
                judged = judge_job(client, job, judge_cfg)
            except GeminiJudgeError as exc:
                LOGGER.error("Failed to judge job %s: %s", job.id, exc)
                continue
            if judged:
                judged_items.append(judged)

    write_jsonl(args.out_path, judged_items)
    LOGGER.info("Saved %d judged items to %s", len(judged_items), args.out_path)


if __name__ == "__main__":
    tyro.cli(main)
