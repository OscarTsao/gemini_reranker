"""Deterministic mock for Gemini API judging.

This module provides a mock implementation of Gemini judging that uses
keyword overlap scoring instead of calling the actual API. This is useful
for testing, CI/CD pipelines, and development without API costs.
"""

from __future__ import annotations

import re

from criteriabind.schemas import Candidate, Judgment, Preference


def _normalize_text(text: str) -> str:
    """Normalize text for keyword comparison.

    Args:
        text: Input text to normalize.

    Returns:
        Lowercased text with only alphanumeric characters and spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_keywords(text: str) -> set[str]:
    """Extract keywords from text (words longer than 2 characters).

    Args:
        text: Input text to extract keywords from.

    Returns:
        Set of keywords (words > 2 chars).
    """
    normalized = _normalize_text(text)
    words = normalized.split()
    return {w for w in words if len(w) > 2}


def _compute_overlap_score(criterion_keywords: set[str], candidate_keywords: set[str]) -> float:
    """Compute overlap score between criterion and candidate keywords.

    Args:
        criterion_keywords: Set of keywords from the criterion.
        candidate_keywords: Set of keywords from the candidate.

    Returns:
        Jaccard similarity score (intersection / union).
    """
    if not criterion_keywords or not candidate_keywords:
        return 0.0
    intersection = len(criterion_keywords & candidate_keywords)
    union = len(criterion_keywords | candidate_keywords)
    return intersection / union if union > 0 else 0.0


def score_candidates(criterion: str, candidates: list[Candidate]) -> Judgment:
    """Score candidates deterministically based on keyword overlap with criterion.

    This function provides a deterministic alternative to Gemini API judging.
    It scores candidates by computing keyword overlap (Jaccard similarity)
    between the criterion and each candidate text.

    Args:
        criterion: The criterion text to match against.
        candidates: List of candidate snippets to score.

    Returns:
        JudgeResult with winner_index, rank, rationales, safety, and rubric_version.

    Raises:
        ValueError: If candidates list is empty.
    """
    if not candidates:
        raise ValueError("Cannot score empty candidate list")

    criterion_keywords = _extract_keywords(criterion)

    # Compute scores for all candidates
    scores: list[tuple[int, float]] = []
    for idx, candidate in enumerate(candidates):
        candidate_keywords = _extract_keywords(candidate.text)
        score = _compute_overlap_score(criterion_keywords, candidate_keywords)
        scores.append((idx, score))

    # Sort by score (descending), then by index (ascending) for determinism
    scores.sort(key=lambda x: (-x[1], x[0]))

    # Extract winner and rank
    winner_index = scores[0][0]
    rank = [idx for idx, _ in scores]

    # Generate rationale
    winner_score = scores[0][1]
    winner_keywords = _extract_keywords(candidates[winner_index].text)
    overlap_keywords = criterion_keywords & winner_keywords

    if overlap_keywords:
        keyword_sample = ", ".join(sorted(list(overlap_keywords))[:5])
        rationales = (
            f"Candidate {winner_index} has the best keyword overlap (score: {winner_score:.3f}). "
            f"Shared keywords include: {keyword_sample}."
        )
    else:
        rationales = (
            f"Candidate {winner_index} selected as best match (score: {winner_score:.3f}). "
            "Mock evaluation based on keyword overlap heuristic."
        )

    # Mock safety - no concerns for deterministic scoring
    safety = {
        "flags": [],
        "notes": "Mock evaluation mode. No actual safety analysis performed.",
    }

    preferences = [
        Preference(winner_idx=rank[i], loser_idx=rank[j])
        for i in range(len(rank))
        for j in range(i + 1, len(rank))
    ]

    return Judgment(
        job_id="mock|job",
        note_id="mock|note",
        criterion_id="mock|criterion",
        criterion_text=criterion,
        note_text="mock note",
        candidates=candidates,
        best_idx=winner_index,
        preferences=preferences,
        rationale=rationales,
        provider="mock",
        model="mock_gemini",
        meta={"safety": safety, "rank": rank},
    )


if __name__ == "__main__":
    # Quick test of the mock scoring
    from criteriabind.schemas import Candidate

    criterion = "Patient shows symptoms of depression including low mood and loss of interest"
    test_candidates = [
        Candidate(text="Patient reports feeling happy and energetic"),
        Candidate(text="Patient has low mood and shows loss of interest in activities"),
        Candidate(text="Blood pressure is normal"),
    ]

    result = score_candidates(criterion, test_candidates)
    print(f"Winner: Candidate {result.winner_index}")
    print(f"Rank: {result.rank}")
    print(f"Rationale: {result.rationales}")
    print(f"Safety: {result.safety}")
