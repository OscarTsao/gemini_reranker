"""Throughput tracking and worker probing utilities."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable

from torch.utils.data import DataLoader, Dataset


class Speedometer:
    """Tracks running throughput statistics."""

    def __init__(self, window: int = 50) -> None:
        self.start_time = time.perf_counter()
        self.sample_count = 0
        self.token_count = 0
        self.step_times: deque[float] = deque(maxlen=window)

    def update(self, batch_size: int, token_count: int, step_time: float) -> None:
        self.sample_count += batch_size
        self.token_count += token_count
        self.step_times.append(step_time)

    def summary(self) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0.0
        return {
            "samples_per_sec": self.sample_count / elapsed,
            "tokens_per_sec": self.token_count / elapsed,
            "avg_step_time": avg_step,
        }


def probe_best_num_workers(
    dataset: Dataset,
    collate_fn,
    base_kwargs: dict[str, object],
    batch_size: int,
    candidates: Iterable[int] | None = None,
) -> dict[str, object]:
    """Empirically choose the best num_workers from a shortlist."""

    num_workers = int(base_kwargs.get("num_workers", 0))
    if num_workers <= 0 or len(dataset) == 0:
        return base_kwargs

    candidate_list = sorted(set(candidates or [2, 4, 6, 8]))
    if 0 not in candidate_list:
        candidate_list.insert(0, 0)
    candidate_list = [w for w in candidate_list if w <= max(8, num_workers)]
    if not candidate_list:
        return base_kwargs

    best_workers = num_workers
    best_throughput = 0.0
    for workers in candidate_list:
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=workers,
                pin_memory=base_kwargs.get("pin_memory", False),
                persistent_workers=False,
                prefetch_factor=None if workers == 0 else base_kwargs.get("prefetch_factor", 2),
            )
        except (RuntimeError, OSError, PermissionError):
            continue
        start = time.perf_counter()
        batches = 0
        try:
            for _ in loader:
                batches += 1
                if batches >= 3:
                    break
        except (RuntimeError, OSError, PermissionError):
            continue
        finally:
            del loader  # ensure worker shutdown
        elapsed = time.perf_counter() - start
        throughput = batches / elapsed if elapsed > 0 else 0.0
        if throughput > best_throughput:
            best_workers = workers
            best_throughput = throughput

    base_kwargs["num_workers"] = best_workers
    if best_throughput == 0.0:
        base_kwargs["num_workers"] = 0
        base_kwargs["persistent_workers"] = False
    return base_kwargs
