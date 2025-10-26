"""Utilities shared across Hydra entrypoints for seeding and hardware configuration."""

from __future__ import annotations

import contextlib
import logging
import os
import random

import numpy as np
import torch

from .config_schemas import AppConfig


LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""

    random.seed(seed)
    np.random.default_rng(seed)
    np.random.seed(seed)  # noqa: NPY002 - maintain legacy RNG behaviour
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where it does not have a high performance penalty.
    torch.use_deterministic_algorithms(False)


def resolve_device(cfg: AppConfig) -> tuple[str, torch.dtype]:
    """Resolve the compute device and automatic mixed precision dtype."""

    choice = cfg.hardware.device

    def _cuda_supported() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major < 7:  # torch 2.9 requires Ampere+ kernels by default
                LOGGER.warning(
                    "Detected CUDA capability %s which is not supported by this build; using CPU.",
                    major,
                )
                return False
        except Exception as exc:  # pragma: no cover - env specific
            LOGGER.warning("CUDA capability detection failed: %s. Falling back to CPU.", exc)
            return False
        return True

    if choice == "auto":
        device = "cuda" if _cuda_supported() else "cpu"
    elif choice == "cuda" and not _cuda_supported():
        LOGGER.warning("CUDA requested but unavailable or unsupported; falling back to CPU.")
        device = "cpu"
    else:
        device = choice

    amp_pref = (cfg.hardware.amp_dtype or "none").lower()
    if device != "cuda":
        return device, torch.float32

    if amp_pref == "none":
        return device, torch.float32

    capability = torch.cuda.get_device_capability()
    major, minor = capability
    supports_bf16 = major >= 8  # Ampere+
    torch_fp16_ok = True  # All CUDA devices with torch >= 1.6

    if amp_pref == "bf16":
        if supports_bf16:
            return device, torch.bfloat16
        LOGGER.info("bf16 requested but unsupported on this GPU; falling back to fp16.")
        amp_pref = "fp16"

    if amp_pref == "fp16" and torch_fp16_ok:
        return device, torch.float16

    LOGGER.info("Falling back to float32 precision.")
    return device, torch.float32


def auto_workers(cfg: AppConfig) -> dict[str, int | bool]:
    """Infer DataLoader worker configuration from hardware config."""

    num_workers = cfg.hardware.num_workers
    if isinstance(num_workers, str) and num_workers.lower() == "auto":
        cpu_count = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_count // 2))
    num_workers = int(num_workers)
    pin_memory = bool(cfg.hardware.pin_memory and torch.cuda.is_available())
    persistent_workers = bool(cfg.hardware.persistent_workers and num_workers > 0)
    prefetch_factor = int(cfg.hardware.prefetch_factor)
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def enable_speed_flags(cfg: AppConfig) -> None:
    """Configure Torch backend knobs for performance when safe."""

    if torch.cuda.is_available():
        tf32_allowed = bool(cfg.hardware.tf32)
        torch.backends.cuda.matmul.allow_tf32 = tf32_allowed
        torch.backends.cudnn.allow_tf32 = tf32_allowed
        torch.backends.cudnn.benchmark = bool(cfg.hardware.cudnn_benchmark)
        if tf32_allowed:
            with contextlib.suppress(RuntimeError):
                torch.set_float32_matmul_precision("high")
    else:
        # CPU path - leave defaults alone to avoid regressions.
        pass


def maybe_compile(model: torch.nn.Module, cfg: AppConfig) -> torch.nn.Module:
    """Apply torch.compile when requested and supported."""

    should_compile = bool(cfg.hardware.compile)
    if not should_compile:
        return model

    if not hasattr(torch, "compile"):
        LOGGER.info("torch.compile unavailable on this version of PyTorch.")
        return model

    device = next(model.parameters()).device
    if device.type != "cuda":
        LOGGER.info("torch.compile requested but running on CPU - skipping.")
        return model

    try:
        compiled = torch.compile(model, mode="max-autotune", fullgraph=False)
        LOGGER.info("Enabled torch.compile with max-autotune mode.")
        return compiled
    except Exception as exc:  # pragma: no cover - depends on backend support
        LOGGER.warning("torch.compile failed: %s. Continuing without compilation.", exc)
        return model
