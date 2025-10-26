from __future__ import annotations

import torch
from hydra import compose, initialize

from criteriabind.config_schemas import parse_config
from criteriabind.hydra_utils import enable_speed_flags, maybe_compile, resolve_device


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.linear(x)


def test_resolve_device_cpu_fallback() -> None:
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(config_name="config", overrides=["hardware.device=auto"])
    app_cfg = parse_config(cfg)
    device, amp_dtype = resolve_device(app_cfg)
    assert device in {"cpu", "cuda"}
    if device == "cpu":
        assert amp_dtype == torch.float32


def test_maybe_compile_no_cuda() -> None:
    model = TinyNet()
    with initialize(version_base="1.3", config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=["hardware.device=cpu", "hardware.compile=true"],
        )
    app_cfg = parse_config(cfg)
    compiled = maybe_compile(model, app_cfg)
    # CPU path should return original module instance
    assert compiled is model
    enable_speed_flags(app_cfg)
