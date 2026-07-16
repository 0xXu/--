"""GPU-aware Chronos-2 inference, isolated from the application pipeline."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

Device = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class ChronosConfig:
    """Stable inference settings for the pretrained Chronos-2 model."""

    model_id: str = "amazon/chronos-2"
    device: Device = "auto"


def _load_torch() -> object:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("Chronos-2 requires project dependencies. Run `uv sync`.") from error
    return torch


def select_device(requested_device: Device, torch: object) -> str:
    """Select CUDA when available, or fail clearly when CUDA was explicitly required."""
    cuda_available = torch.cuda.is_available()
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is unavailable to PyTorch.")
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if cuda_available else "cpu"


def configure_acceleration(torch: object, device: str) -> object | None:
    """Enable safe CUDA inference accelerations and return the selected dtype."""
    if device != "cuda":
        return None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def effective_attention_implementation(pipeline: object) -> str | None:
    """Read the backend selected by Transformers after model loading."""
    return getattr(pipeline.model.config, "_attn_implementation", None)


def acceleration_status(device: str, dtype: object | None, attention_implementation: str | None) -> str:
    """Return the effective inference configuration for the command-line user."""
    if device == "cuda":
        return f"device=cuda, dtype={dtype}, TF32=enabled, attention={attention_implementation}, inference_mode=enabled"
    return f"device=cpu, dtype=float32, TF32=disabled, attention={attention_implementation}, inference_mode=enabled"


def assert_cuda_residency(pipeline: object) -> None:
    """Reject a partially offloaded model: every parameter and buffer must be on CUDA."""
    tensors = [*pipeline.model.parameters(), *pipeline.model.buffers()]
    if not tensors or any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("Chronos-2 did not load completely onto CUDA.")


def load_pipeline(pipeline_class: object, config: ChronosConfig, device: str, dtype: object | None) -> object:
    """Load the model with Chronos' CUDA path and verify complete GPU placement."""
    pipeline = pipeline_class.from_pretrained(
        config.model_id,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    )
    if device == "cuda":
        assert_cuda_residency(pipeline)
        if effective_attention_implementation(pipeline) != "sdpa":
            raise RuntimeError("Chronos-2 did not retain SDPA attention on CUDA.")
    return pipeline


def build_context_frame(training_target: pd.Series) -> pd.DataFrame:
    """Adapt the project's date-indexed series to Chronos-2's dataframe API."""
    return pd.DataFrame(
        {
            "id": "stock-price",
            "timestamp": training_target.index,
            "target": training_target.to_numpy(),
        }
    )


def forecast_chronos(training_target: pd.Series, horizon: int, config: ChronosConfig) -> pd.Series:
    """Run zero-shot Chronos-2 inference and return its median point forecast."""
    if horizon <= 0:
        raise ValueError("Forecast horizon must be positive.")
    torch = _load_torch()
    device = select_device(config.device, torch)
    dtype = configure_acceleration(torch, device)

    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError("Chronos-2 is unavailable. Run `uv sync` to install it.") from error

    pipeline = load_pipeline(Chronos2Pipeline, config, device, dtype)
    print(f"Chronos-2 inference: {acceleration_status(device, dtype, effective_attention_implementation(pipeline))}")
    with torch.inference_mode():
        forecast = pipeline.predict_df(
            build_context_frame(training_target),
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
    if len(forecast) != horizon:
        raise RuntimeError(f"Chronos-2 returned {len(forecast)} predictions; expected {horizon}.")
    return pd.Series(forecast["predictions"].to_numpy())
