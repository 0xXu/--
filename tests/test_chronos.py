from types import SimpleNamespace

import pandas as pd
import pytest

from stock_forecast.chronos import ChronosConfig, build_context_frame, configure_acceleration, load_pipeline, select_device
from stock_forecast.cli import build_parser


class FakeCuda:
    def __init__(self, available: bool, bfloat16_supported: bool = True) -> None:
        self.available = available
        self.bfloat16_supported = bfloat16_supported

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bfloat16_supported


class FakeTorch:
    def __init__(self, cuda_available: bool) -> None:
        self.cuda = FakeCuda(cuda_available)


class FakeMatmul:
    allow_tf32 = False


class FakeCudnn:
    allow_tf32 = False
    benchmark = False


class FakeAcceleratedTorch(FakeTorch):
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self) -> None:
        super().__init__(cuda_available=True)
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=FakeMatmul()), cudnn=FakeCudnn())
        self.matmul_precision = None

    def set_float32_matmul_precision(self, precision: str) -> None:
        self.matmul_precision = precision


class FakePipeline:
    class Model:
        config = SimpleNamespace(_attn_implementation="sdpa")

        @staticmethod
        def parameters():
            return [SimpleNamespace(device=SimpleNamespace(type="cuda"))]

        @staticmethod
        def buffers():
            return []

    model = Model()
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append((model_id, kwargs))
        return cls()


def test_build_context_frame_preserves_dates_and_values() -> None:
    series = pd.Series([10.0, 12.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))

    frame = build_context_frame(series)

    assert frame.columns.tolist() == ["id", "timestamp", "target"]
    assert frame["target"].tolist() == [10.0, 12.0]
    assert frame["timestamp"].tolist() == series.index.tolist()


def test_select_device_uses_cuda_when_available() -> None:
    assert select_device("auto", FakeTorch(cuda_available=True)) == "cuda"


def test_select_device_rejects_required_unavailable_cuda() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda", FakeTorch(cuda_available=False))


def test_configure_acceleration_enables_cuda_fast_paths() -> None:
    torch = FakeAcceleratedTorch()

    dtype = configure_acceleration(torch, "cuda")

    assert dtype == "bf16"
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cudnn.benchmark
    assert torch.matmul_precision == "high"


def test_configure_acceleration_falls_back_to_fp16_without_bfloat16_support() -> None:
    torch = FakeAcceleratedTorch()
    torch.cuda.bfloat16_supported = False

    assert configure_acceleration(torch, "cuda") == "fp16"


def test_load_pipeline_uses_sdpa_and_asserts_cuda_placement() -> None:
    FakePipeline.calls.clear()

    load_pipeline(FakePipeline, ChronosConfig(), "cuda", "bf16")

    assert FakePipeline.calls == [
        (
            "amazon/chronos-2",
            {"device_map": "cuda", "torch_dtype": "bf16", "attn_implementation": "sdpa", "local_files_only": False},
        )
    ]


def test_cli_uses_automatic_cuda_selection_by_default() -> None:
    arguments = build_parser().parse_args([])

    assert arguments.device == "auto"
