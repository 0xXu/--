# Stock Forecast

将原本的 Jupyter Notebook 转换为可测试、可复用的 Python 命令行程序。程序读取训练集与测试集 CSV，以预训练的 Chronos-2 模型预测测试时间段的 `price`，并生成提交文件。

## 安装与运行

使用 [uv](https://docs.astral.sh/uv/) 管理环境和依赖：

```bash
uv sync --dev
uv run stock-forecast
```

默认使用原 Notebook 中的数据地址，结果写入当前目录的 `submission.csv`。也可以指定本地文件或其他地址：

```bash
uv run stock-forecast \
  --train-path data/train.csv \
  --test-path data/test.csv \
  --output predictions/submission.csv \
  --plot predictions/forecast.png
```

Chronos-2 默认在可用时使用 CUDA；CUDA 环境会启用 BF16、TF32、SDPA 注意力以及 PyTorch inference mode。若没有 CUDA，程序会自动使用 CPU，并在输出中说明实际设备。可通过 `--device cuda` 强制要求 GPU，或通过 `--device cpu` 调试：

```bash
uv run stock-forecast --device cuda
```

在 Windows + NVIDIA GPU 上，请使用 Python 3.12；`uv` 会自动创建它。项目已配置 PyTorch 的 CUDA 12.8 官方 wheel 源，RTX 4060 的 CUDA 13.3 驱动可向后兼容。

训练集与测试集的第一列必须是可解析的日期索引；训练集必须含有 `price` 列。测试集只需提供日期索引（即使含有 `price` 列，也不会作为输入值使用）。

## 开发

```bash
uv run pytest
```
