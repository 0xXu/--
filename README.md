# Stock Forecast

将原本的 Jupyter Notebook 转换为可测试、可复用的 Python 命令行程序。程序读取训练集与测试集 CSV，以预训练的 Chronos-2 模型预测测试时间段的 `price`，并生成提交文件。

## 安装与运行

使用 [uv](https://docs.astral.sh/uv/) 管理环境和依赖：

```bash
uv sync --dev
uv run stock-forecast
```

默认使用原 Notebook 中的数据地址，并采用比赛式的“校准 + 确认”流程：前 3 个 182 天滚动折学习非负集成权重，最近一折只用于确认。候选覆盖不同历史窗口的 ETS、对数差分 ETS、两日周期、相似历史片段和持久性预测；只有集成在确认折以至少 3% 优于持久性基线才会用于提交。结果写入当前目录的 `submission.csv`。也可以指定本地文件或其他地址：

```bash
uv run stock-forecast \
  --train-path data/train.csv \
  --test-path data/test.csv \
  --output predictions/submission.csv \
  --plot predictions/forecast.png
```

用 `--selection-output` 保存候选排名、每折 MAE/RMSE/MASE；同目录还会生成 `*-step-mae.csv`、`*-oof.csv` 和 `*-ensemble.json`，分别记录逐预测步误差、OOF 预测和锁定的集成权重。`--model auto --include-chronos` 会将 Chronos-2 也放入同一回测门槛；需要强制生成其预测时才使用 `--model chronos`。Chronos 不再是未经回测验证的默认提交路径。

远程 CSV 会在首次成功下载后缓存至 `.cache/stock-forecast/`；之后回测和预测均直接使用缓存。可用 `--cache-dir` 指定其他缓存位置。

Chronos-2 默认在可用时使用 CUDA；CUDA 环境会启用 BF16、TF32、SDPA 注意力以及 PyTorch inference mode。若没有 CUDA，程序会自动使用 CPU，并在输出中说明实际设备。可通过 `--device cuda` 强制要求 GPU，或通过 `--device cpu` 调试：

```bash
uv run stock-forecast --device cuda
```

在采用提交结果前，先执行与比赛预测期一致的滚动回测；它会比较 Chronos-2、持久性基线、阻尼趋势 ETS 与原始 ARIMA：

```bash
uv run stock-backtest --device cuda --output backtest-results.csv
```

可用下列命令将 Chronos-2 LoRA 作为独立候选加入回测。每一折都会从基础权重重新训练 adapter，且内部验证标签严格早于该折的 182 天测试期：

```bash
uv run stock-backtest --device cuda --include-lora --lora-steps 200 --output lora-backtest-results.csv
```

仅当 Chronos-2 在多数折中稳定领先时，才应将其作为最终提交模型。

回测会强制从本地 Hugging Face 模型缓存加载 Chronos-2，避免每一折都重复访问网络；请先成功运行一次 `stock-forecast` 完成模型权重下载。

在 Windows + NVIDIA GPU 上，请使用 Python 3.12；`uv` 会自动创建它。项目已配置 PyTorch 的 CUDA 12.8 官方 wheel 源，RTX 4060 的 CUDA 13.3 驱动可向后兼容。

训练集与测试集的第一列必须是可解析的日期索引；训练集必须含有 `price` 列。测试集只需提供日期索引（即使含有 `price` 列，也不会作为输入值使用）。

## 开发

```bash
uv run pytest
```
