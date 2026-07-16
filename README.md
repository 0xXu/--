# Stock Forecast

将原本的 Jupyter Notebook 转换为可测试、可复用的 Python 命令行程序。程序读取训练集与测试集 CSV，以 ARIMA 模型预测测试时间段的 `price`，并生成提交文件。

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

训练集与测试集的第一列必须是可解析的日期索引；训练集必须含有 `price` 列。测试集只需提供日期索引（即使含有 `price` 列，也不会作为输入值使用）。

## 开发

```bash
uv run pytest
```
