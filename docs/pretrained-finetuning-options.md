# 单序列价格预测：预训练模型 + 本地微调调研（2026-07）

## 结论

可以做“预训练后本地训练”，但**不能把它等同于一定优于持久性基线**。本数据只有一条日频 `price` 序列、1,826 个观测、一次性预测 182 天；滑窗会制造很多高度相关的训练样本，却不会创造新的市场状态。上一轮 Chronos-2 零样本在同口径 4 折回测中显著落后持久性基线，因此下一轮应是一个受严格约束的微调实验，而不是继续扩大模型规模。

首选实施顺序是：

1. **Chronos-2 LoRA**：现有依赖已官方支持，120M 参数，接口直接支持 `fit(..., finetune_mode="lora")`；是改动最小、最可审计的主试验。
2. **IBM TinyTimeMixer (TTM) r2.1**：这是更适合本硬件和样本量的独立 challenger。它约 0.8M 参数，官方模型卡明确支持日频、微调和最长 720 步的匹配模型；比再试一个数亿参数 Transformer 更合理。
3. **TimesFM 2.5 200M LoRA**：官方仓库已给出 Transformers + PEFT LoRA 示例，可作为第三个独立架构。只在前两者没有达到门槛时纳入，避免在极少数据上“挑冠军”。

不推荐在本任务优先投入 Moirai/Moirai-MoE 或 Time-MoE：前者的官方 Moirai-1 微调流程存在但偏重 Hydra/GluonTS 训练栈，Moirai-MoE 文档重点是推理；后者 README 仍把补齐微调/动态特征支持列为 TODO。它们不是不能研究，而是不如前三者具备低风险、可复现的路径。

## 候选模型对比

| 候选 | 官方微调支持 | RTX 4060 8GB 可行性 | 对本数据的结论 |
| --- | --- | --- | --- |
| **Chronos-2 120M + LoRA** | `Chronos2Pipeline.fit` 明确提供 `full` / `lora`；默认 LoRA 为 r=8，并默认指向注意力 Q/K/V/O 与输出层。官方建议 LoRA 学习率约 `1e-5`。 [官方 pipeline 源码](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) | 可行：120M BF16 权重约 0.24GB；LoRA 仅训练小量适配器。8GB 下从 `batch_size=1`、BF16、`context_length=512` 开始，梯度累积 8；实际以首个 outer fold 的峰值显存为准。全量微调虽由 API 支持，但本数据上既更易过拟合也没有必要。Chronos-2 官方列为 120M 参数。 [官方模型表](https://github.com/amazon-science/chronos-forecasting#available-models) | **首选**。当前项目已经有该模型和 CUDA 路径；只需新增训练/评估层。 |
| **TTM r2.1（匹配日频的 512/1024 context、192 horizon 分支）** | IBM 官方仓库提供 TTM 的预训练和 finetuning notebook；模型卡明确区分 zero-shot 与 finetuned forecasting，支持用目标数据微调。 [官方仓库](https://github.com/ibm-granite/granite-tsfm) [官方模型卡](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) | **最稳妥**：模型卡主分支约 805k 参数、约 3.24MB；官方称 zero-shot、微调和推理可在一张 GPU 或笔记本上执行。r2.1 还明确加入日频/周频。无需量化或 LoRA；全量微调也远低于 8GB。 | **最值得新增的 challenger**。选择能直接输出 192 的 r2/r2.1 分支，设置 `prediction_filter_length=182`，不要递归两段预测。官方 `get_model()` 可按 context/prediction length 选模型，并说明 r2 可到 720、固定长度包括 192。 [模型卡说明](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) |
| **TimesFM 2.5 200M + LoRA** | 官方仓库在 2026-04 加入 Hugging Face Transformers + PEFT (LoRA) 微调示例；TimesFM 2.5 为 200M 参数。 [官方仓库及示例入口](https://github.com/google-research/timesfm) | 有条件可行：以 BF16、batch 1、context 512 或 1024、gradient accumulation、checkpointing 为起点；不得假定 8GB 一定容纳全部激活/优化器状态。LoRA 而非全量微调；首次运行记录 `torch.cuda.max_memory_allocated()`，OOM 就降 context，**不**更换成未经验证的量化训练。 | **第三顺位**。能提供独立的预训练归纳偏置，但参数更大，且单一短序列很容易在 LoRA 阶段过拟合。其最大 16k context、可选量化头最长 1k horizon 是能力上限，不是此任务应当使用的上下文长度。 [官方 README](https://github.com/google-research/timesfm) |
| **Moirai-1.1-R small / Moirai-MoE small** | Uni2TS 是统一的预训练、微调、推理框架。其官方 Moirai-1 微调文档提供滑窗、训练统计标准化、全量/`head_only`/`freeze_ffn` 三种方式，并给出 `5e-7` 小学习率。 [官方微调文档](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-1/finetune_lsf) | Moirai small 可尝试，但官方未给本机 8GB 的保证；从 batch 1、context 512、head-only 开始才可控。Moirai-MoE small 为 117M 总参数（11M activated），base 为 935M 总参数（86M activated）；base 不适合 8GB。 [官方 MoE 模型表](https://github.com/SalesforceAIResearch/uni2ts/tree/main/project/moirai-moe-1) | **备选，不进第一轮**。Moirai-1 有正式微调资料；对 Moirai-MoE，当前官方 README 展示预训练推理但未给同等级微调 recipe，故不能把“框架可能可训”当作已验证支持。 |
| **Time-MoE** | 不作为本次微调候选：官方 README 的 TODO 仍列出“启用 Time-MoE 微调以支持动态特征和分类”。 [官方 README](https://github.com/Time-MoE/Time-MoE) | 推理可选 base/large，README 推荐 FlashAttention 以降低显存；但这并不构成该单序列的官方 PEFT 微调方案。模型最大 2.4B 参数，超出本任务需要。 [官方 README](https://github.com/Time-MoE/Time-MoE) | **排除**。没有成熟、低风险的本地微调入口，工程风险大于潜在收益。 |

## Chronos-2 LoRA：应采用的首轮规格

这是实验规格，而不是用测试集调出来的结果：

- 输入只用历史 `price`；`volume` 等列只有在每一个 182 天预测窗口开始时就能取得未来真值时才可进入 `future_covariates`。否则一律不使用，避免泄漏。Chronos-2 的官方接口明确区分历史 context 与可选 future covariates。 [官方预测示例](https://github.com/amazon-science/chronos-forecasting#forecasting)
- 每个训练折把**标签完全落在 cutoff 之前**的滑窗作为训练样本：`context_length=512`、`prediction_length=182`、步长 7；不够长时 context 再降到 256 做一个预先声明的消融。严禁一个窗口的 182 天标签跨越 cutoff。
- `finetune_mode="lora"`，固定 `r=4`、`lora_alpha=8`、`lora_dropout=0.05`（官方默认 r=8/alpha=16 可作为第二个、而非无限扩展的候选）；`learning_rate=1e-5`、BF16、batch 1、梯度累积 8、最大 200 steps、每 10 steps 在内部验证集评估、patience 3。官方代码的 LoRA 默认 target modules 与推荐学习率见上表链接。
- 验证集必须来自当前 outer-train 区间最后的 182 天；训练窗口标签不能接触这 182 天。按验证 MAE 早停，随后只在该 outer-train 内按固定步骤重训后预测 outer-test。若没有足够窗口，宁愿减少 steps / 用 256 context，也不合并 outer-test。
- 只记录并保存 adapter（含模型 revision、随机种子、训练/验证 cutoff、峰值显存、最优 step）；最终选定后才用全 1,826 天的同一训练规程重训一次。不要把四个回测折训练出的 adapter 用于最终提交。

## 严格的 182 天滚动回测与决策规则

已有结果说明“跑通 GPU”不是胜利条件。以下规则在试验前冻结：

1. **外层切分**：训练数据第 1,098 / 1,280 / 1,462 / 1,644 天为 cutoffs；每个 outer-test 是随后的 182 天。每折模型、归一化器、LoRA adapter 都从头开始，不能复用后折信息。
2. **调参不看 outer-test**：在每个 outer-train 内，以其末尾 182 天做早停/规格选择，且所有训练窗口标签均早于这段内部验证；候选仅为上节的 `context={256,512}` × `r={4,8}`，最多四个。TTM 只比较预训练冻结与全量微调两个预先固定规格；TimesFM 只一组 LoRA。不要按 4 个 outer-test 再扩张网格。
3. **比较对象一致**：所有候选、持久性、ETS、ARIMA 都从同一 cutoff 预测完整 182 步。报告每折 MAE、RMSE、平均 MAE 和相对持久性误差；不以训练 loss、一步误差或图形平滑度选模型。
4. **选择门槛**：候选必须同时满足 `(a)` 四折平均 MAE 比持久性至少低 3%，`(b)` 至少 3/4 折 MAE 获胜，`(c)` 最接近提交期的第 4 折也至少低 3%，`(d)` 不 OOM、不使用未来列、没有训练发散。否则认定“该微调没有证据优于基线”。
5. **避免多重试验自欺**：先用前三个 outer folds 选一个最终规格；第 4 折只做一次锁定的确认。即使前三折很好而第 4 折不达标，也不以事后修改门槛补救。最终提交仅以完整 1,826 天重训获胜规格一次，输出 182 天。

这条流程会给出有价值的阴性结论：若 Chronos LoRA、TTM 和 TimesFM 都过不了同一门槛，应提交基线/统计组合，而不是相信更大的预训练模型会从一条短价格序列中学到不可验证的规律。

## 推荐的工程落地顺序（本文件不改代码）

1. 为现有 backtest 保留缓存数据、持久性/ETS/ARIMA，新增“滑窗构造、内部验证、adapter 隔离、每折指标与预测落盘”。
2. 先实现 Chronos-2 LoRA 的唯一首轮网格，运行四折；显存不足只调 batch/context/accumulation，不改变评价协议。
3. 若未过门槛，增加 TTM r2.1 的直接 182（用 192 模型裁剪）冻结 + 微调对照；TTM 是下一步，而非更大的 MoE。
4. 仅当前两者仍不达标时增加 TimesFM 2.5 LoRA；完成后封存候选表和版本锁，不再以测试期表现挑选。

