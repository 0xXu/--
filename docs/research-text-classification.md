# 新闻主题分类：现代化方案调研

调研范围：四分类英文新闻、约 10 万条有标注样本、字段类似 `Title` + `Description`。以下建议以可复现、成熟的监督学习方案为优先，并非声称某个模型在本项目隐藏测试集上必然最优。

## 结论与推荐路线

**首选：微调 `microsoft/deberta-v3-base` 的序列分类头。** 将 `Title` 与 `Description` 一起输入（例如 `title + tokenizer.sep_token + description`），保留原始文字，不做词形还原、停用词删除或手工去标点。该规模远高于 few-shot 场景，适合直接全参数微调预训练 encoder。DeBERTa-v3 以更高样本效率的 replaced-token detection 预训练，并在其论文报告的多项 NLU 基准中优于相同结构比较对象；这使其成为强而成熟的实际起点，而不是为分类任务从零训练模型。[DeBERTa-v3 论文](https://arxiv.org/abs/2111.09543)

实现采用 Hugging Face 官方的 `AutoTokenizer`、`AutoModelForSequenceClassification(num_labels=4)`、`DataCollatorWithPadding` 和 `Trainer`。官方任务指南给出了这个序列分类流程，特别指出逐 batch 动态 padding 比把整个数据集 padding 到最大长度更高效。[Transformers：文本分类](https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification)

建议起始实验（所有候选均在同一验证集上比较）：

| 项目 | 起始值 | 目的 |
| --- | --- | --- |
| 模型 | `microsoft/deberta-v3-base` | 强的通用英文 encoder；GPU 受限时切换 `distilroberta-base`。 |
| 输入 | 标题 + 分隔符 + 正文 | 原始 AG News 格式本来就含这两个文本字段；标题是高信息密度信号。 |
| 切分 | 分层 10%–15% 验证集，固定随机种子 | 四类原始数据集每类均衡，分层可保持类别比例。 |
| 最大长度 | 先统计 token 长度；起点 256 | 不要盲目截断；以覆盖绝大多数样本的长度为准。 |
| 优化 | AdamW，2–4 epoch；学习率搜索 `1e-5, 2e-5, 3e-5` | 把轮次、学习率和最大长度作为验证集实验，而不是固定真理。 |
| 选择准则 | macro-F1 为主，同时报告 accuracy 和各类 F1 | 即便类别近似平衡，也能暴露单类退化。 |

AG News 的原始语料记录说明该基准取四个最大类（World、Sports、Business、Sci/Tech），并具有 title 与 description 字段；这支持“二者均入模”和分层评估这两个决定。实际竞赛文件的行数、列名与切分仍应以本地数据画像为准。[AG’s News 语料记录](https://zenodo.org/records/7555424)

## 训练与评估护栏

1. **先处理重复，再切分。** 对规范化后的 `title + description` 精确去重；若相同文本出现多次，确认标签是否冲突，并确保同一文本不会同时进入训练和验证。否则本地分数会被泄漏抬高。
2. **保持一次不可触碰的验证集。** 所有模型、特征、学习率、长度和 ensemble 权重都只能基于训练部分及该验证集确定。最终才用选定配置在全部标注数据重训一次并预测测试集。
3. **记录可复现性。** 固定 Python/NumPy/PyTorch/Trainer 的随机种子，保存数据行数、类别分布、去重数、配置、最佳 checkpoint、验证预测和混淆矩阵。多 seed 重跑可区分真实提升与随机波动。
4. **以动态 padding 降低成本。** Transformers 官方指南明确推荐按 batch 补齐；若显存不足，用梯度累积来获得更大的有效 batch。梯度累积只解决显存约束，并不天然提高吞吐。[动态 padding 指南](https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification)；[梯度累积指南](https://huggingface.co/docs/transformers/grad_accumulation)
5. **按硬件启用混合精度。** 兼容 GPU 上使用 bf16/fp16 能减少激活/计算开销；官方说明其保留 fp32 主权重用于优化更新，以兼顾速度、内存和训练稳定性。[混合精度训练](https://huggingface.co/docs/transformers/mixed_precision_training)
6. **不要仅报 accuracy。** 官方 Evaluate 文档建议分类任务除 accuracy 外计算 precision、recall、F1，并可用 bootstrap 置信区间判断细小差异是否稳定。[Evaluate 快速指南](https://huggingface.co/docs/evaluate/main/en/a_quick_tour)

## 备选与对照

### 强而便宜的稀疏基线（必须保留）

`TfidfVectorizer`（word 1–2 gram，必要时叠加 char 3–5 gram）+ `LinearSVC` 或逻辑回归，输入同样拼接标题和正文。它训练很快，适合验证数据流水线、做 CPU fallback，并可作为 transformer ensemble 的多样化候选。它不是主力“先进模型”，但若验证分数接近 transformer，应优先检查数据泄漏、标题是否遗漏与训练设置，而不是盲目增加模型复杂度。

### 资源极紧或标签很少：SetFit

SetFit 是“先对比式微调句向量、后训练分类头”的无 prompt 方法；论文将其定位为 few-shot，报告比当时 PEFT/PET 快一个数量级。因此它适合作为少量标注、快速迭代或 CPU/GPU 预算很低时的备选，**不应替代**这里 10 万级标注数据上的全量 encoder 微调。[SetFit 论文](https://arxiv.org/abs/2209.11055)

### 更大模型与集成

验证成绩与预算允许时，再比较 `microsoft/deberta-v3-large` 或不同 seed 的 DeBERTa-base。只在独立验证预测上平均概率、确认有稳定增益后才采用集成；不要根据隐藏测试集反馈反复选择。RoBERTa 的作者也强调训练数据规模和超参数等配方会显著影响结果，因此严谨实验通常比无控制地替换模型更有价值。[RoBERTa 论文](https://arxiv.org/abs/1907.11692)

## 不建议的路线

- 不从零训练 CNN/RNN/Transformer：该任务已经有足够强的英文预训练 encoder，预训练成本与风险没有回报。
- 不把生成式大模型的零/少样本提示作为主方案：已有大量标注，推理成本、输出约束与复现性都更差。
- 不延续 Notebook 中仅 500 词的 CountVectorizer 或只用 description：它们可以留作冒烟基线，但不是合理的最终竞赛方案。

## 建议实施顺序

1. 增加数据画像命令：空值、重复、字段长度/token 长度分位数、各类别数量。
2. 增加分组去重与固定分层验证，先跑 TF-IDF 基线并写出指标/预测。
3. 加入 DeBERTa-v3-base 训练器与可配置实验参数；保存最佳 checkpoint 和验证产物。
4. 比较少量受控配置，锁定方案，再以全部标注数据重训并生成提交文件。
