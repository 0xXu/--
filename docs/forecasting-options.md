# 面向该数据集的预测方案调研

## 结论

优先试用预训练基础模型 **Chronos-2**，但把它当作零样本 champion candidate，而不是未经验证的“必胜替换”。这个任务只有一条日频价格序列（训练集 1,826 天、测试集 182 天），而测试 CSV 没有任何未来协变量；因此，最终 182 步预测只能依赖历史 `price` 和日历特征，不能把训练集中 `volume`、市场/行业指数、利率或波动率当作未来已知输入。[训练/测试读取约定](../src/stock_forecast/data.py) [默认数据源](../src/stock_forecast/pipeline.py)

最有价值的下一步是建立**滚动回测的模型选择器**：以 `amazon/chronos-2` 的零样本预测为主候选，`amazon/chronos-bolt-small` 为更轻量的直接多步候选，并以朴素持久性预测、自动 SARIMAX 和阻尼趋势 ETS 作为必须击败的基线。Chronos-2 官方定位为最新的零样本单变量/多变量/协变量预测模型；Chronos-Bolt 则直接产生多步分位数预测。[Chronos 官方仓库](https://github.com/amazon-science/chronos-forecasting)

一次轻量的可复现实验（四个连续、各 182 天的 expanding-window 回测）已经说明这一点：现有 `ARIMA(7,1,2)` 的平均 MAE 为 **49.85**，朴素持久性预测为 **48.06**，阻尼加性趋势 ETS 为 **48.19**，局部线性趋势结构模型为 **57.60**。这不是最终选型结论（折数太少、超参数未调），但足以否定“固定 `(7,1,2)` 已经是可靠基准”的假设；数据接口见 [当前编排](../src/stock_forecast/pipeline.py)。

## 数据约束与建模含义

- 训练时间跨度为 2015-01-01 至 2019-12-31，测试跨度为 2020-01-01 至 2020-06-30；索引是日频，预测期恰为 182 天。[数据加载实现](../src/stock_forecast/data.py) [默认数据源](../src/stock_forecast/pipeline.py)
- 原始 notebook 只将 `price` 传给固定的 `ARIMA(order=(7,1,2))`，没有模型选择、残差检验、预测区间或回测；当前程序已切换为 Chronos-2。[当前编排](../src/stock_forecast/pipeline.py)
- 价格水平通常非平稳，建议同时评估 `log(price)` 的一阶差分/收益率目标和价格水平目标；提交时将预测严格逆变换回价格。是否保留价格层面的直接建模，应由同一滚动回测决定，而不是由训练集内拟合优度决定。

## 候选方案

| 优先级 | 方案 | 能解决什么 | 本数据集的判断 |
| --- | --- | --- | --- |
| 1 | Chronos-2 / Chronos-Bolt 预训练基础模型 | 零样本概率预测；Bolt 可直接多步分位数预测 | 首选试验；单序列只有 1,826 点且目标为 182 步，仍必须在同口径回测中胜出。 |
| 1 | 自动 SARIMAX / ARIMA | 自动选择差分与 AR/MA 阶数，并可加入季节项、趋势与外生变量 | 强制保留的可解释基线；必须由滚动 MAE 决定季节周期和阶数。 |
| 1 | ETS（Holt 阻尼趋势） | 用水平、趋势和可选季节项进行指数平滑 | 低方差基线；优先测 `additive + damped_trend`，不要预设季节性。 |
| 2 | 结构时间序列 / 状态空间 | 将局部水平、局部线性趋势、季节项、回归显式分解并通过 Kalman filter 估计 | 适合需要趋势随时间改变、缺失值或可解释区间的情形；本次快速回测的局部线性版本较弱，因此作为诊断 challenger。 |
| 2 | 滞后特征 + 梯度提升 | 从多个滞后、滚动统计、日历变量学习非线性 | 只允许使用预测时可得的滞后与日历特征；182 步递归误差会累积，须比较 recursive、direct 与 multi-output 策略。 |

### 1. 自动 SARIMAX（首选的可解释 challenger）

`SARIMAX` 原生支持 `(p,d,q)`、`(P,D,Q,s)`、趋势与外生回归量，也可约束 AR 平稳和 MA 可逆；因此可统一表达非季节 ARIMA、季节 ARIMA 和未来协变量可得时的动态回归模型。[statsmodels SARIMAX API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

可使用 `pmdarima.AutoARIMA` 在单位根/季节差分检验后搜索候选并以 AIC/AICc/BIC 或 OOB 指标选取模型；其文档同时明确搜索可能不收敛，非 stepwise 搜索在季节模型上会很慢。[AutoARIMA 官方 API](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.AutoARIMA.html)

实施时应将自动搜索限制为小网格（例如 `p,q≤5`、`d≤2`，候选季节周期仅包含有业务依据的 7/365），并在每个训练折内重新搜索。AIC 只能用来缩小该折的候选范围，最终排名必须按未来窗口误差。模型报告应保留 Ljung–Box 残差检验、残差 ACF 和收敛状态；statsmodels 的状态空间诊断示例也提供标准化残差、正态性、异方差和序列相关检验。残差仍有自相关或不收敛的规格不应入选。[statsmodels 状态空间诊断示例](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html)

训练中的其他列只有在能为 2020-01-01 至 2020-06-30 提供**真实可得的未来路径**时才可作为 `exog`；用测试期后才知道的值会泄漏。SARIMAX 的 `exog` 参数要求观测数 × 回归变量数的数组，且可支持时变回归系数。[statsmodels SARIMAX API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

### 2. ETS（必须保留的强基线）

statsmodels 的 `ExponentialSmoothing` 支持加性/乘性趋势、阻尼趋势与加性/乘性季节项；对于价格，先比较无季节的加性阻尼趋势与仅水平模型即可，避免把未经验证的“周季节性”硬编码进去。[statsmodels ExponentialSmoothing API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)

ETS 的价值不在于“先进”，而在于以少量参数提供稳定的长跨度趋势外推；在本项目的初步 4 折实验中，它已优于固定 ARIMA，故应成为 Chronos-2 的正式对照基线。

### 3. 结构时间序列（可解释性与不确定性）

`UnobservedComponents` 可以指定局部水平、局部线性趋势、随机/确定性趋势、季节项、周期项、AR 项和回归项，并以状态空间形式估计；它适合将“平滑趋势变化”与短期噪声分开处理。[statsmodels UnobservedComponents API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html)

优先测试 `local level`、`local linear trend` 和（仅有证据时）带 7 日季节项的版本，并使用其预测区间。快速基准中局部线性趋势最差，故不建议在未调参回测前替换主模型。[当前数据与模型入口](../src/stock_forecast/pipeline.py)

### 4. 滞后特征 + HistGradientBoosting（非线性 challenger）

scikit-learn 的官方时间序列示例展示了将 `lagged_count` 和时间相关特征输入 `HistGradientBoostingRegressor`，并明确指出随机拆分会产生过于乐观的误差估计，时间顺序拆分更符合预测场景。[scikit-learn 滞后特征示例](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)

特征应仅包括：`price` 的 1/2/3/7/14/28/56/91 天滞后、过去窗口均值/标准差/动量（全部先 `shift(1)`）、星期几/月/年内日，以及在预测时已知的节假日日历。直接多步模型要为每个预测步长单独训练；递归模型只训练一步但会反馈自身预测。两者都应在相同 182 步回测中比较，并对树深、叶子数和最小叶样本做严格正则化。

### 5. Chronos（首选零样本方案）

Chronos 将时间序列缩放并量化为 token，用语言模型架构训练；其论文报告在多种公开基准上的零样本表现。当前官方仓库列出的首选模型是 120M 参数的 `amazon/chronos-2`，轻量直接多步模型可用 48M 参数的 `amazon/chronos-bolt-small`；因此实验必须固定模型 ID、包版本、权重修订和 context window，而不能笼统比较“Chronos”。[Chronos 论文](https://arxiv.org/abs/2403.07815) [Chronos 模型与用法](https://github.com/amazon-science/chronos-forecasting)

本项目已将 `chronos-forecasting` 作为 uv 的一等依赖，并锁定在可复现的版本。官方最小用法通过 `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")` 推理；仓库同时给出 CPU/GPU 部署选项。GPU 是本项目的默认加速路径，CPU 仅作为明确可见的回退；下载模型权重、PyTorch/Transformers 依赖和版本锁定都是需要接受的代价。[Chronos 安装与 API 示例](https://github.com/amazon-science/chronos-forecasting)

此次数据没有未来协变量，所以给 Chronos-2 的 `context_df` 只能使用历史 `price`（可将历史协变量作为 past context 单独做消融），不提供 `future_df`。官方 API 明确区分历史 context 和可选的未来协变量；不能为测试期伪造后验协变量。[Chronos-2 预测接口](https://github.com/amazon-science/chronos-forecasting)

Chronos 的预训练优势来自外部训练数据而非本项目的 1,826 个样本，因此先做**冻结权重的零样本验证**，不微调；它必须与统计模型使用完全相同的 cutoff、182 步 horizon、指标和反变换。只有在多个 origins 上稳定且有意义地领先持久性、ETS 和 SARIMAX 后，才将其设为最终模型；否则不增加这条依赖链。

TimesFM 是 Google Research 维护的另一预训练时间序列基础模型，可作为第二阶段的独立零样本对照，而不应与 Chronos 同时成为第一轮工程范围。[TimesFM 官方仓库](https://github.com/google-research/timesfm) 在本数据集上没有提供优于 Chronos 或统计模型的直接证据，因此只有 Chronos 未达门槛时再纳入同一回测。

## 推荐的滚动原点评估协议

1. **冻结最终测试集。** 不读取测试期标签（目前本来就没有），也不在它上面调参。[测试集接口](../src/stock_forecast/data.py)
2. **设定真实目标 horizon = 182 天。** 从训练序列末端向前创建至少 4 个 expanding-window origins；本次可用训练样本的第 1,098、1,280、1,462、1,644 天作为 cutoffs，随后各验证 182 天。若运行成本允许，再以 28 天步长增加 origin 以降低偶然性。
3. **每折独立完成所有选择。** 变换、缺失处理、特征产生、auto-ARIMA 搜索、ETS/树模型调参都只能拟合该折训练区间；任何滚动统计必须 `shift(1)`。时间序列交叉验证的训练集应在后续折中扩展、测试集位于其后，且样本必须等间隔，这与该数据的日频索引相符。[TimeSeriesSplit 官方 API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
4. **同时报告点预测与风险。** 主指标使用 182 步总体 MAE；并报告 RMSE、相对朴素预测的 MASE、各 forecast step 的 MAE 曲线、每折结果和 80%/95% 区间覆盖率（若模型提供区间）。不要只平均一步预测误差，因为提交是一次性 182 步外推。
5. **选择规则。** 模型须在大多数 origins 胜过持久性基线，平均 MAE 改善达到预先声明的门槛（建议至少 3%），无收敛/泄漏/残差自相关红旗，才成为最终模型；否则保留更简单的基线。最终仅以全部 1,826 天重训已选规格，再生成 182 天提交预测。

## 建议的实施顺序

1. 先增加评估层（`splitter`、`metrics`、`backtest`），把当前 ARIMA 和持久性基线纳入；这是之后所有比较的共同接口。
2. 增加 ETS 和受限 auto-SARIMAX，执行上面的回测并保存每折预测、参数和诊断。
3. 只有前两者无法稳健击败基线时，再加滞后梯度提升；未来协变量可交付时再加动态回归 SARIMAX。
4. 默认以冻结权重运行 `amazon/chronos-2`；若其未达到门槛，再以 `amazon/chronos-bolt-small` 和 TimesFM 进行第二轮对照。
