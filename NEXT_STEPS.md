# GHXTox 后续工作路线

更新时间：2026-07-12

## 1. 当前结论与基线

当前默认模型保持不变：

```text
ESM2-650M residue embedding
+ ESMFold C-alpha geometry
+ learned-confidence fusion
```

- 默认 checkpoint：`runs/plm_fusion_esm2_geometry_confidence/best_model.pt`
- 默认阈值：`0.85`
- test1：MCC `0.8458`，AUPRC `0.9426`
- test2：F1 `0.6602`，MCC `0.6320`，AUPRC `0.6969`

二维路线当前最佳版本：

```text
ESM2 + 30D atom features + 3-layer Atom-MPNN
+ residue-aligned bidirectional cross-attention
```

- checkpoint：`runs/plm_sequence_atom_cross_attention/best_model.pt`
- 验证集阈值：`0.25`
- test2：F1 `0.6429`，MCC `0.6183`，AUPRC `0.6622`

已经确认不采用的方案：

- Atom-only：test2 MCC `0.3805`
- 34D扩展特征 + 四层多尺度：test2 MCC `0.5801`
- 多尺度强正则：test2 MCC `0.4924`
- ProtT5、SupCon、Focal BCE、高pLDDT子集训练和已有stacking均未超过默认模型

## 2. 实验原则

后续实验必须遵守以下规则：

1. 每次只改变一个主要变量，避免无法判断提升来源。
2. checkpoint 只根据训练集内部验证划分选择。
3. 阈值只根据训练集内部验证数据选择，禁止根据 test1/test2 调阈值。
4. test1/test2 仅用于最终一次外部评估。
5. 主要采用指标是 test2 MCC 和 F1；AUPRC/AUROC作为排序能力辅助指标。
6. 新模型只有在 test2 MCC 超过 `0.6320`，且F1无明显下降时，才可替换默认模型。
7. 同一方向连续两个清晰实验不改善 test2 MCC，即停止该方向。
8. 所有实验保留配置、checkpoint、history、metrics、predictions和验证阈值文件。

## 3. 第一阶段：拆分二维改动

状态：已完成，两个实验均未达到采用标准。

- 实验A：34D扩展特征 + 三层单尺度，test2 F1 `0.5806`、MCC `0.5443`、AUPRC `0.6828`。
- 实验B：30D基础特征 + 四层直接拼接多尺度，test2 F1 `0.6000`、MCC `0.5688`、AUPRC `0.6851`。
- 结论：扩展特征和直接拼接多尺度均停止，不再增加Atom-MPNN深度。

上一轮同时加入34维扩展特征和四层多尺度，无法确定下降来自哪一项。首先完成两个单变量消融。

### 实验 A：只测试扩展原子特征

状态：已完成，未采用。

目的：判断新增的总氢数、隐式价态、五元环和六元环是否有独立贡献。

固定项：

- ESM2 residue embedding
- 三层单尺度 Atom-MPNN
- residue-aligned bidirectional cross-attention
- 不使用标签平滑

改变项：

- 原子特征 `30D -> 34D`
- 建议 dropout `0.30`
- weight decay `0.001`
- learning rate `0.0003`

待创建配置：

```text
configs/plm_sequence_atom_cross_attention_extended_regularized.json
```

训练数据已存在：

```text
data/processed/train_cached_func_esm2_atom_multiscale.pt
data/processed/test1_cached_func_esm2_atom_multiscale.pt
data/processed/test2_cached_func_esm2_atom_multiscale.pt
```

执行步骤：

```powershell
$py = "D:\anaconda3\envs\dachuang-26\python.exe"
$env:PYTHONPATH = "src"

& $py -m ghxtox.train `
  --train data\processed\train_cached_func_esm2_atom_multiscale.pt `
  --config configs\plm_sequence_atom_cross_attention_extended_regularized.json `
  --output-dir runs\plm_sequence_atom_cross_attention_extended_regularized

& $py -m ghxtox.optimize_threshold `
  --checkpoint runs\plm_sequence_atom_cross_attention_extended_regularized\best_model.pt `
  --processed data\processed\train_cached_func_esm2_atom_multiscale.pt `
  --output runs\plm_sequence_atom_cross_attention_extended_regularized\threshold_mcc.json `
  --metric mcc
```

采用标准：

- 首先比较单尺度二维基线 MCC `0.6183`。
- 若 test2 MCC 不高于 `0.6183`，扩展特征不再进入后续二维模型。
- 若超过 `0.6183`，保留扩展特征，但仍需超过默认 `0.6320` 才能成为主模型候选。

### 实验 B：只测试多尺度编码

状态：已完成，未采用。

目的：判断性能下降是否主要来自四层多尺度，而不是扩展特征。

固定项：

- 原30维 base 原子特征
- ESM2和双向cross-attention
- 不使用标签平滑

改变项：

- 四层Atom-MPNN
- 融合第1至第4层输出

需要使用30维缓存：

```text
data/processed/train_cached_func_esm2_atom.pt
data/processed/test1_cached_func_esm2_atom.pt
data/processed/test2_cached_func_esm2_atom.pt
```

待创建配置：

```text
configs/plm_sequence_atom_cross_attention_multiscale_base.json
```

停止标准：

- 若test2 MCC不高于`0.6183`，停止直接拼接式多尺度。
- 不再继续增加Atom-MPNN深度。

## 4. 第二阶段：受控多尺度与结构化正则

状态：跳过。实验B没有超过二维基线，按停止标准不再进行门控多尺度或更深Atom-MPNN。

只有实验B证明多尺度本身有潜力时才执行本阶段。

### 门控多尺度

用加权和替代当前直接拼接：

```text
h_atom = alpha1*h1 + alpha2*h2 + alpha3*h3 + alpha4*h4
alpha = softmax(scale_logits)
```

初始化先验：

```text
[0.40, 0.30, 0.20, 0.10]
```

目标：限制深层尺度记忆训练集特定子图，同时保留局部化学模式。

### 精准正则化

不再统一提高全模型dropout，优先尝试：

- Atom feature masking：训练时随机遮蔽 `5%` 原子特征
- Modality dropout：以 `10%` 概率屏蔽原子上下文
- DropEdge：随机删除 `10%` 非肽键消息，必须保留肽键
- Scale dropout：训练时随机屏蔽一个深层尺度

每种正则单独实验，不同时全部开启。

标签平滑暂不继续使用。当前结果表明它改善概率排序，却降低了固定验证阈值在test2上的MCC。

## 5. 第三阶段：二维信息辅助默认3D模型

状态：已完成，未采用，二维主模型优化结束。

- 初始原子残差权重：`0.05`
- 最佳checkpoint学习权重：`0.0477`
- test1 MCC：`0.8241`
- test2 F1：`0.6018`
- test2 MCC：`0.5728`
- test2 AUPRC：`0.6339`
- 结论：未达到MCC、F1和test1保持标准，不再继续二维/三维联合调参。

只有第一阶段中至少一个二维模型稳定达到test2 MCC `>= 0.6183` 时才进入。

目标架构：

```text
default_3d_fused_nodes + alpha * atom_context
```

要求：

- 3D主干继续使用当前learned-confidence fusion。
- `alpha`初始化为`0.05`或`0.10`。
- 原子信息只能作为残差辅助，不能与3D等权。
- 对`alpha`增加轻微L1约束，防止模型过度依赖二维线路。
- 必须报告学习后的平均`alpha`和样本分布。

采用标准：

- test2 MCC必须超过`0.6320`。
- test2 F1不得低于`0.6500`。
- test1 MCC下降不得超过`0.01`。
- 若未达到，保留默认3D模型并结束二维主模型优化。

## 6. 第四阶段：3D结构主线优化

状态：已完成局部坐标系实验，未采用。

- 15维边特征：源/目标局部方向各3维，相对旋转矩阵9维
- 旋转等变性测试：通过
- test1 MCC：`0.7953`
- test2 F1：`0.5200`
- test2 MCC：`0.4770`
- test2 AUPRC：`0.6440`
- 结论：C-alpha轨迹不足以稳定估计完整局部姿态，停止该方向。

二维路线完成后，优先级更高的模型改进是对齐PeptiTox的局部坐标系几何特征。

建议仅做一个原则清晰的实验：

- 用连续残基构建局部坐标系
- 边方向向量投影到源残基局部坐标系
- 加入相邻局部坐标系旋转关系
- 保留距离RBF和pLDDT边权
- 不重复已经失败的简单6维`enhanced_edge_features`

采用标准仍为test2 MCC超过`0.6320`。

## 7. 阈值与数据分布分析

test2正类比例约`7.9%`，训练集约`28.5%`。需要补充验证阈值稳定性分析，但不能使用test2标签选阈值。

计划：

1. 从内部验证集构造正类比例约`8%`、`15%`和原始比例的多个子集。
2. 在各子集分别计算MCC最优阈值。
3. 选择平均MCC高且阈值方差小的候选阈值。
4. 固定该阈值后仅评估一次test1/test2。
5. 与原验证集单阈值结果并列报告。

如果不同先验下阈值差异很大，应在论文中说明模型排序能力强于固定阈值迁移能力，并重点报告AUPRC、AUROC和BACC。

## 8. 实验可信度与论文对齐

模型优化停止后完成以下工作：

### 多随机种子

- 对最终默认模型和关键消融至少运行3个种子：`42`、`123`、`2025`
- 报告均值和标准差
- 不只报告最佳种子

### 数据去冗余

- 使用CD-HIT完成`0.9`和`0.8`阈值分析
- 检查train与test之间的高相似序列
- 报告去冗余前后样本数、正负比例和性能

### 置信区间

- 对test1/test2做bootstrap
- 至少报告MCC、F1、AUROC、AUPRC的95%置信区间

### 可解释性

- 输出原子attention权重
- 按残基聚合原子权重
- 与ESM2 cross-attention权重对照
- 展示毒性预测中关键带电、芳香和含硫基团案例

## 9. 最终交付物

项目收尾时应生成：

- [x] `RESULTS.md`：所有主实验和消融结果表
- [x] `EXPERIMENT_LOG.md`：配置、命令、随机种子、阈值和结论
- [x] 最终默认checkpoint及对应config
- [x] test1/test2预测CSV
- [x] ROC、PR曲线和混淆矩阵
- [x] 多随机种子均值与标准差表
- [x] bootstrap置信区间表
- [ ] 论文方法架构图和消融实验图

## 10. 下一条立即执行的任务

模型结构优化和实验可信度阶段已经结束。以下任务已完成：

```text
1. 默认模型 seed=42/123/2025 复现实验和均值/标准差
2. test1/test2 分层层次 bootstrap 95% 置信区间
3. Biopython 序列相似性预审计和原生 CD-HIT 0.8/0.9 实验
4. CD-HIT 0.8 严格子集训练与公平评估
5. RESULTS.md、EXPERIMENT_LOG.md、ROC/PR/混淆矩阵交付
```

默认模型继续保持 `runs/plm_fusion_esm2_geometry_confidence/best_model.pt` 和阈值 `0.85`。后续不再根据当前test1/test2结果修改模型或阈值。

下一条立即执行的任务：生成论文方法架构图和消融实验图，并将 `RESULTS.md` 表格转写为论文结果章节。
