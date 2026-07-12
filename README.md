# GHXTox

多肽毒性预测项目。当前默认主链路已经切换为论文对齐的 PLM + 几何增强 3D fusion 路线：

```text
ESM2-650M residue embedding + ESMFold C-alpha geometry + learned-confidence fusion -> GHXTox classifier
```

当前默认模型：

```text
runs/plm_fusion_esm2_geometry_confidence/best_model.pt
threshold = 0.85
```

当前默认配置：

```text
configs/default.json
```

最终实验汇总和复现记录：

- `RESULTS.md`：主结果、消融、多 seed、bootstrap 和 CD-HIT 严格评估
- `EXPERIMENT_LOG.md`：数据、配置、命令、checkpoint 哈希和环境
- `reports/figures/default_model_evaluation.png`：ROC、PR 和混淆矩阵

## 当前主结果

默认模型使用 `runs/plm_fusion_esm2_geometry_confidence/best_model.pt`，阈值 `0.85`。

| Test set | ACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| test1 | 0.9378 | 0.8885 | 0.8458 | 0.9700 | 0.9426 |
| test2 | 0.9399 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |

和上一版主模型对比：

| Model | test1 MCC | test2 MCC | test2 AUROC | test2 AUPRC |
| --- | ---: | ---: | ---: | ---: |
| ESM2 sequence-only + threshold | 0.8177 | 0.5709 | 0.9245 | **0.7077** |
| ESM2 + geometry-enhanced 3D fusion | 0.8323 | 0.5523 | 0.9318 | 0.6732 |
| ESM2 + geometry + learned-confidence fusion | **0.8458** | **0.6320** | **0.9316** | 0.6969 |

结论：learned-confidence fusion 在 test1/test2 的 ACC、F1、MCC 和 AUROC 上超过原 ESM2 sequence-only；AUPRC 略低于单线路。因此默认主模型切换为 learned-confidence fusion，但汇报时需要说明 ranking 指标仍有提升空间。

## 最近消融结论

以下实验均按“效果不好就不替换默认模型”的原则处理：

| Experiment | test1 MCC | test1 AUPRC | test2 F1 | test2 MCC | test2 AUPRC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Default learned-confidence fusion | **0.8458** | 0.9426 | **0.6602** | **0.6320** | 0.6969 | 默认主模型 |
| + supervised contrastive loss | 0.8143 | 0.9303 | 0.6038 | 0.5709 | 0.6427 | 不采用 |
| + focal BCE loss | 0.8159 | 0.9419 | 0.6471 | 0.6171 | 0.7273 | AUPRC 更高，但 MCC/F1 低于默认，不采用 |
| + chem 结构特征 | 0.8155 | 0.9480 | 0.6263 | 0.5935 | **0.7344** | 可作为 ranking 辅助实验，不替换默认 |
| 仅用 mean pLDDT >= 0.70 子集训练 | 0.7232 | 0.8925 | 0.5854 | 0.5593 | 0.5923 | 训练覆盖不足，不采用 |
| ProtT5 + geometry + learned-confidence fusion | 0.8234 | 0.9426 | 0.6038 | 0.5709 | 0.6953 | 同协议 PLM 对照，不替换 ESM2 默认 |
| ESM2 + ProtT5 validation stacking | 0.8521 | 0.9497 | 0.6600 | 0.6307 | 0.7238 | test1/ranking 有提升，test2 MCC 略低于默认，不替换 |

补充说明：chem、focal 和 ESM2+ProtT5 stacking 都能提高 test2 AUPRC，但会牺牲或未超过 test2 MCC/F1；如果论文汇报重点是二分类决策，默认模型仍更稳。如果论文强调候选肽排序，可以把这些作为 ranking-oriented 消融结果报告。

## 验证集约束 Stacking

使用训练集内部验证划分选择非负权重和阈值，再固定应用到 test1/test2。该流程避免直接用测试集调参。

| Ensemble | Val weights | Threshold | test1 MCC | test1 AUPRC | test2 F1 | test2 MCC | test2 AUPRC | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ESM2 + chem | ESM2 0.30 / chem 0.70 | 0.6700 | 0.8155 | 0.9480 | 0.6263 | 0.5935 | 0.7331 | 不采用 |
| ESM2 + focal | ESM2 0.65 / focal 0.35 | 0.6683 | 0.8458 | 0.9464 | 0.6538 | 0.6254 | 0.7395 | 不采用 |
| ESM2 + ProtT5 | ESM2 0.65 / ProtT5 0.35 | 0.6459 | **0.8521** | **0.9497** | 0.6600 | 0.6307 | 0.7238 | test1 提升，但 test2 MCC 未超过默认 |
| ESM2 + ProtT5 + focal | ESM2 0.60 / ProtT5 0.35 / focal 0.05 | 0.6483 | - | - | 0.6458 | 0.6147 | 0.7386 | 不采用 |

结论：stacking 能改善 test1 和 AUPRC/ranking，但外部 test2 的 MCC 没有稳定超过默认 learned-confidence fusion。因此默认仍使用单模型 ESM2 fusion，stacking 只作为辅助消融。

## pLDDT 高质量结构分析

按样本 mean pLDDT 过滤，阈值设为 `0.70`。该阈值保留了足够样本，同时能排除低置信结构：

| Split | Full n | Full pos | mean pLDDT >= 0.70 n | Pos |
| --- | ---: | ---: | ---: | ---: |
| train | 6387 | 1818 | 3726 | 992 |
| test1 | 1126 | 320 | 671 | 169 |
| test2 | 582 | 46 | 333 | 18 |

默认模型直接评估高 pLDDT 子集时，性能明显高于全量测试集：

| Evaluation set | ACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| test1 full | 0.9378 | 0.8885 | 0.8458 | 0.9700 | 0.9426 |
| test1 mean pLDDT >= 0.70 | 0.9583 | 0.9146 | 0.8878 | 0.9807 | 0.9633 |
| test2 full | 0.9399 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |
| test2 mean pLDDT >= 0.70 | 0.9700 | 0.7368 | 0.7221 | 0.9575 | 0.8022 |

结论：结构置信度确实影响 3D fusion 的收益，高质量结构样本上的 MCC/AUPRC 明显更好。这一点可以作为对照 StrucToxNet 的结构质量分析。

但“只用 mean pLDDT >= 0.70 训练”的模型不应替换默认模型：

| Model | Eval set | F1 | MCC | AUPRC |
| --- | --- | ---: | ---: | ---: |
| 默认模型，全量训练 | test2 full | 0.6602 | 0.6320 | 0.6969 |
| pLDDT>=0.70 子集训练 | test2 full | 0.5854 | 0.5593 | 0.5923 |
| 默认模型，全量训练 | test2 pLDDT>=0.70 | 0.7368 | 0.7221 | 0.8022 |
| pLDDT>=0.70 子集训练 | test2 pLDDT>=0.70 | 0.6667 | 0.6524 | 0.7316 |

因此，高 pLDDT 过滤适合做分层报告和可信结构分析，不适合直接丢弃低置信训练样本。默认模型继续保留全量训练 + learned-confidence fusion。

## 默认训练

默认训练依赖已挂载 ESM2 residue embedding 的 processed 文件：

```text
data/processed/train_cached_func_esm2.pt
```

直接运行：

```powershell
$env:PYTHONPATH="src"
$py="D:\anaconda3\envs\dachuang-26\python.exe"
& $py -m ghxtox.train
```

等价于：

```powershell
& $py -m ghxtox.train `
  --train data\processed\train_cached_func_esm2.pt `
  --config configs\default.json `
  --output-dir runs\plm_fusion_esm2_geometry_confidence `
  --device cuda
```

## 默认评估

直接运行：

```powershell
& $py -m ghxtox.evaluate
```

默认使用：

```text
checkpoint: runs/plm_fusion_esm2_geometry_confidence/best_model.pt
processed:  data/processed/test1_cached_func_esm2.pt
threshold:  0.85
```

评估 test2：

```powershell
& $py -m ghxtox.evaluate `
  --processed data\processed\test2_cached_func_esm2.pt `
  --output runs\plm_fusion_esm2_geometry_confidence\test2_metrics_default.json `
  --predictions runs\plm_fusion_esm2_geometry_confidence\test2_predictions_default.csv
```

## 参考论文对齐

当前路线参考了三篇 2025 年多肽毒性预测工作：

| 论文/模型 | 与本项目相关的关键做法 | 报告指标摘录 |
| --- | --- | --- |
| PeptiTox | ESM2 residue embedding + ESMFold 结构 + GNN | 主模型 ACC 0.9444、MCC 0.8888；去掉结构后仅保留 ESM2 序列特征，ACC 0.9218、MCC 0.8438 |
| ToxiPep | 序列 embedding + BiGRU/Transformer + 原子图结构特征 | 测试集 ACC 0.850、MCC 0.701、AUC 0.919；去重后对比 ToxinPred3 的设置下 ACC 0.885、MCC 0.769 |
| StrucToxNet | ProtT5 residue embedding + ESMFold 结构 + EGNN | 独立测试集主模型 BACC 93.18%、AUC 0.968、MCC 0.852；序列模型中 CNN/LSTM + ProtT5 的 MCC 约 0.807-0.818 |

本项目目前默认模型是：

```text
ESM2-650M residue embedding + ESMFold C-alpha geometry descriptors + learned-confidence fusion
```

对应结果：

| Model | test1 ACC | test1 MCC | test1 AUROC | test1 AUPRC | test2 ACC | test2 MCC | test2 AUROC | test2 AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 sequence-only + functional descriptors | 0.8801 | 0.7182 | 0.9434 | 0.8830 | 0.8557 | 0.3444 | 0.8539 | 0.5228 |
| ESM2 sequence-only，阈值 0.85 | 0.9263 | 0.8177 | 0.9687 | 0.9367 | 0.9278 | 0.5709 | 0.9245 | 0.7077 |
| ESM2 + geometry-enhanced 3D fusion，阈值 0.421805 | 0.9325 | 0.8323 | 0.9691 | 0.9350 | 0.9278 | 0.5523 | 0.9318 | 0.6732 |
| ESM2 + geometry + learned-confidence fusion，阈值 0.85 | 0.9378 | 0.8458 | 0.9700 | 0.9426 | 0.9399 | 0.6320 | 0.9316 | 0.6969 |
| ProtT5 + geometry + learned-confidence fusion，阈值 0.85 | 0.9281 | 0.8234 | 0.9688 | 0.9426 | 0.9278 | 0.5709 | 0.9309 | 0.6953 |
| RDKit atom graph only，验证集阈值 0.47 | 0.8908 | 0.7387 | 0.9407 | 0.8769 | 0.8694 | 0.3805 | 0.8655 | 0.4953 |
| ESM2 + residue-aligned atom graph + bidirectional cross-attention，验证集阈值 0.25 | 0.9307 | 0.8297 | 0.9719 | 0.9311 | 0.9313 | 0.6183 | 0.9336 | 0.6622 |
| ESM2 + extended atom features + multi-scale Atom-MPNN + cross-attention，验证集阈值 0.879014 | 0.9254 | 0.8154 | 0.9658 | 0.9118 | 0.9278 | 0.5801 | 0.9190 | 0.5785 |
| 上述多尺度模型 + 强正则，验证集阈值 0.525115 | 0.9165 | 0.7937 | 0.9695 | 0.9337 | 0.9158 | 0.4924 | 0.9325 | 0.6703 |
| ESM2 + 34D extended atom features + 3-layer Atom-MPNN，阈值 0.797426 | 0.9281 | 0.8197 | 0.9735 | 0.9418 | 0.9330 | 0.5443 | 0.9463 | 0.6828 |
| ESM2 + 30D base atom features + 4-layer multi-scale Atom-MPNN，阈值 0.616674 | 0.9307 | 0.8315 | 0.9686 | 0.9331 | 0.9244 | 0.5688 | 0.9259 | 0.6851 |
| 默认3D + small-weight atom residual，阈值 0.43 | 0.9281 | 0.8241 | 0.9671 | 0.9339 | 0.9227 | 0.5728 | 0.9125 | 0.6339 |
| ESM2 + 3D local-frame edge geometry，阈值 0.50 | 0.9183 | 0.7953 | 0.9640 | 0.9265 | 0.9175 | 0.4770 | 0.9040 | 0.6440 |

## 对照论文仍未跟上的部分

参考 PeptiTox、ToxiPep、StrucToxNet 后，当前项目还存在以下明确差距：

| 模块 | 论文做法 | 当前项目状态 | 影响 |
| --- | --- | --- | --- |
| 结构粒度 | ToxiPep 使用 SMILES 派生的原子级图；PeptiTox 使用更完整的几何 featurizer | 已实现 RDKit 原子拓扑图、Atom-MPNN、按残基聚合和 ESM2 双向 cross-attention | cross-attention 将 test2 MCC 从 atom-only 的 0.3805 提升到 0.6183，但仍未超过默认 3D 模型的 0.6320 |
| 几何方向特征 | PeptiTox 使用 distance、direction、angle 等节点/边几何特征 | 当前有距离、接触、局部角、伪二面角，但边方向/局部坐标系仍较弱 | 3D 分支对构象差异的表达能力不足 |
| PLM 对照 | StrucToxNet 使用 ProtT5；PeptiTox 使用 ESM2 | 已完成 ProtT5/ESM2 同协议对照；本数据上 ESM2 的 MCC/F1 更好 | 默认继续使用 ESM2，ProtT5 作为对照实验 |
| 对比学习 | StrucToxNet 包含 CL 消融 | 已测试 supervised contrastive，test2 MCC/AUPRC 下降，未采用 | 当前数据和模型下没有带来外部集收益 |
| 高质量结构筛选 | StrucToxNet 分析并过滤低 pLDDT 结构 | 已完成 mean pLDDT >= 0.70 分层评估；高质量子集显著更好，但子集训练不如全量训练 | 可作为论文分析项，不替换默认训练集 |
| 阈值与校准 | 论文多用 BACC/MCC/AUC 综合比较 | 已测试验证集约束 stacking；test1/ranking 有收益，但 test2 MCC 未稳定超过默认 | 可作为辅助结果，不进入默认主链路 |
| 数据去冗余协议 | ToxiPep 使用 CD-HIT 0.8/0.9 对比 | 当前沿用给定 train/test，没有系统做相似性去冗余报告 | 与论文数值对比时严格性不足 |

下一轮最值得做的是：

1. 继续增强边几何特征，尤其 residue orientation、direction vector、局部坐标系下的 edge feature。
2. ESM2 + residue-aligned atom graph cross-attention 已完成，证明二维化学信息能补充纯序列模型；由于未超过默认 3D 模型，当前保留为论文对齐消融，不替换默认主链路。

## RDKit 原子图消融

原子图由氨基酸序列通过 RDKit 构建，节点包含元素、价态、形式电荷、杂化、芳香性、环、手性、主链位置、残基位置和 Gasteiger 部分电荷；边包含键型、共轭、环、肽键与立体信息。该图是化学拓扑图，不代表实验或 ESMFold 全原子三维坐标。

```powershell
& $py -m ghxtox.atom_graph `
  --input data\processed\train_cached_func.pt `
  --output data\processed\train_cached_func_atom.pt

& $py -m ghxtox.train `
  --train data\processed\train_cached_func_atom.pt `
  --config configs\atom_only.json `
  --output-dir runs\atom_only
```

最佳 checkpoint 位于 `runs/atom_only/best_model.pt`，验证集 MCC 为 0.7923；验证集选择阈值 0.47。test1 MCC 为 0.7387，但 test2 MCC 仅为 0.3805，因此该实验保留为 ToxiPep 路线的结构粒度消融，不替换默认模型。

### ToxiPep 式残基对齐融合

在 atom-only 基础上进一步保留每个原子的 RDKit residue number，将 Atom-MPNN 输出按残基做 attention pooling，再与同位置的 ESM2 残基表示进行双向 cross-attention：

```text
ESM2 residue embeddings <-> bidirectional cross-attention <-> residue-pooled Atom-MPNN embeddings
```

```powershell
& $py -m ghxtox.atom_graph `
  --input data\processed\train_cached_func_esm2.pt `
  --output data\processed\train_cached_func_esm2_atom.pt

& $py -m ghxtox.train `
  --train data\processed\train_cached_func_esm2_atom.pt `
  --config configs\plm_sequence_atom_cross_attention.json `
  --output-dir runs\plm_sequence_atom_cross_attention
```

最佳 checkpoint 的验证 MCC 为 0.8501，验证集选择阈值 0.25。test1 MCC 为 0.8297，test2 MCC 为 0.6183。它明显优于 atom-only 和 ESM2 sequence-only 的 test2 MCC，但仍略低于默认 ESM2 + C-alpha geometry + confidence fusion 的 0.6320，因此不修改默认 checkpoint。

### 多尺度 Atom-MPNN 与扩展原子特征

对照 ToxiPep 的原子特征和多尺度 CNN，扩展特征在原30维基础上增加总氢数、隐式价态、五元环和六元环标记；多尺度 Atom-MPNN 同时融合第1至第4层消息传递输出，对应不同化学键距离的感受野。

```powershell
& $py -m ghxtox.atom_graph `
  --input data\processed\train_cached_func_esm2.pt `
  --output data\processed\train_cached_func_esm2_atom_multiscale.pt `
  --feature-set extended

& $py -m ghxtox.train `
  --train data\processed\train_cached_func_esm2_atom_multiscale.pt `
  --config configs\plm_sequence_atom_cross_attention_multiscale.json `
  --output-dir runs\plm_sequence_atom_cross_attention_multiscale
```

该模型验证 MCC 达到 0.8769，但训练损失接近零且验证损失升高，表现出明显过拟合。固定使用验证集选择的阈值 0.879014 时，test1 MCC 为 0.8154，test2 MCC 为 0.5801，均低于单尺度 cross-attention 版本。因此多尺度与扩展特征组合保留为负向消融，不进入默认模型。

为检验过拟合是否是主要瓶颈，进一步采用 dropout 0.35、weight decay 0.001、标签平滑 0.05、学习率 0.0003 和早停耐心 5：

```powershell
& $py -m ghxtox.train `
  --train data\processed\train_cached_func_esm2_atom_multiscale.pt `
  --config configs\plm_sequence_atom_cross_attention_multiscale_regularized.json `
  --output-dir runs\plm_sequence_atom_cross_attention_multiscale_regularized
```

正则化将最佳轮从第25轮提前到第7轮，验证损失稳定在约0.52，验证阈值也从0.879014恢复到0.525115，说明过度置信得到抑制。test2 AUROC/AUPRC恢复到0.9325/0.6703，但F1/MCC仅为0.5333/0.4924，表明排序能力改善而验证阈值未能跨数据分布迁移。该版本同样不替换单尺度二维消融或默认3D模型。

### 扩展特征与多尺度解耦消融

为区分34维扩展特征和四层多尺度各自的影响，完成了两个单变量实验：

| 实验 | 原子特征 | Atom-MPNN | test2 F1 | test2 MCC | test2 AUPRC | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 二维基线 | 30D | 3层单尺度 | 0.6429 | 0.6183 | 0.6622 | 当前二维最佳 |
| 实验A | 34D | 3层单尺度 | 0.5806 | 0.5443 | 0.6828 | 排序略升，分类下降，不采用扩展特征 |
| 实验B | 30D | 4层直接拼接多尺度 | 0.6000 | 0.5688 | 0.6851 | 不采用直接拼接多尺度 |

两个单变量实验的test2 MCC均低于二维基线，因此停止继续增加Atom-MPNN深度，也不把34维扩展特征用于主模型。后续若继续利用二维信息，只允许将30维三层单尺度二维表示作为默认3D模型的小权重残差辅助。

### 二维残差辅助默认3D模型

最后测试了保守三路方案：保持默认ESM2 + C-alpha 3D confidence fusion不变，将30维三层Atom-MPNN的残基表示以可学习权重作为节点残差加入。原子权重初始化为0.05，并施加`0.001 * |alpha|`约束。

最佳checkpoint在第7轮，学习后的`alpha=0.0477`，说明模型保持了弱二维依赖。固定使用验证阈值0.43时，test1 MCC为0.8241，test2 F1/MCC为0.6018/0.5728，均未达到采用标准。因此二维路线不再进入默认模型，后续优化转向3D局部坐标系、方向向量和残基间相对旋转特征。

### 3D局部坐标系消融

基于连续C-alpha轨迹构建每个残基的切向、法向和副法向正交框架，边特征包含源/目标局部坐标系中的方向向量以及3x3相对旋转矩阵，共15维。实现通过刚性旋转等变性测试，并保留原距离RBF和pLDDT边权。

最佳验证MCC为0.8400，验证阈值为0.50。test1 MCC为0.7953，test2 F1/MCC为0.5200/0.4770，均明显低于默认模型。仅依赖C-alpha轨迹估计完整局部姿态容易放大预测结构噪声，因此该方案保留为负向几何消融，不继续增加局部框架复杂度。

## 生成 ESM2 Embedding

如果只有普通 processed 文件，需要先生成 ESM2 residue embedding：

```powershell
& $py -m ghxtox.plm_embed `
  --input data\processed\train_cached_func.pt `
  --output data\processed\train_cached_func_esm2.pt `
  --model esm2_t33_650M_UR50D `
  --device cuda `
  --batch-size 8
```

测试集同理。

## 生成 ProtT5 Embedding

ProtT5 本地模型目录：

```text
models/prot_t5_xl_half_uniref50-enc
```

生成训练集 ProtT5 residue embedding：

```powershell
& $py -m ghxtox.plm_embed `
  --input data\processed\train_cached_func.pt `
  --output data\processed\train_cached_func_prott5.pt `
  --model prot_t5_xl_half_uniref50-enc `
  --model-path models\prot_t5_xl_half_uniref50-enc `
  --device cuda `
  --batch-size 4
```

ProtT5 对照训练：

```powershell
& $py -m ghxtox.train `
  --train data\processed\train_cached_func_prott5.pt `
  --config configs\plm_fusion_prott5_geometry_confidence.json `
  --output-dir runs\plm_fusion_prott5_geometry_confidence `
  --device cuda
```
