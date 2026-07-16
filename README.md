# GHXTox：多维度肽毒性预测、结果与生物学解释

更新时间：2026-07-15
状态：冻结主模型、严格外部评估和残基解释框架均已完成

## 阅读导航与当前结论

- 历史 Test1/Test2 上，pLDDT-aware 3D learned-confidence fusion 是冻结默认决策模型。
- 严格 ToxinPred3 外部集上，1D ESM2-BiLSTM 的 MCC/F1/AUROC/AUPRC 均为三条线路最佳。
- 2D 原子图保留为化学解释和消融专家；三线路路由保留为可靠度工具，不替换默认分类器。
- 当前四样本残基解释只证明解释链路可运行，不能代表通用氨基酸毒性规律。
- 本文件是项目唯一 Markdown 说明，模型、结果、进度、命令和论文表述边界均在此维护。

## 摘要

GHXTox 是一个面向肽毒性二分类的多维度建模项目。项目在导师论文所用相同基础数据框架上，分别建立并比较三条信息线路：

1. **1D 序列线路**：学习氨基酸顺序、上下文语义和显式理化属性；
2. **2D 原子图线路**：学习原子、化学键及其与残基上下文之间的对应关系；
3. **3D 结构线路**：学习 ESMFold 预测结构中的空间邻域、紧致性和非局部接触，并依据 pLDDT 对不可靠结构信息降权。

当前冻结默认分类器是 **ESM2 + 序列 Transformer + pLDDT-aware EGNN + learned-confidence fusion** 的 3D 融合模型。它在历史 Test1/Test2 上具有当前最好的决策指标，但在严格 ToxinPred3 外部集上，**不依赖结构的 1D ESM2-BiLSTM 迁移最好**。因此，本项目最合理的结论不是“3D 在所有数据上都优于 1D”，而是：多维信息在同域评估中具有互补性；预测结构的收益受结构质量和分布偏移影响；1D 线路应作为正式的无结构回退模型保留。

本文档把结论分为三个证据等级：

- **已实现并验证**：可以由源码、配置、冻结指标或输出文件直接确认；
- **模型支持的生物学解释**：说明模型使用了哪些与生物化学相关的信号，但不等同于实验因果机制；
- **待验证假说**：需要扩大解释样本、独立数据或湿实验后才能形成生物学结论。

---

## 1. 任务、数据与证据边界

### 1.1 任务定义

输入为线性、标准氨基酸字母表示的肽序列，输出为毒性概率：

\[
p(y=1\mid x)=\sigma(z),
\]

其中 \(x\) 为肽序列及可用的 2D/3D 表征，\(z\) 为分类器 logit，\(y=1\) 表示毒性肽。当前任务是**总体毒性二分类**，不是毒性类型、靶器官、作用机制或剂量-反应预测。

### 1.2 数据组成

当前基础数据与导师论文 ToxMSRC 对应：

| 数据集 | 毒性肽 | 非毒性肽 | 总数 | 当前用途 |
| --- | ---: | ---: | ---: | --- |
| Train | 1818 | 4569 | 6387 | 模型训练与组感知验证 |
| Test1 | 320 | 806 | 1126 | 历史对照 |
| Test2 | 46 | 536 | 582 | 历史对照，正样本很少 |
| ToxinPred3 strict external | 471 | 506 | 977 | 冻结外部迁移评估 |

ToxinPred3 严格外部集经过精确去重和相对当前 Train/Test1/Test2 的 `>=0.8` 同源筛查；原生 CD-HIT 4.8.1 复核确认当前 977 条是更保守的子集。**这些外部样本没有加入训练集，模型参数和阈值也没有根据其标签重新调整。**

### 1.3 评估边界

Test1/Test2 在历史开发中已被多次查看，因此可用于复现和横向历史比较，但不再适合支持新的架构选择或无偏创新主张。新的模型筛选主要依赖组间同源隔离的 nested group CV；外部泛化主张主要依赖冻结的 ToxinPred3 strict 评估。

Test2 只有 46 个正样本，少量 TP/FN 的变化即可明显改变召回率、F1 和 MCC。因此，Test2 的小数点后微小差异不能脱离多随机种子方差和置信区间解释。

### 1.4 ToxinPred3 在本项目中的作用、增益与预测支持

#### 1.4.1 核心定位

ToxinPred3 在 GHXTox 中是**严格低同源外部压力测试集**，不是训练数据、模型输入分支或集成专家。它用于回答：当模型、checkpoint 和阈值全部冻结后，1D、2D、3D 三条线路面对另一个数据库来源的肽序列时，是否仍能保持排序、分类和概率可靠性。

| 作用 | ToxinPred3 是否承担 | 具体说明 |
| --- | --- | --- |
| 扩充训练集 | 否 | 471 条毒性肽和 506 条非毒性肽均未加入训练 |
| 模型微调 | 否 | 不更新任何神经网络参数 |
| checkpoint/架构选择 | 否 | 评估前已经冻结 1D、2D、3D 模型 |
| 阈值选择 | 否 | 继续使用 1D `0.50`、2D `0.25`、3D `0.85` |
| 外部泛化验证 | 是 | 检查跨数据库和低同源条件下的性能 |
| 外部概率校准检查 | 是 | 报告 Brier 和 ECE-10，但不据此重新校准 |
| 三线路迁移比较 | 是 | 比较序列、原子拓扑和预测结构的跨域稳定性 |
| 部署策略支持 | 是 | 为保留 1D 回退、报告结构质量和处理线路冲突提供证据 |
| 生物学解释外部复核 | 可以 | 仅能在冻结模型上做 TP/TN/FP/FN 分层，不可反向调参 |

ToxinPred3 不会出现在单条样本的前向计算中。模型预测仍然只依赖该样本的序列、ESM2 表征以及相应线路需要的原子图或 ESMFold 结构。ToxinPred3 不提供额外特征、相似样本投票或概率加成，所以它带来的不是“模型数值性能增益”，而是“结果证据和使用策略增益”。

#### 1.4.2 严格集构建和独立性

项目使用 ToxinPred3 官方独立测试来源，先相对当前 Train、Test1、Test2 的完整序列集合执行精确去重，再删除序列一致性 `>=0.8` 的同源样本。筛查后保留 977 条：

| 类别 | 数量 |
| --- | ---: |
| 毒性肽 | 471 |
| 非毒性肽 | 506 |
| 总计 | 977 |

原生 CD-HIT 4.8.1 复核保留 987 条，当前 977 条全部包含在原生结果中；另外 10 条仅被原生方法保留。因此，当前集合是一个更保守的、已经过原生 CD-HIT 验证的外部子集。ToxinPred3 的训练文件只参与重叠审计，不进入 GHXTox 训练。

关键文件：

- `data/external/toxinpred3/strict/strict.fasta`：严格外部序列；
- `data/external/toxinpred3/strict/strict_manifest.csv`：样本、标签与筛查清单；
- `data/external/toxinpred3/strict/audit_summary.json`：去重和同源审计摘要；
- `data/external/toxinpred3/cdhit_audit/native_cdhit_audit.json`：原生 CD-HIT 复核；
- `data/external/toxinpred3/processed/strict_esm2_structure.pt`：冻结 3D 评估数据；
- `runs/external_toxinpred3_1d/`、`runs/external_toxinpred3_2d/`、`runs/external_toxinpred3_3d/`：预测与指标。

#### 1.4.3 对预测结果的直接证据

三条线路均使用训练/验证阶段已经确定的固定阈值，一次性外部评估结果为：

| 路线 | BACC | Precision | Recall | F1 | MCC | AUROC | AUPRC | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1D ESM2-BiLSTM | **0.7780** | 0.8885 | **0.6299** | **0.7369** | **0.5859** | **0.8266** | **0.8673** | **0.1745** | **0.1301** |
| 2D ESM2 + Atom-MPNN | 0.7540 | 0.8911 | 0.5732 | 0.6977 | 0.5488 | 0.7786 | 0.8373 | 0.2160 | 0.2031 |
| 3D pLDDT-aware 默认模型 | 0.7484 | **0.9162** | 0.5428 | 0.6815 | 0.5495 | 0.7629 | 0.8271 | 0.2260 | 0.2208 |

这些结果对预测结论提供了四层支持：

1. **支持 1D 的跨域稳定性。** 1D 的 MCC、F1、AUROC、AUPRC、Brier 和 ECE 均最好，说明不依赖预测结构的 ESM2-BiLSTM 在该外部域上迁移更稳健。
2. **限定 3D 增益的适用范围。** 3D 在历史 Test1/Test2 上是默认决策模型，但其外部 MCC/AUPRC 没有超过 1D，说明结构信息的同域收益没有完整迁移到 ToxinPred3。
3. **揭示 3D 的保守预测倾向。** 3D precision 为 0.9162、recall 为 0.5428：一旦报毒通常较可靠，但约 45.7% 的外部毒性肽未达到 0.85 阈值。
4. **揭示概率可靠度偏移。** 2D/3D 的 Brier 和 ECE 明显高于 1D，说明复杂线路在新数据域中的概率置信度发生偏移，不能把域内校准直接当作域外可靠度。

因此，ToxinPred3 不支持“3D 对所有数据都最好”，而支持更精确的结论：**3D 是历史同域默认决策模型，1D 是当前严格外部域中最稳定的迁移和无结构回退模型。**

#### 1.4.4 ToxinPred3 带来的项目增益

ToxinPred3 没有改变模型参数，所以不能把它描述为训练增益或准确率提升。它带来的实际增益是：

- **独立性增益**：补足 Test1/Test2 已被历史开发多次观察的不足；
- **同源控制增益**：降低因重复或高相似序列造成的性能高估；
- **模型定位增益**：把“默认 3D”修正为条件性最优，而不是普遍最优；
- **部署增益**：为保留 1D 正式回退、结构低可信时降低 3D 依赖提供实证依据；
- **风险识别增益**：识别 3D 高 precision/低 recall 的外部漏检风险；
- **校准证据增益**：证明 OOF 或历史测试域的可靠度不能自动迁移到外部域；
- **论文可信度增益**：可以同时报告正结果、外部退化和模型适用边界，减少选择性报告；
- **解释验证增益**：允许检查残基热点和错误模式是否跨数据源稳定，而不是只解释同域样本。

#### 1.4.5 对实际预测使用的支持方式

ToxinPred3 结果支持以下预测策略，但不直接改变单条样本概率：

| 使用场景 | 建议 |
| --- | --- |
| 有完整且较高可信的 ESMFold 缓存、样本接近训练域 | 使用 3D 默认模型，同时报告 pLDDT |
| 无结构、结构生成失败或平均 pLDDT 较低 | 使用 1D ESM2-BiLSTM 回退 |
| 未知数据库或明显分布偏移 | 同时报告 1D/3D 概率、pLDDT 和跨种子不确定性 |
| 1D 与 3D 判断一致 | 作为跨线路一致性支持，但仍不是实验真实性证明 |
| 1D 与 3D 判断冲突 | 标记为高风险样本，不应因 3D 更复杂就自动采信 3D |
| 3D 高概率报毒 | 外部高 precision 提供一定支持，但不能外推为精确的个体后验概率 |
| 3D 低于 0.85 | 不能直接判定安全，因为外部 recall 只有 0.5428，应参考 1D 和人工复核 |

ToxinPred3 最重要的预测支持不是提高某一条预测，而是帮助解释“该预测在什么条件下更可信、什么时候应回退或谨慎”。

#### 1.4.6 后续使用边界

ToxinPred3 的标签和结果现已被查看，因此它不再能用于下一轮无偏模型选择。后续允许：

- 作为冻结结果的最终报告集；
- 进行不改变模型的 TP/TN/FP/FN 错误分析；
- 进行冻结残基解释和跨数据集稳定性分析；
- 报告 bootstrap 区间、校准和线路差异。

后续禁止：

- 根据 ToxinPred3 调整 1D/2D/3D 阈值；
- 反复修改架构直到该集合性能提高；
- 使用其标签训练路由器、校准器或残差专家；
- 把其样本加入当前训练集后仍称为独立外部验证；
- 仅凭这一个外部来源宣称跨数据库普遍泛化。

新的泛化或校准创新需要第二个独立数据来源验证。

---

## 2. 三维度总体框架

```text
肽序列
  |
  +-- 共同残基输入
  |     +-- 氨基酸类别嵌入（32D）
  |     +-- 官能团类别嵌入（12D）
  |     +-- 显式残基理化描述符（42D）
  |     +-- ESM2-650M 上下文残基向量（1280D -> 128D）
  |     +-- 全局特征：长度、净电荷、芳香族比例、Cys 比例、平均疏水性
  |
  +-- 1D：BiLSTM -> mean/max pooling -> MLP
  |
  +-- 2D：序列 Transformer <-> Atom-MPNN 残基对齐双向跨注意力
  |        -> mean pooling -> MLP
  |
  +-- 3D：序列 Transformer + pLDDT-aware EGNN
           -> 残基级门控 + 全局结构门控 + 跨注意力
           -> mean pooling -> MLP
```

三个维度不是三个完全独立的“生物机制模型”。它们共享序列和 ESM2 信息，再分别增加原子拓扑或预测空间几何。尤其是当前 3D 空间分支也接收 ESM2 表征，因此准确名称应为 **PLM-enhanced geometric branch**，不能称为纯几何模型。

共同的 42 维显式残基特征覆盖：疏水性、形式化电荷类别、极性、芳香性、含硫性、正/负电类别、转角倾向、相对分子量、长度为 5 的局部窗口电荷/疏水性/官能团比例、相对位置、N/C 端倾向、脂肪族/支链/小残基类别、G/P/C/M 标记、羟基/酰胺/羧酸盐/伯胺/胍基/咪唑/酚/吲哚等官能团，以及由序列构造的理化关联图社区编号、社区比例和加权度。它们使模型可以直接访问可解释的理化信号，但这些人工描述符仍是序列派生量，不是实验测量值。

---

## 3. 1D 序列线路：从残基语法到整体毒性倾向

### 3.1 实际保留的架构

冻结 1D 配置见 [`configs/sequence_1d_bilstm.json`](configs/sequence_1d_bilstm.json)。实际保留架构为：

```text
[AA embedding 32D;
 functional-group embedding 12D;
 residue descriptors 42D;
 projected ESM2 128D]
             |
       Linear -> 128D
             |
     positional embedding
             |
   1-layer bidirectional LSTM
   (64D forward + 64D backward)
             |
 masked mean pooling + masked max pooling
             |
      256D peptide vector
             + 5D global features
             |
        MLP: 261 -> 128 -> 64 -> 1
```

虽然代码支持核大小 3/5/7 的多尺度 CNN 和残差连接，但冻结配置中 `sequence_use_multiscale=false`、`sequence_use_residual=false`，仅保留 BiLSTM。原因是多尺度 CNN 路线没有稳定超过 ESM2-BiLSTM，不能把“代码中存在的可选模块”写成最终模型组成。

训练采用类别自动加权的 BCE，不使用 SMOTE。模型固定阈值为 `0.50`，用于当前 1D 冻结评估。

### 3.2 每一层可对应的生物学含义

#### 氨基酸与官能团嵌入

氨基酸身份决定侧链大小、极性、电离状态、芳香性和构象倾向。官能团类别嵌入进一步把化学性质相近的残基放在可学习空间中。例如 Lys/Arg 的正电性质、Asp/Glu 的负电性质、Phe/Tyr/Trp 的芳香性、Cys/Met 的含硫属性可被模型显式区分。

这条支路支持的解释是：模型能够学习“哪些残基类型及其组合与训练标签相关”。它不直接证明某个残基通过某种受体或膜破坏机制导致毒性。

#### 42 维理化描述符

显式电荷和局部电荷密度可以表示阳离子簇；疏水性和芳香族比例可以表示可能参与膜分配或疏水核心形成的序列片段；Cys 可提示潜在二硫键稳定性；Gly/Pro 与局部柔性和转角倾向相关；N/C 端位置特征允许模型区分相同残基出现在肽链不同位置时的统计作用。

这些关系是合理的生物化学先验，但模型只看到序列派生描述符。比如“正电且疏水”可以与膜活性相关，却不能据此断言样本一定通过细胞膜裂解致毒。

#### ESM2 上下文表示

ESM2-650M 为每个残基提供 1280 维上下文向量，再投影到 128 维。该向量不是单纯的氨基酸类别，而是依赖整条序列的上下文表示，因此同一个 Lys 出现在不同基序、不同端部和不同长程背景时可得到不同表示。

它可以补充人工特征难以穷举的高阶序列模式，例如：

- 带电残基与疏水残基的间隔和组合；
- Cys 的上下文及可能的成对模式；
- 局部基序与远端残基的共同出现；
- 类似已知肽家族的统计“序列语法”。

但 ESM2 表示来自大规模蛋白序列预训练，不能直接等价为毒性机制知识。

#### BiLSTM 与双向依赖

前向 LSTM 按 N 端到 C 端整合信息，后向 LSTM按 C 端到 N 端整合信息。双向输出允许一个残基的表示同时受其上游和下游序列影响，适合描述端部效应、局部片段与远端背景之间的顺序依赖。

BiLSTM 的生物学意义不是模拟肽的物理动力学，而是学习**有方向的序列统计依赖**。它比简单残基计数更能区分组成相近但排列不同的肽。

#### mean/max pooling 与全局描述符

mean pooling 概括全序列的平均模式，max pooling 捕捉最强的局部激活。因此，1D 模型可以同时利用“整体组成倾向”和“少数强基序”。长度、净电荷、芳香族比例、Cys 比例和平均疏水性作为 5 个全局特征直接进入分类器，减少模型必须从局部向量重新推导这些总量的负担。

### 3.3 1D 线路能够与不能够解释什么

**可以解释：**

- 哪些序列位置对某一预测起促进或抑制作用；
- 模型是否依赖带电、疏水、芳香、含硫或端部等序列信号；
- 不同随机种子是否对关键位置形成一致判断；
- 无结构场景下，序列信息能否稳定迁移。

**不能直接解释：**

- 肽在溶液中的真实构象；
- 残基之间是否形成真实空间接触或氢键；
- 二硫键具体配对、质子化微状态和膜结合姿态；
- 具体毒性靶点、剂量和实验因果机制。

### 3.4 当前证据

1D 三随机种子结果为：

| 数据集 | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test1 | 0.8768 ± 0.0039 | 0.8285 ± 0.0036 | 0.9709 ± 0.0021 | 0.9432 ± 0.0032 |
| Test2 | 0.6051 ± 0.0182 | 0.5773 ± 0.0222 | 0.9361 ± 0.0122 | 0.6965 ± 0.0178 |
| ToxinPred3 strict | 0.7369 | **0.5859** | **0.8266** | **0.8673** |

在严格外部集上，1D 的 MCC、F1、AUROC 和 AUPRC 均高于当前 2D/3D 路线。这说明当结构质量或数据域发生变化时，ESM2-BiLSTM 学到的序列信号更稳健。该结果支持把 1D 定义为正式部署回退线路，而不只是“去掉结构的消融”。

---

## 4. 2D 原子图线路：从化学拓扑到序列—原子对齐

### 4.1 实际保留的架构

冻结 2D 配置见 [`configs/plm_sequence_atom_cross_attention.json`](configs/plm_sequence_atom_cross_attention.json)。架构由两个分支和双向跨注意力组成：

```text
序列分支：
AA/group/42D/ESM2 -> 128D -> 2-layer Transformer (4 heads)
                              |
                              | sequence queries atom residues
                              | atom residues query sequence
                              v
原子分支：
RDKit MolFromSequence
 -> atom features 30D + bond features 11D
 -> 3-layer residual Atom-MPNN (hidden 128D)
 -> attention pooling of atoms within each residue
 -> residue-aligned atom representation
                              |
                  concatenate two 128D residue views
                              |
                       masked mean pooling
                              |
                   + 5D global features -> MLP
```

序列 Transformer 使用两层、四头自注意力，以学习全局上下文。原子图由 RDKit 根据标准线性肽序列确定性构建。每个化学键被展开为两个方向的消息传递边。

### 4.2 原子和化学键具体表示

基础原子节点为 30 维，包含：

- 元素类型：C、N、O、S、P 或其他；
- 原子度数和形式电荷；
- `sp`、`sp2`、`sp3` 或其他杂化类型；
- 芳香性、环属性和手性标签；
- 主链原子身份：N、CA、C、O 或侧链；
- 所属残基的归一化位置；
- Gasteiger 部分电荷。

化学键边为 11 维，包含：

- 单键、双键、三键、芳香键；
- 共轭和成环属性；
- 是否为相邻残基间的肽键；
- 键立体信息。

三层 Atom-MPNN 让每个原子逐层接收邻近原子和化学键信息。随后在同一残基内部进行注意力聚合，把数量不等的原子压缩成与序列位置一一对应的残基级原子表示。

### 4.3 生物学解释线路

#### 局部化学环境

原子图比“氨基酸字母”更直接地表达羰基、胺基、硫原子、芳香体系、形式/部分电荷和共轭结构。消息传递可把单个原子属性与其邻域结合，从而区别相同元素处于不同化学环境中的含义。

因此，2D 模型可以支持以下层面的解释：

- 哪些残基的原子组成和局部键环境与预测相关；
- 模型是否偏向带电、芳香、含硫或特殊主链环境；
- 原子化学表征是否为单纯序列表征提供互补信息。

#### 残基内原子注意力池化

同一残基的不同原子对功能贡献并不等价。注意力池化允许模型对主链、侧链或特定杂原子赋予不同权重，然后将其对齐到对应残基位置。这一步建立了“原子化学—残基序列”的桥梁。

注意力权重反映模型在当前前向计算中如何聚合信息，但不等价于因果重要性。论文级解释应与遮蔽、反事实突变或梯度归因联合使用，不能只展示注意力热图。

#### 双向跨注意力

序列查询原子表示回答“当前上下文残基需要读取哪些局部化学信息”；原子表示查询序列回答“当前化学环境应结合哪些序列背景理解”。双向对齐使模型可以区分：

- 化学组成相似、但序列上下文不同的残基；
- 序列位置相似、但侧链原子和键环境不同的残基；
- 局部原子属性与长程序列语义共同出现的模式。

### 4.4 2D 线路的边界

当前原子图是 `RDKit Chem.MolFromSequence` 产生的**确定性二维共价拓扑图**，不是实验结构，也不是分子动力学构象。因此它不能表示：

- 溶剂、离子、pH 和真实质子化微状态；
- 非共价空间接触、氢键距离和膜结合构象；
- 未在输入中编码的环肽、非天然氨基酸和翻译后修饰；
- 真实二硫键配对及多构象动态。

### 4.5 当前证据

最佳 2D 单检查点 ESM2 + Atom-MPNN cross-attention 的结果为：

| 数据集 | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test1 | 0.8781 | 0.8297 | 0.9719 | 0.9311 |
| Test2 | 0.6429 | 0.6183 | 0.9336 | 0.6622 |
| ToxinPred3 strict | 0.6977 | 0.5488 | 0.7786 | 0.8373 |

2D 明显优于 atom-graph-only 分支，但没有超过历史默认 3D；在严格外部集上也未超过 1D。它证明原子拓扑有互补信号，却不足以支持“原子图必然改善跨域泛化”的结论。因此当前把它保留为化学解释和消融专家，不重新开启已完成的二维扩展消融。

---

## 5. 3D 结构线路：从空间邻域到质量感知融合

### 5.1 实际保留的架构

冻结默认配置见 [`configs/default.json`](configs/default.json)：

```text
序列分支
AA/group/42D/ESM2 -> 128D -> 2-layer Transformer, 4 heads
                                            |
                                            | query
                                            v
空间分支                              pLDDT-aware cross-attention
42D/16D geometry/pLDDT/ESM2                 ^
 -> hybrid graph                            | gated key/value
 -> 3 pLDDT-aware EGNN layers               |
 -> spatial residue representations --------+
                                            |
                node gate × global graph gate
                                            |
       concatenate(sequence, gated spatial context)
                                            |
                  masked mean pooling
                                            |
                 + 5D global features
                                            |
                    MLP -> logit
```

序列分支与 2D 路线相同，使用 ESM2 增强的两层 Transformer。空间分支输入 ESMFold 预测的 Cα 坐标、逐残基 pLDDT、42 维残基属性、16 维几何描述符以及 ESM2 表征，再通过三层等变图消息传递更新残基状态和坐标。

### 5.2 16 维几何描述符

当前冻结模型实际使用的 16 维 Cα 几何特征为：

1. 当前残基 pLDDT；
2. 低置信标记（pLDDT < 0.55）；
3. 局部窗口 pLDDT 均值；
4. 局部窗口 pLDDT 最小值；
5. 到肽几何中心的归一化径向距离；
6. 6 Å 内接触密度；
7. 8 Å 内接触密度；
8. 10 Å 内接触密度；
9. 指数衰减软接触密度；
10. 近邻平均距离；
11. 最近非局部残基距离；
12. 相邻主链距离；
13. 局部夹角余弦；
14. 局部二面角正弦；
15. 局部二面角余弦；
16. 长度缩放的径向位置。

这些量描述的是预测 Cα 骨架的紧致性、局部弯折、空间中心/表面倾向和序列远隔残基是否靠近。它们没有显式给出完整侧链取向、氢键、溶剂可及面积或静电势。

### 5.3 混合图和边表示

残基作为节点。混合图同时保留：

- 序列相邻边，保证主链连续信息可传播；
- 每个残基的空间 top-10 近邻边，捕获折叠后靠近但序列上可能远隔的残基。

边表示包括 16 个径向基函数距离通道、成对 pLDDT 置信度和序列间隔。空间距离越远，消息通过 `exp(-distance/12)` 衰减；成对置信度近似为两个残基 pLDDT 乘积的平方根，再以幂次 1.5 映射到边权。当前最低边权保留为 0.1，因此低 pLDDT 边被降权而不是完全删除。

### 5.4 pLDDT-aware EGNN

EGNN 在不依赖任意坐标轴方向的前提下，根据节点表征、成对距离和边特征传递信息。旋转或平移输入坐标不应改变毒性判断的物理含义，这比直接把 x/y/z 坐标当普通数值更合理。

每层主要完成：

1. 根据两个残基状态、距离 RBF、成对置信度和序列间隔生成消息；
2. 用图连接、距离衰减和 pLDDT 权重调节消息强度；
3. 聚合邻域消息更新残基表征；
4. 用较小系数更新相对坐标，并重新中心化。

训练时加入标准差 0.12 的坐标扰动，低 pLDDT 位置扰动更强。这相当于告诉模型：对预测不稳定的结构细节不应过度拟合。

### 5.5 两级置信门控与跨注意力

当前 `learned_confidence` 融合不是简单把 1D 和 3D 向量相加，而是设置两个层次的结构使用量：

#### 残基级门控

每个位置的门控由残基理化特征、该位置 pLDDT、pLDDT 相对中心值、平方项和低置信标记共同决定。空间表示先乘残基门控，再作为跨注意力的 key/value。其含义是：模型可以在同一条肽内更多使用高可信结构区域，减少低可信区域的影响。

#### 全局结构门控

整条肽的门控使用平均 pLDDT、最小 pLDDT 和低置信残基比例。跨注意力输出再乘全局门控后与序列表示拼接。其含义是：当整条预测结构整体不稳定时，模型可退回更多序列信息，而不是强制相信 3D。

重要限定：**pLDDT 是结构预测模型的置信度，不是肽自身的生物活性属性。**低 pLDDT 可能来自真实柔性，也可能来自 ESMFold 对短肽缺乏信息；不能把“低 pLDDT”直接解释成“更柔性”“更无序”或“更无毒”。

### 5.6 3D 生物学解释线路

3D 模型能够提出但尚不能直接证明的机制相关假说包括：

- 高接触密度与短非局部距离可能对应更紧致的构象；
- 序列远隔残基在空间上靠近时，模型可以学习协同的局部表面模式；
- 电荷/疏水残基的空间聚集可能比单纯序列组成更有判别力；
- 某些局部弯折或端部靠近模式可能与稳定性、膜接触面或靶标结合倾向相关。

这些只能表述为“模型利用的结构统计关联”或“可供实验验证的机制假说”。仅凭 ESMFold Cα 结构不能声称发现了真实结合口袋、氢键网络、膜插入角度或受体相互作用。

### 5.7 当前证据

冻结 seed 42 默认模型：

| 数据集 | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test1 | 0.9179 | 0.8885 | 0.8458 | 0.9699 | 0.9426 |
| Test2 | 0.8481 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |

三随机种子均值：

| 数据集 | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test1 | 0.8678 ± 0.0213 | 0.8195 ± 0.0276 | 0.9669 ± 0.0027 | 0.9336 ± 0.0079 |
| Test2 | 0.6119 ± 0.0449 | 0.5782 ± 0.0499 | 0.9225 ± 0.0099 | 0.6356 ± 0.0557 |

高 pLDDT 测试子集性能明显更高：Test1 MCC 从 0.8458 升至 0.8878，Test2 MCC 从 0.6320 升至 0.7221。但只用高 pLDDT 样本训练会降低完整测试域性能，所以当前采用连续降权和门控，而不是硬删除低置信样本。

ToxinPred3 strict 上 3D 的 precision 为 0.9162、recall 为 0.5428、MCC 为 0.5495，表现出很强的保守预测倾向。它在外部域没有超过 1D，说明预测结构带来的同域增益未完全跨域迁移。

### 5.8 化学位点—方向—连续多尺度相互作用图创新尝试

#### 5.8.1 数据基础与默认模型边界

2026-07-16 获得了覆盖 Train、Test1、Test2 和 ToxinPred3 strict 的 ESMFold 全原子 PDB：分别为 6,387、1,126、582 和 977 条，共 9,072 条。压缩包 SHA-256 已核对一致。包内旧格式 NPZ 仍只包含 `coords`、`plddt`、`backbone_coords`、`backbone_mask`、`sequence` 和 `source_pdb`，化学位点必须从保留的全原子 PDB 重新解析。

旧单中心解析在 274,141 个残基中得到 178,795 个有效中心，覆盖率为 65.22%。最后一轮训练缓存进一步加入脂肪族疏水位点，并把 Tyr/Trp 等多功能侧链拆为两个位点：训练集 201,994 个残基中，186,263 个残基至少有一个位点，覆盖率提高到 92.21%，共得到 195,031 个位点。**当前冻结默认分类器仍是 Cα 几何模型；以下化学位点支路作为保留的实验候选存在，但尚未升级为默认模型。**

#### 5.8.2 第一轮：单中心、多锚点和固定双尺度图

第一轮把每个残基压缩成一个官能团几何中心，计算其相对本残基 Cα、N 端 Cα、全链 Cα 质心和 C 端 Cα 的位置关系；距离除以 Cα 回转半径，方向以夹角余弦编码，再建立 6 Å 与 12 Å 两张官能团图并残差融合到原始 Cα-EGNN。该实现的主要问题是：一个中心会混合 Tyr 的芳香环与羟基、Trp 的芳香环与供体氮；固定截断使边在阈值处突变；只保留归一化距离还可能抹掉有意义的绝对尺寸。

#### 5.8.3 最后一轮：多化学位点、方向和连续相互作用图

最终候选不再增加锚点或继续调整 6/12 Å，而采用以下冻结设计：

1. 每个残基最多保留两个化学位点，位点具有正电、负电、氢键供体、氢键受体、芳香、疏水、含硫和构象约束八类可重叠属性；
2. Tyr 分为芳香环中心和羟基，Trp 分为芳香环中心和 `NE1`，带平面基团使用环面/胍基/羧酸盐法向，末端单原子基团使用局部键方向；
3. 为 Ala/Val/Leu/Ile/Pro 增加脂肪族疏水位点，避免只建模带电或极性残基而遗漏潜在膜接触斑块；
4. 排除同一残基内部连边，只传递跨残基空间关系，减少“由氨基酸身份即可确定”的平凡几何对 ESM2 信息的重复；
5. 同时使用 0–16 Å 的 16-bin 原始距离 RBF 和 0–4 回转半径单位的 8-bin 归一化 RBF，由连续核学习多尺度关系，不再手工建立两张截断图；
6. 边显式标记异号电荷、供体–受体、芳香–芳香、阳离子–π、硫–硫和疏水–疏水六类候选相互作用，并加入位点—连线方向余弦、位点间方向余弦和成对 pLDDT 权重；
7. 两层 64 维位点消息网络先聚合到残基，再经零初始化投影写入 128 维 Cα 空间分支；初始候选输出与对照完全一致；
8. 候选从 fold 0 最佳对照 checkpoint 初始化，训练期间冻结 ESM2 序列支路、Cα-EGNN、融合层和分类器，只允许新增化学位点支路学习，直接检验其增量信息。

整个表示只使用距离和方向点积，因此对整体平移和旋转不变；坐标增强对同一残基的 Cα、主链、旧官能团中心和新化学位点施加相同平移，不破坏残基内几何。

#### 5.8.4 公平对照结果与保留决策

采用 seed 42、固定 CD-HIT 80% 同源分组 fold 0、相同全原子 Cα 坐标、ESM2 输入、损失和 MCC 早停规则。保留判断只使用 1,271 条训练内分组验证样本；Test1、Test2 和 ToxinPred3 没有参与本轮选择。下表均为各自最佳验证 MCC epoch：

| 模型 | Epoch | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 pLDDT-aware Cα 默认架构重训对照 | 20 | 0.9183 | 0.8981 | 0.8616 | 0.9735 | 0.9529 |
| 第一轮单中心、多锚点、6/12 Å 候选 | 11 | 0.9165 | 0.8789 | 0.8297 | 0.9734 | 0.9387 |
| 最终多位点连续相互作用候选 | 8 | **0.9257** | **0.9040** | **0.8677** | **0.9737** | **0.9551** |
| 最终候选 − 对照 | — | +0.0074 | +0.0058 | +0.0061 | +0.0002 | +0.0021 |

最后一轮扭转了第一轮的明显下降，五项指标方向均为正，说明分离化学位点、加入方向和连续距离核比“单中心＋固定双图”更合理。因此实现、配置、训练缓存和 fold 0 checkpoint 作为**实验候选**保留。但 MCC 增量 +0.0061 低于预先设定的约 +0.01 单折晋级门槛，而且只有一个 group-aware fold，不能排除 checkpoint 选择波动；本轮不进入继续调参或五折三种子扩展，不替换 [`configs/default.json`](configs/default.json) 和当前冻结默认 checkpoint，也不在 Test1/Test2/ToxinPred3 上追加一次选择性评估。

#### 5.8.5 生物学解释与证据边界

该支路把模型可读取的空间证据细化为：Lys/Arg 与 Asp/Glu 的异号电荷邻近，正电位点与芳香面的阳离子–π候选关系，芳香/脂肪族位点形成的疏水斑块，供体–受体的极性邻近，以及 Cys/Met 含硫位点的空间聚集。Tyr/Trp 的双位点拆分使“芳香表面”和“羟基/吲哚氮”不再被同一个平均坐标混淆；脂肪族位点提高覆盖率后，模型也能描述不依赖芳香残基的疏水表面。

这些边仍然只是**由预测坐标推导的候选相互作用**，不是已确认的盐桥、氢键、阳离子–π作用、二硫键或膜结合机制。ESMFold 的残基 pLDDT 不是侧链原子级置信度；单一构象没有覆盖短肽柔性和构象集合；固定类型没有表示 pH 依赖质子化；使用绝对方向余弦牺牲了部分有向信息；溶剂、膜、离子和暴露度也未进入模型。新增位点类型仍由残基身份决定，可能与 ESM2 和 42 维理化特征重复。

因此当前论文级结论应是：“更化学化、方向感知且连续多尺度的残差支路在一个训练内同源分组 fold 上获得小幅一致改善，支持该表示优于第一轮形式，但证据不足以升级默认模型。”如果未来获得新的开发预算，应预先冻结五折三随机种子方案后一次性验证；不得根据已经查看的 Test1、Test2 或 ToxinPred3 继续选择位点定义和距离范围。

---

## 6. 三条线路的互补关系与当前部署定位

| 线路 | 主要回答的问题 | 核心优势 | 核心风险 | 当前定位 |
| --- | --- | --- | --- | --- |
| 1D | “序列顺序和上下文像不像毒性肽？” | 无需结构、计算较低、外部迁移最好 | 缺少显式原子和空间关系 | 正式回退模型；外部优先参考 |
| 2D | “局部原子化学和共价拓扑提供了什么额外信息？” | 化学粒度高、原子到残基可对齐 | 无真实构象；外部增益不足 | 化学解释/消融专家 |
| 3D | “预测空间邻域和结构质量是否改变判断？” | 可表示非局部空间接触；质量感知 | ESMFold 误差、计算高、外部迁移弱 | 历史默认决策模型；结构解释主线 |

当前三通路动态路由在 group-aware OOF 中改善了校准和风险分层，但在 Test2 没有超过冻结 3D 默认模型，因此没有替换默认分类器。合理部署方式是同时报告：3D 默认概率、1D 回退概率、结构质量和跨种子不确定性；当结构不可得或低可信时优先参考 1D，而不是宣称路由已稳定提高最终分类性能。

---

## 7. 已实现的残基级生物学解释

解释模块已经对冻结 3D 默认模型实现以下互补方法：

### 7.1 固定上下文单残基遮蔽

逐位置遮蔽氨基酸身份、官能团、42 维显式特征和该位置 ESM2 向量，并保持几何与全局描述符不变。记录原始 logit 与遮蔽后 logit 的差：

\[
\Delta z_i = z(x)-z(x_{\setminus i}).
\]

- \(\Delta z_i>0\)：该位置的当前序列/PLM 信息促进毒性预测；
- \(\Delta z_i<0\)：该位置的当前信息抑制毒性预测。

这是“固定结构上下文下的模型干预”，不是完整生物突变。真实突变还可能改变全局描述符和三维结构。

### 7.2 ESM2 通路 Integrated Gradients

以肽内 ESM2 向量均值重复到各位置作为基线，用 32 步 Integrated Gradients 分配 ESM2 输入通路对 logit 的贡献。该方法回答“上下文化 PLM 表示的哪些位置推动了当前预测”。当前四样本的平均相对 completeness error 约为 0.16%–0.70%，说明数值积分在该冒烟范围内基本闭合。

### 7.3 注意力与门控诊断

输出残基级结构门控、全局 3D 门控和跨注意力。它们用于判断模型是否读取某段结构信息。它们是**模型使用诊断**，不能单独当作因果重要性。

### 7.4 空间热点聚集

取综合归因绝对值较高的残基，比较热点间平均 Cα 距离与全部残基平均距离，并检查 8 Å 接触和最小序列间隔。若热点在空间上显著更近，可提出“模型重要残基形成空间簇”的假说；若不更近，则不应声称存在结构热点。

### 7.5 多随机种子一致性

当前解释同时使用 seed 42、123、2025 三个冻结检查点。概率标准差、归因方向一致性和热点重合比单个检查点热图更可信。若不同种子对某个位置方向相反，应标记为不稳定，而不是挑选最符合预期的结果。

### 7.6 已生成文件与正确解读

输出目录为 [`runs/biological_interpretation/smoke_test1/`](runs/biological_interpretation/smoke_test1/)：

- `residue_attributions.csv`：逐样本、逐残基、多方法归因与结构诊断；
- `sample_summary.csv`：样本概率、跨种子方差、热点和空间聚集摘要；
- `amino_acid_summary.csv`：在当前解释样本内按氨基酸聚合；
- `summary.json`：方法、阈值、样本数和解释边界。

当前只解释 4 条肽、126 个残基，包含 2 条毒性肽和 2 条非毒性肽。固定阈值 0.85 下，两条毒性肽均为假阴性，两条非毒性肽均为真阴性。因此，这批输出只能证明解释管线可运行、数值归因可闭合，**不能用来总结“某种氨基酸普遍促进/抑制毒性”**，也不能作为正确阳性机制案例。

`amino_acid_summary.csv` 中 H、K、P 等在这 4 条样本内呈正向均值，D、Q 等呈负向均值，但样本过少、序列组成不平衡且包含错误预测；这些数值不得直接写入论文结论。正确的论文级下一步应在冻结模型上覆盖 Test1、Test2 和严格外部集，按 TP/TN/FP/FN、长度、标签、pLDDT 和同源组分层，再以 bootstrap 置信区间检验方向稳定性。

### 7.7 已实现但尚未执行的反事实突变

代码支持在重算突变后的氨基酸描述符和 ESM2 表征时，固定原始 ESMFold 几何进行突变扫描。由于当前本机没有可直接使用的 ESM2-650M 权重缓存，现有预计算 `.npz/.pt` 也不能生成任意突变序列的 ESM2 表征，所以本次输出中 `num_mutations=0`。

即使未来执行，该结果也只能叫“固定几何近似下的反事实序列效应”。高影响突变仍应重新预测结构，最好再通过分子模拟或实验验证。

### 7.8 论文级生物学分析建议

在不再训练或调参的前提下，完整解释应按以下固定流程进行：

1. 对全部评估样本输出三随机种子概率、遮蔽、IG、门控和接触信息；
2. 把样本划分为 TP、TN、FP、FN，避免只解释预测正确或最漂亮的案例；
3. 在每个标签内比较促进/抑制热点的电荷、疏水性、芳香性、Cys 和端部富集；
4. 对热点空间聚集比、长程接触数和 pLDDT 做分层 bootstrap；
5. 检查 1D 与 3D 热点是否一致：一致表示序列证据主导，3D 特有热点才可能代表空间补充；
6. 对外部集重复同一分析，区分同域规律与跨域稳定规律；
7. 只把跨数据集、跨种子且置信区间不跨零的关联写成稳定模型发现；
8. 选择少量高置信 TP、稳定 FN 和高置信 FP 作为案例，分别展示成功机制、漏检机制和伪相关风险。

---

## 8. 创新点及参考论文来源

本项目的创新应表述为**对已有思想的质量感知整合、严格验证和工程化改造**，而不是声称 ESM2、EGNN、原子图或跨注意力本身由本项目首创。

| 本项目组成 | 主要思想来源 | 原论文做法 | 本项目的具体改造或新增 |
| --- | --- | --- | --- |
| ESM2 + ESMFold + 几何图 | PeptiTox | ESM2-650M、ESMFold、几何 featurizer 和图模型 | 拆分序列/空间支路，引入 pLDDT 边权、两级门控和坐标噪声；不无条件相信结构 |
| 原子级化学与跨模态对齐 | ToxiPep | BiGRU/Transformer 序列支路、原子级表示、多尺度 2D CNN、cross-attention | 用 ESM2 Transformer + 三层 Atom-MPNN；先聚合到残基，再做双向序列↔原子跨注意力 |
| PLM + ESMFold + EGNN | StrucToxNet | ProtT5、ESMFold、EGNN、对低 pLDDT 样本进行质量分析/筛选 | ProtT5 对照后保留 ESM2；把硬筛选改为连续边权、残基门控、全局门控和低置信坐标增强 |
| 序列局部/长程建模与突变解释 | 导师论文 ToxMSRC | Word2Vec、3/5/7 多尺度 CNN、BiLSTM、残差、SMOTE、位置突变 | 实际测试多尺度 CNN 后因无提升停止；保留 ESM2-BiLSTM 回退；使用 `pos_weight` 而非 SMOTE；解释扩展为遮蔽+IG+种子一致性+空间聚集 |
| 多化学位点、方向与连续多尺度相互作用图 | 本项目提出；受 PeptiTox 多原子几何、ToxiPep 原子化学和 StrucToxNet EGNN 启发 | 三篇工作分别提供多原子几何、原子级化学或等变图思想，但没有采用本项目这一组合 | 从全原子 PDB 拆分芳香/极性/电荷/疏水等位点，加入键方向或平面法向，以原始 Å 距离和回转半径归一化距离的连续 RBF 编码六类候选相互作用，再零初始化残差并入 Cα-EGNN；fold 0 小幅提升但未达升级门槛 |

### 8.1 创新一：残基对齐的 1D/2D/3D 三维度表征

三条线路都以残基位置为共同坐标：序列 token、原子聚合后的残基表示和 Cα 空间节点可以逐位置对齐。这使模型比较的不只是三个整肽向量，而是每个残基在序列语义、局部原子化学和预测空间环境中的不同视图。

### 8.2 创新二：预测结构质量进入消息传递和融合过程

项目不只在结果分析时报告 pLDDT，而是让 pLDDT 参与：

- EGNN 边消息权重；
- 残基级结构门控；
- 整肽级全局结构门控；
- 低置信区域更强的训练坐标扰动；
- 结果分层和解释可信度判断。

相较“删除所有低 pLDDT 样本”，连续质量建模保留了样本覆盖，并允许同一条肽内高低可信区域被不同对待。

### 8.3 创新三：把无结构模型作为可靠部署回退

严格外部评估发现 1D 迁移优于 2D/3D，这一负结果被保留并转化为部署设计：当结构不可用、低可信或域外偏移明显时，1D 不是临时消融，而是正式回退能力。这比单纯追求更复杂结构模型更符合真实使用场景。

### 8.4 创新四：解释链路同时约束“重要性”和“可信性”

本项目把固定上下文遮蔽、ESM2 IG、跨种子一致性、pLDDT/门控和 Cα 空间聚集联合起来。这样可以区分：

- 序列输入真正改变输出的位置；
- PLM 上下文贡献的位置；
- 模型读取但未必因果重要的注意力/门控；
- 可能形成空间簇的热点；
- 对随机种子或结构质量不稳定的解释。

### 8.5 创新五：严格验证与负结果报告

项目建立了 group-aware OOF、nested group CV、三随机种子、bootstrap 区间、Brier/ECE 校准、pLDDT 分层和严格外部同源筛查。多尺度 CNN、ProtT5、SupCon、Focal BCE、高 pLDDT 子集训练、增强 2D 特征、局部坐标系、全主链几何、官能团多锚点双尺度图、残差专家、动态路由、选择性风险和 conformal 等未达到门槛的路线均被记录或停止，没有只保留正结果。最后一轮多化学位点连续相互作用图虽在单折上小幅转正，仍因未达到预设晋级门槛而只保留为实验候选。

### 8.6 创新六：把项目原创结构假说也纳入可证伪验证

化学位点—方向—连续多尺度相互作用图不是四篇论文中可直接复制的现成模块，而是本项目根据肽化学提出的组合式结构假说。项目先证伪了单中心、多锚点和固定 6/12 Å 双图，再用芳香/极性位点拆分、脂肪族疏水覆盖、方向不变量、双距离 RBF、显式相互作用类型和冻结基线零初始化残差进行最后一次优化。该优化在 fold 0 上把 MCC 从 0.8616 提高到 0.8677，但未达到升级门槛。这一创新当前体现为**问题提出、可运行形式化、失败实现的针对性修正和有限正证据**，而不是已经证明的性能领先。

---

## 9. 四篇参考论文与本项目的具体关系

### 9.1 PeptiTox

论文：Wang 等，*Integrating Protein Language Models and Geometric Deep Learning for Peptide Toxicity Prediction*，J. Chem. Inf. Model. 2025，DOI: 10.1021/acs.jcim.5c01073。项目内文件：[`2025PeptiTox.docx`](2025PeptiTox.docx)。

PeptiTox 使用冻结 ESM2-650M 残基向量、ESMFold 结构和较完整几何 featurizer，残基图边按约 15 Å Cα 距离构建，并利用主链/侧链几何信息。该论文为本项目的 ESM2 + 预测结构 + 几何图主线提供最直接启发。

不同点：PeptiTox 使用不同的平衡数据和随机拆分，报告 ACC 0.9444、MCC 0.8888，不能与当前不平衡 Test1/Test2 直接比较。本项目更强调结构质量的连续降权、序列与空间分支分离、严格外部迁移和校准；但当前几何主要基于 Cα，几何完整性仍弱于 PeptiTox 的多原子 featurizer。

### 9.2 ToxiPep

论文：Guan 等，*ToxiPep: Peptide toxicity prediction via fusion of context-aware representation and atomic-level graph*，Computational and Structural Biotechnology Journal 27 (2025) 2347–2358。项目内文件：[`2025ToxiPep.docx`](2025ToxiPep.docx)。

ToxiPep 的核心是上下文序列表示、原子级图/张量、多尺度二维卷积和跨注意力。它还整合了多个数据来源。该论文启发本项目建立原子级化学分支以及序列—原子对齐。

不同点：本项目没有新增训练数据；原子侧使用消息传递网络而非多尺度 2D CNN；先把原子聚合到残基位置，再进行双向跨注意力。ToxiPep 论文中的 C 富集、L/V/I 差异或特定位置结论属于其数据统计，不能直接搬到本项目；必须用当前完整数据重新检验。

### 9.3 StrucToxNet

论文：Jiao 等，*Integration of pre-trained protein language models with equivariant graph neural networks for peptide toxicity prediction*，BMC Biology 23, 229 (2025)，DOI: 10.1186/s12915-025-02329-1。项目内文件：[`StrucToxNet 论文`](<Jiao, S., Ye, X., Sakurai, T. et al. Integration of pre-trained protein language models with equivariant graph neural networks for peptide toxicity prediction. BMC Biol 23, 229 (2025)..docx>)。

StrucToxNet 使用与当前 Test1 对应的数据框架，组合 ProtT5、ESMFold 和 EGNN，并系统讨论 pLDDT。它是当前 3D 架构和质量意识最接近的参考工作。

该论文报告 Test1 BACC 0.9318、AUROC 0.968、MCC 0.852；当前 seed 42 为 BACC 0.9179、AUROC 0.9699、MCC 0.8458。即当前 AUROC 略高，但 BACC/MCC 略低，不能声称总体超过 StrucToxNet。StrucToxNet 的高 pLDDT 筛选实验可进一步提高其结果；本项目实测只用高 pLDDT 训练会降低完整域性能，因此改为连续质量感知，而非复制硬筛选。

### 9.4 导师论文 ToxMSRC

论文：Zhang、Ren、Liang，*An innovative peptide toxicity prediction model based on multi-scale convolutional neural network and residual connection*。项目内文件：[`2025ToxMSRC.docx`](2025ToxMSRC.docx)。

ToxMSRC 使用 CBOW Word2Vec、SMOTE、多尺度 CNN（核 3/5/7）、BiLSTM、残差连接和 MLP。其数据与当前基础数据一致，并未依赖新增训练数据，因此是最重要的直接基准。

本项目借鉴了局部多尺度、顺序长程依赖和突变解释思路，但没有机械复现最终结构：多尺度 CNN/残差路线已经测试且没有超过 ESM2-BiLSTM；类别不平衡使用损失权重而不是 SMOTE；序列表示由 Word2Vec 升级为上下文化 ESM2；解释从单位置突变扩展到多方法、跨种子和空间质量联合分析。

---

## 10. 预测指标的详细解释

设 TP 为正确预测的毒性肽，TN 为正确预测的非毒性肽，FP 为被误报为毒性的非毒性肽，FN 为被漏检的毒性肽。

### 10.1 Accuracy

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}.
\]

表示总体预测正确比例。当前 Test2 非毒性肽远多于毒性肽，即使模型偏向预测非毒，也可能得到较高 Accuracy，因此不能作为唯一指标。

### 10.2 Sensitivity / Recall

\[
Recall=\frac{TP}{TP+FN}.
\]

表示真实毒性肽中被检出的比例。毒性筛查中，低 Recall 意味着漏检风险高。当前 3D 外部 Recall 0.5428，说明约 45.7% 的外部毒性样本未达到冻结阈值。

### 10.3 Specificity

\[
Specificity=\frac{TN}{TN+FP}.
\]

表示真实非毒性肽中被正确排除的比例。高 Specificity 可减少把安全候选误报为有毒。

### 10.4 Precision 与 FDR

\[
Precision=\frac{TP}{TP+FP}, \qquad FDR=\frac{FP}{TP+FP}=1-Precision.
\]

Precision 表示所有“预测有毒”样本中真正有毒的比例。当前 3D 外部 Precision 0.9162 很高，但与较低 Recall 同时出现，说明阈值下模型非常保守：一旦报毒通常可靠，但漏掉较多毒性肽。

### 10.5 F1

\[
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}.
\]

F1 是 Precision 和 Recall 的调和平均，对正类检测有意义，但不直接使用 TN，因此仍需与 MCC/BACC 配合。F1 依赖具体阈值。

### 10.6 Balanced Accuracy

\[
BACC=\frac{Recall+Specificity}{2}.
\]

BACC 给正负类同等权重，适合当前类别不平衡数据。导师论文与当前项目的 BACC 在相同 Test1/Test2 上可以直接作历史比较。

### 10.7 Matthews Correlation Coefficient

\[
MCC=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
\]

MCC 同时考虑四个混淆矩阵元素，范围为 -1 到 1：1 表示完全正确，0 接近随机/无相关，-1 表示完全相反。它对类别不平衡较稳健，是本项目最重要的阈值决策指标之一。

### 10.8 AUROC

AUROC 衡量随机抽取一个毒性肽和一个非毒性肽时，模型把毒性肽排在更高分的概率。它综合所有阈值，反映排序能力，不保证某一固定阈值下的 Precision/Recall。正类稀少时，AUROC 可能在实际阳性检索仍不理想的情况下保持较高。

### 10.9 AUPRC

AUPRC 是 Precision-Recall 曲线下面积，更聚焦正类检出。在毒性肽相对稀少时，它通常比 AUROC 更敏感。AUPRC 的基线受正类比例影响，不同正类比例的数据集不能只根据绝对 AUPRC 大小简单排名。

### 10.10 Brier score

\[
Brier=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
\]

Brier 同时惩罚错误概率和过度自信，越低越好。一个分类正确但概率极端失真的模型仍可能有较差 Brier。它评估的是概率质量，不是单纯分类边界。

### 10.11 Expected Calibration Error

ECE 把预测概率分箱，比较每箱平均置信度与实际正确率/阳性率的差异，再按箱样本量加权。越低表示概率越接近可解释频率。ECE 依赖分箱数和样本量，本项目报告 ECE-10，并与 Brier、可靠性图联合解释。

### 10.12 阈值、随机种子和置信区间

AUROC/AUPRC 是阈值无关的排序指标；BACC、MCC、F1、Precision 和 Recall 都依赖阈值。不同实验阈值不同，不能只抄指标而忽略阈值来源。当前冻结 3D 阈值为 0.85，1D 为 0.50，2D 为 0.25。

神经网络结果还受初始化、批次和优化过程影响。单个 seed 42 可用于复现冻结检查点，三随机种子均值更接近期望性能。bootstrap 置信区间描述样本抽样不确定性，跨种子标准差描述训练不确定性，两者回答的问题不同。

---

## 11. 当前模型与导师论文的直接比较

### 11.1 相同 Test1/Test2 的指标差异

| 数据集 | 模型 | BACC | MCC | AUROC |
| --- | --- | ---: | ---: | ---: |
| Test1 | 导师 ToxMSRC | **0.9217** | **0.8520** | 0.9650 |
| Test1 | 当前 3D seed 42 | 0.9179 | 0.8458 | **0.9699** |
| Test1 | 当前减导师 | -0.0038 | -0.0062 | +0.0049 |
| Test2 | 导师 ToxMSRC | **0.8689** | **0.6550** | **0.9430** |
| Test2 | 当前 3D seed 42 | 0.8481 | 0.6320 | 0.9316 |
| Test2 | 当前减导师 | -0.0208 | -0.0230 | -0.0114 |

当前模型没有在全部指标上超过导师论文：Test1 AUROC 略高，但 Test1/Test2 的 BACC、MCC 以及 Test2 AUROC 较低。考虑到 Test1/Test2 已被历史开发多次查看，不能据这些微小差异宣称显著优劣；应把它们作为同数据历史基准，并把严格外部集和 nested group CV 作为更重要证据。

### 11.2 当前项目相对导师论文的优势

1. **表征更丰富**：从 Word2Vec 序列扩展到 ESM2 上下文、原子拓扑和质量感知空间几何。
2. **长程上下文更强**：ESM2 和 Transformer/BiLSTM 能表达上下文化序列关系，不限于相邻氨基酸对。
3. **不依赖 SMOTE 生成训练样本**：通过 `pos_weight` 调整损失，避免合成向量可能带来的分布扭曲。
4. **显式处理预测结构不确定性**：pLDDT 同时进入边权、门控、扰动和分层分析。
5. **验证更完整**：有组感知 OOF、nested CV、三随机种子、bootstrap、校准和严格外部同源筛查。
6. **负结果透明**：没有因复杂模块未提升而隐去结果。
7. **解释维度更完整**：不仅做位置变化，还联合遮蔽、IG、种子一致性、结构质量和空间热点。
8. **具有无结构回退能力**：1D 在严格外部域表现最好，适合服务器结构不可得或结构低可信时使用。

### 11.3 当前项目相对导师论文的不足

1. **核心决策指标未形成明确领先**：相同 Test1/Test2 上 BACC/MCC 仍略低。
2. **计算与存储成本明显增加**：需要 ESM2 表征、ESMFold 结构和图网络推理。
3. **系统更复杂**：数据缓存、模型分支和质量控制增加复现与部署难度。
4. **3D 外部迁移不足**：严格外部集上复杂结构路线反而低于 1D。
5. **生物学解释仍处于技术验证阶段**：当前只有 4 条肽的冒烟输出，没有形成群体统计结论。
6. **没有湿实验验证**：所有解释仍是模型关联或计算假说。

### 11.4 导师论文仍然具有的优势

ToxMSRC 架构更简洁、推理更快，在相同 Test1/Test2 上有较强的 BACC/MCC，且位置突变分析直观。对计算资源有限或只需要快速序列筛查的场景，简单模型具有现实价值。

但其位置突变热点也可能部分反映训练集中氨基酸位置频率，而不是通用生物机制；Word2Vec 对长程上下文表达有限，模型主要适用于论文设定的 11–50 aa 范围；缺少原子/结构信息、严格外部迁移、概率校准和多随机种子不确定性分析。

---

## 12. 当前项目总体优缺点

### 12.1 优点

- 已形成从序列、原子拓扑到预测空间几何的完整多维建模框架；
- pLDDT 不是只作为筛选阈值，而是进入模型内部的质量感知计算；
- 有可独立部署的 1D 回退线路，避免结构失败导致系统不可用；
- 同时报告决策、排序、校准、随机种子和 bootstrap 不确定性；
- 外部集没有参与训练和阈值调整，并进行了严格同源审计；
- 已建立可追溯到残基、种子、门控和接触的解释流水线；
- 保留未提升结果，降低重复试验和选择性报告风险；
- 与四篇参考论文的继承关系和差异能够明确定位。

### 12.2 缺点与风险

- 当前没有证据支持“普遍达到 SOTA”或“显著超过所有参考论文”；
- 历史 Test1/Test2 已参与开发观察，独立性有限；
- Test2 只有 46 个正样本，决策指标方差较大；
- 只有一个新外部来源，仍不足以证明跨实验室、跨数据库普遍泛化；
- ESMFold 对短肽、柔性肽和多构象体系可能不稳定；
- 当前 3D 主要是 Cα 几何，不能完整表达侧链取向、氢键和溶剂作用；
- 当前 2D 图没有独立的结构质量分数，也没有真实构象；
- 只处理标准线性序列，对环肽、非天然残基和翻译后修饰支持不足；
- 只预测总体毒性，不能区分溶血、细胞毒、神经毒等亚型或剂量；
- 模型解释可能受到训练分布、标签噪声和 PLM 先验影响；
- 当前残基解释样本过少，不能形成群体生物学结论；
- 没有实验结构、分子动力学或湿实验验证。

---

## 13. 论文中可以与不可以使用的表述

### 13.1 当前证据支持的表述

- “本项目构建了残基对齐的 1D 序列、2D 原子图和 3D 预测结构三条线路。”
- “默认 3D 模型通过 pLDDT 加权消息传递与两级门控降低对低可信预测结构的依赖。”
- “在历史 Test1/Test2 上，3D 融合模型的固定阈值决策指标优于本项目其他单线路。”
- “在严格 ToxinPred3 外部集上，1D ESM2-BiLSTM 的迁移性能高于当前 2D/3D 路线。”
- “高 pLDDT 子集性能更好，但硬筛选高 pLDDT 训练样本没有改善完整测试域，因此采用连续质量感知。”
- “多化学位点、方向和连续距离核残差支路在一个训练内同源分组 fold 上取得五项指标同向小幅改善，但未达到升级默认模型的预设门槛。”
- “已实现多方法、多随机种子的残基级解释流水线，当前冒烟输出证明技术可行。”

### 13.2 当前证据不支持的表述

- “本项目已达到普遍 SOTA。”
- “当前模型显著超过导师论文、StrucToxNet 或所有参考方法。”
- “3D 信息在所有数据域都优于序列信息。”
- “当前 3D 分支是纯几何模型。”
- “pLDDT 是肽柔性、无序程度或毒性的直接实验指标。”
- “注意力最高的残基就是毒性因果位点。”
- “四样本氨基酸汇总揭示了通用毒性规律。”
- “动态路由或残差专家已经稳定改善最终分类性能。”
- “当前默认模型已经使用化学位点图，或化学位点候选已经获得多折稳定提升。”
- “化学位点图中的相互作用边证明了真实盐桥、氢键、阳离子–π作用、二硫键或膜结合机制。”
- “模型已经证明某条肽通过膜破坏、受体结合或特定分子机制致毒。”

---

## 14. 关键项目资产与复现入口

| 内容 | 文件或目录 |
| --- | --- |
| 1D 冻结配置 | [`configs/sequence_1d_bilstm.json`](configs/sequence_1d_bilstm.json) |
| 2D 冻结配置 | [`configs/plm_sequence_atom_cross_attention.json`](configs/plm_sequence_atom_cross_attention.json) |
| 3D 冻结默认配置 | [`configs/default.json`](configs/default.json) |
| 全原子 ESMFold PDB（9,072 条） | [`data/esmfold_fullatom/pdb/`](data/esmfold_fullatom/pdb/) |
| 全原子服务器结果压缩包 | [`GHXTox_esmfold_fullatom_20260716_014641.tar.gz`](GHXTox_esmfold_fullatom_20260716_014641.tar.gz) |
| 官能团中心与主链几何解析 | [`src/ghxtox/esmfold_cache.py`](src/ghxtox/esmfold_cache.py) |
| 多化学位点定义与全原子解析 | [`src/ghxtox/chemical_sites.py`](src/ghxtox/chemical_sites.py) |
| 连续多尺度化学相互作用支路 | [`src/ghxtox/models/chemical_sites.py`](src/ghxtox/models/chemical_sites.py) |
| 最终实验候选配置 | [`configs/chemical_site_interaction_final.json`](configs/chemical_site_interaction_final.json) |
| 最终实验训练缓存 | [`data/processed/train_chemical_sites_final_esm2.pt`](data/processed/train_chemical_sites_final_esm2.pt) |
| 匹配对照与实验候选 checkpoint | [`runs/chemical_site_final_control_fold0/`](runs/chemical_site_final_control_fold0/)、[`runs/chemical_site_interaction_final_fold0/`](runs/chemical_site_interaction_final_fold0/) |
| 共同残基特征 | [`src/ghxtox/features.py`](src/ghxtox/features.py) |
| 2D 原子图构建 | [`src/ghxtox/atom_graph.py`](src/ghxtox/atom_graph.py) |
| Atom-MPNN | [`src/ghxtox/models/atom_graph.py`](src/ghxtox/models/atom_graph.py) |
| 序列、EGNN 与 pLDDT 融合层 | [`src/ghxtox/models/layers.py`](src/ghxtox/models/layers.py) |
| 三种模态总模型 | [`src/ghxtox/models/dual_modal.py`](src/ghxtox/models/dual_modal.py) |
| 3D 几何描述符 | [`src/ghxtox/geometry_features.py`](src/ghxtox/geometry_features.py) |
| 生物学解释实现 | [`src/ghxtox/biological_interpretation.py`](src/ghxtox/biological_interpretation.py) |
| 当前解释输出 | [`runs/biological_interpretation/smoke_test1/`](runs/biological_interpretation/smoke_test1/) |
| 1D 外部结果 | [`runs/external_toxinpred3_1d/`](runs/external_toxinpred3_1d/) |
| 2D 外部结果 | [`runs/external_toxinpred3_2d/`](runs/external_toxinpred3_2d/) |
| 3D 外部结果 | [`runs/external_toxinpred3_3d/`](runs/external_toxinpred3_3d/) |

本项目本地 Python 解释器应使用 Conda 环境 `D:\anaconda3\envs\dachuang-26\python.exe`。

### 14.1 最终化学位点实验复现

现有派生缓存已经生成；如需从全原子 PDB 重建，可执行：

```powershell
$env:PYTHONPATH = "src"
D:\anaconda3\envs\dachuang-26\python.exe -B -m ghxtox.attach_chemical_sites `
  --processed data\processed\train_cached_func_esm2.pt `
  --pdb-dir data\esmfold_fullatom\pdb\train `
  --output data\processed\train_chemical_sites_final_esm2.pt
```

匹配对照与冻结基线残差候选依次运行：

```powershell
D:\anaconda3\envs\dachuang-26\python.exe -B -m ghxtox.train `
  --train data\processed\train_chemical_sites_final_esm2.pt `
  --fold-manifest data\folds\train_cdhit080_fallback_folds.csv `
  --fold 0 `
  --config configs\default_groupfold_control.json `
  --output-dir runs\chemical_site_final_control_fold0 `
  --device cuda

D:\anaconda3\envs\dachuang-26\python.exe -B -m ghxtox.train `
  --train data\processed\train_chemical_sites_final_esm2.pt `
  --fold-manifest data\folds\train_cdhit080_fallback_folds.csv `
  --fold 0 `
  --config configs\chemical_site_interaction_final.json `
  --output-dir runs\chemical_site_interaction_final_fold0 `
  --device cuda
```

第二条命令会按 `--fold` 自动从 `runs/chemical_site_final_control_fold{fold}/best_model.pt` 初始化，并只训练 `chemical_site_branch.*`。删除或移动对照目录后，需同步修改候选配置中的 `initial_checkpoint` 模板。

---

## 15. 最终结论

GHXTox 当前最有价值的贡献不是单一指标上的绝对领先，而是把肽毒性预测从单一序列模型扩展为**残基对齐、结构质量感知、可回退且可审计**的多维系统：

- 1D 线路提供稳定的序列语义和外部迁移能力；
- 2D 线路补充原子与化学键层面的局部化学环境；
- 3D 线路在 pLDDT 约束下利用预测空间邻域，并在历史同域测试上形成当前默认决策模型；
- 严格外部结果证明复杂结构信息并非总能迁移，因而必须保留 1D 回退并报告结构质量；
- 生物学解释已经从单一注意力或位置突变推进到遮蔽、IG、种子一致性、门控和空间热点联合分析，但目前仍需扩大到完整评估集才能形成论文级群体结论。

最稳妥的论文定位是：**提出一种融合上下文序列、原子拓扑和质量感知预测结构的肽毒性建模与解释框架，并通过严格验证揭示结构信息的条件性收益及无结构回退的重要性。**

---

## 16. 环境、运行命令与验证

### 环境

项目使用以下解释器：

```powershell
$py = "D:\anaconda3\envs\dachuang-26\python.exe"
$env:PYTHONPATH = "src"
```

已验证环境为 Python 3.10、PyTorch 2.7.1+cu128，CUDA 可用。IDE 配置位于 `.vscode/settings.json`。

### 默认训练与评估

训练：

```powershell
& $py -B -m ghxtox.train `
  --train data\processed\train_cached_func_esm2.pt `
  --config configs\plm_fusion_esm2_geometry_confidence.json `
  --output-dir runs\plm_fusion_esm2_geometry_confidence `
  --device cuda
```

评估 Test2：

```powershell
& $py -B -m ghxtox.evaluate `
  --checkpoint runs\plm_fusion_esm2_geometry_confidence\best_model.pt `
  --processed data\processed\test2_cached_func_esm2.pt `
  --threshold 0.85 `
  --output runs\plm_fusion_esm2_geometry_confidence\test2_metrics_default.json `
  --predictions runs\plm_fusion_esm2_geometry_confidence\test2_predictions_default.csv
```

### 17.1 重新生成 ToxinPred3 严格集

重新生成严格集：

```powershell
& $py -B -m ghxtox.prepare_toxinpred3 `
  --train-positive data\external\toxinpred3\raw\train_pos.csv `
  --train-negative data\external\toxinpred3\raw\train_neg.csv `
  --test-positive data\external\toxinpred3\raw\test_pos.csv `
  --test-negative data\external\toxinpred3\raw\test_neg.csv `
  --reference-manifests data\cdhit\input\train_manifest.csv data\cdhit\input\test1_manifest.csv data\cdhit\input\test2_manifest.csv `
  --output-dir data\external\toxinpred3\strict `
  --identity-threshold 0.8
```

### 冻结模型的生物学解释

`ghxtox.biological_interpretation` 在不改变 checkpoint、阈值、训练数据和结构缓存的条件下，跨三个冻结随机种子生成：

- 固定上下文的逐残基遮蔽效应；
- 以肽内平均 ESM2 向量为基线的 Integrated Gradients；
- 归因方向的随机种子一致性和双方法一致性；
- residue-level pLDDT、3D node gate、attention received；
- C-alpha 接触数、长程接触数和热点空间聚集；
- 残基归因表、样本摘要、氨基酸汇总和代表性图。

快速检查两个样本：

```powershell
& $py -B -m ghxtox.biological_interpretation `
  --processed data\processed\test1_cached_func_esm2.pt `
  --output-dir runs\biological_interpretation\test1 `
  --max-samples 2 `
  --device cuda
```

由于 Test1 按标签顺序保存，做小规模正负对照时应使用 `--max-samples-per-label 10`，而不是只用 `--max-samples`。完整 Test1 只需移除样本数量参数。输出中的正 `occlusion_delta_logit` 表示该位置在固定上下文下支持毒性预测；attention 和 gate 只表示模型的信息使用情况，不能单独解释为生物因果。

可选固定几何突变扫描会重新生成突变序列的理化特征和 ESM2 表示，只扫描每条肽归因最高的三个位置：

```powershell
& $py -B -m ghxtox.biological_interpretation `
  --processed data\processed\test1_cached_func_esm2.pt `
  --output-dir runs\biological_interpretation\test1_mutation `
  --max-samples 4 `
  --mutation-scan `
  --device cuda
```

突变扫描继续使用原 ESMFold 几何，因此只能作为候选突变筛选，不能替代突变体结构预测或湿实验验证。

### 验证

```powershell
$env:MPLBACKEND = "Agg"
& $py -B -m pytest tests -q -p no:cacheprovider
```

项目目录只长期保留冻结主线、辅助专家、OOF/置信度结果、论文报告和可复现的数据审计资产。

---

## 17. 完整机器结果记录

以下保留各随机种子、bootstrap、路由、消融和同源控制的原始数值，作为论文制表和复核依据。

### Frozen ToxinPred3 strict external evaluation: 1D, 2D, and 3D

No training data or model parameters were changed. The official ToxinPred3 independent split was screened against the complete current train/Test1/Test2 sequence set. Exact matches were removed first, followed by the explicitly labeled Biopython global-alignment fallback at identity `>=0.8`. A subsequent native CD-HIT 4.8.1 audit retained 987 samples and confirmed that all 977 fallback-retained samples were also native-retained; the 10 disagreements were native-only. The existing 977-sample set is therefore kept as the more conservative native-verified subset: 471 toxic and 506 non-toxic. All three previously trained ESM2-BiLSTM checkpoints were evaluated once at the frozen threshold `0.50`.

| Seed | Accuracy | BACC | Precision | Recall | F1 | MCC | AUROC | AUPRC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7779 | 0.7720 | 0.8994 | 0.6072 | 0.7250 | 0.5801 | 0.7845 | 0.8445 |
| 123 | 0.7973 | 0.7924 | 0.8980 | 0.6539 | 0.7568 | 0.6122 | 0.8503 | 0.8830 |
| 2025 | 0.7748 | 0.7698 | 0.8680 | 0.6285 | 0.7291 | 0.5656 | 0.8451 | 0.8744 |

Hierarchical seed-then-stratified-sample bootstrap, 5000 iterations:

| Metric | Three-seed mean | 95% CI |
| --- | ---: | ---: |
| BACC | 0.7780 | [0.7485, 0.8107] |
| Precision | 0.8885 | [0.8427, 0.9251] |
| Recall | 0.6299 | [0.5732, 0.6858] |
| F1 | 0.7369 | [0.6959, 0.7819] |
| MCC | 0.5859 | [0.5268, 0.6461] |
| AUROC | 0.8266 | [0.7624, 0.8706] |
| AUPRC | 0.8673 | [0.8288, 0.8972] |
| Brier | 0.1745 | [0.1489, 0.2057] |
| ECE-10 | 0.1301 | [0.1054, 0.1739] |

The frozen model is conservative on this shifted external domain: precision remains high while toxic-peptide recall is only about 0.63. This result is retained as external generalization evidence and must not be used to retune the threshold. Machine-readable outputs are under `runs/external_toxinpred3_1d/`; the data audit is `data/external/toxinpred3/strict/audit_summary.json`.

The existing best 2D ESM2 + 30D Atom-MPNN cross-attention checkpoint was then evaluated once at its frozen validation threshold `0.25`. No atom-graph training or architecture selection was repeated.

| Frozen 2D metric | Point estimate | 95% stratified bootstrap CI |
| --- | ---: | ---: |
| BACC | 0.7540 | [0.7292, 0.7791] |
| Precision | 0.8911 | [0.8567, 0.9236] |
| Recall | 0.5732 | [0.5287, 0.6178] |
| F1 | 0.6977 | [0.6597, 0.7343] |
| MCC | 0.5488 | [0.5003, 0.5958] |
| AUROC | 0.7786 | [0.7480, 0.8091] |
| AUPRC | 0.8373 | [0.8143, 0.8600] |
| Brier | 0.2160 | [0.1955, 0.2360] |
| ECE-10 | 0.2031 | [0.1833, 0.2267] |

The frozen 2D route does not improve transfer over the frozen 1D route on this external set and is less well calibrated. This is a frozen external comparison, not a new ablation; the result is retained under `runs/external_toxinpred3_2d/` and no 2D retuning is allowed.

The three existing pLDDT-aware ESM2 + ESMFold C-alpha geometry checkpoints were evaluated at their frozen threshold `0.85`. The 977 server-generated structure caches were checked for complete ID coverage, sequence-length consistency, finite coordinates/pLDDT, and alignment with the strict manifest before evaluation. No checkpoint, fusion weight, or threshold was changed.

| Seed | Accuracy | BACC | Precision | Recall | F1 | MCC | AUROC | AUPRC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7697 | 0.7629 | 0.9184 | 0.5732 | 0.7059 | 0.5728 | 0.7924 | 0.8459 |
| 123 | 0.7574 | 0.7499 | 0.9270 | 0.5393 | 0.6819 | 0.5559 | 0.7681 | 0.8305 |
| 2025 | 0.7400 | 0.7323 | 0.9033 | 0.5159 | 0.6568 | 0.5197 | 0.7283 | 0.8048 |

Hierarchical seed-then-stratified-sample bootstrap, 5000 iterations:

| Frozen 3D metric | Three-seed mean | 95% CI |
| --- | ---: | ---: |
| BACC | 0.7484 | [0.7142, 0.7811] |
| Precision | 0.9162 | [0.8768, 0.9506] |
| Recall | 0.5428 | [0.4820, 0.6072] |
| F1 | 0.6815 | [0.6269, 0.7329] |
| MCC | 0.5495 | [0.4848, 0.6076] |
| AUROC | 0.7629 | [0.7040, 0.8153] |
| AUPRC | 0.8271 | [0.7869, 0.8625] |
| Brier | 0.2260 | [0.1962, 0.2543] |
| ECE-10 | 0.2208 | [0.1928, 0.2497] |

The 3D route is highly precise but has the lowest toxic-peptide recall of the three frozen routes. It does not improve external transfer over 1D and is not used to start another structure ablation or to retune `0.85`. This does not invalidate the historically frozen default on Test1/Test2; it shows that the no-structure 1D backup transfers better to this shifted external domain. Predictions and bootstrap statistics are retained under `runs/external_toxinpred3_3d/`.

### Nested innovation screening: residual experts

A new five-outer-fold protocol separates group-disjoint train, validation, calibration, and outer-test roles. Each sample is outer test exactly once; calibration is never used for gradient updates or checkpoint selection. The current Test1/Test2 sets were not used to select this architecture.

| Nested outer-fold result | MCC mean +/- SD | AUROC mean +/- SD | AUPRC mean +/- SD | MCC wins |
| --- | ---: | ---: | ---: | ---: |
| ESM2-BiLSTM 1D base | 0.8179 +/- 0.0120 | 0.9661 +/- 0.0048 | 0.9308 +/- 0.0171 | -- |
| 1D base + 2D/3D residual experts | 0.8149 +/- 0.0252 | 0.9684 +/- 0.0052 | 0.9349 +/- 0.0135 | 2/5 |

The residual architecture changed mean MCC by `-0.0030` and mean AUPRC by `+0.0041`. It improved ranking but increased MCC variance and failed the predeclared requirements of at least `+0.005` mean MCC and wins on at least three outer folds. It is rejected as the foundation for the proposed benefit router. Dependent gain-routing, OOD, corruption-training, and conformal experiments were therefore not run. Machine-readable results are in `runs/nested_p1/summary.json`; the nested split audit is in `data/folds/train_nested_groupcv.json`.

### Three-route confidence router

A leakage-controlled meta study combined the frozen architecture families for the 1D ESM2-BiLSTM expert, the 2D ESM2-Atom-MPNN cross-attention expert, and the 3D PLDDT-aware geometry expert. Five group-aware OOF checkpoints per route produced deterministic probabilities, five-sample MC Dropout uncertainty, and route-specific quality features. Temperature scaling, reliability regression, and routing were cross-fitted across the five folds. Test1/test2 labels were not used for router fitting or threshold selection.

| OOF system | MCC | AUROC | AUPRC | Brier | ECE-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1D | 0.8408 | 0.9595 | 0.9201 | 0.0573 | 0.0328 |
| 2D | 0.8414 | 0.9590 | 0.9248 | 0.0569 | 0.0185 |
| 3D | 0.8357 | 0.9542 | 0.9199 | 0.0578 | 0.0270 |
| Uniform calibrated mean | **0.8516** | **0.9705** | **0.9445** | 0.0495 | 0.0327 |
| Dynamic confidence router | 0.8514 | 0.9696 | 0.9424 | **0.0486** | **0.0138** |

The dynamic router beat the strongest single route by 0.0099 MCC and won on three of five held-out folds. It essentially tied the uniform mean on MCC, while improving Brier score and ECE. Selective accuracy rose from 0.9391 at full coverage to 0.9691 at 80% coverage and 0.9872 at 50% coverage, supporting the usefulness of the confidence score for abstention/triage.

Frozen external evaluation used the OOF-selected threshold `0.545`:

| Dataset | Accuracy | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test1 | 0.9378 | 0.9142 | 0.8871 | 0.8451 | 0.9751 | 0.9505 |
| Test2 | 0.9416 | 0.8292 | 0.6531 | 0.6227 | 0.9356 | 0.7111 |

The fallback interaction behaved as intended in OOF: mean 1D weight increased from 0.319 in the joint high-2D/high-3D confidence group to 0.406 in the joint low-confidence group. External confidence shift weakened this effect. Since frozen Test2 MCC/F1 did not exceed the 3D default (`0.6320/0.6602`), the router is **not adopted as the default classifier** and is not retuned on test labels. It is retained for uncertainty reporting and as a documented negative result. Machine-readable artifacts are under `runs/triroute_confidence_oof/` and `runs/triroute_confidence_final/`.

### Frozen Default Model

The frozen default is ESM2-650M residue embeddings plus ESMFold C-alpha geometry, a PLDDT-aware spatial branch, and learned-confidence fusion.

- Checkpoint: `runs/plm_fusion_esm2_geometry_confidence/best_model.pt`
- Config: `configs/default.json`
- Seed: 42
- Checkpoint selection: validation MCC
- Decision threshold: 0.85
- Per-model checkpoints and thresholds were selected from training/validation data. Historical architecture development did inspect test1/test2 repeatedly, so these sets are not treated as pristine model-development-independent data in the new one-dimensional study.

| Dataset | N | Positive | Accuracy | BACC | Precision | Recall | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test1 | 1126 | 320 | 0.9378 | 0.9179 | 0.9058 | 0.8719 | 0.8885 | 0.8458 | 0.9699 | 0.9426 |
| Test2 | 582 | 46 | 0.9399 | 0.8481 | 0.5965 | 0.7391 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |

Confusion matrices at threshold 0.85:

| Dataset | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Test1 | 777 | 29 | 41 | 279 |
| Test2 | 513 | 23 | 12 | 34 |

![Default model ROC, PR, and confusion matrices](reports/figures/default_model_evaluation.png)

The vector version is `reports/figures/default_model_evaluation.pdf`; raw ROC and PR points are stored beside it.

### Multi-Seed Reproducibility

The default architecture was trained with seeds 42, 123, and 2025. Every run used the same fixed threshold 0.85. Values are mean +/- sample standard deviation.

| Dataset | Accuracy | BACC | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test1 | 0.9278 +/- 0.0107 | 0.8999 +/- 0.0174 | 0.8678 +/- 0.0213 | 0.8195 +/- 0.0276 | 0.9669 +/- 0.0027 | 0.9336 +/- 0.0079 |
| Test2 | 0.9341 +/- 0.0060 | 0.8086 +/- 0.0349 | 0.6119 +/- 0.0449 | 0.5782 +/- 0.0499 | 0.9225 +/- 0.0099 | 0.6356 +/- 0.0557 |

Test2 contains only 46 positive peptides, and its decision metrics show meaningful seed variance. The seed-42 result is the frozen single-run result, while the three-seed mean is the more conservative expected result.

### Bootstrap Confidence Intervals

Five thousand class-stratified hierarchical bootstrap iterations were used. Each replicate samples a model seed and then resamples positive and negative test examples separately.

| Dataset | MCC (95% CI) | F1 (95% CI) | AUROC (95% CI) | AUPRC (95% CI) |
| --- | --- | --- | --- | --- |
| Test1 | 0.8195 [0.7611, 0.8703] | 0.8678 [0.8239, 0.9063] | 0.9669 [0.9551, 0.9774] | 0.9336 [0.9078, 0.9557] |
| Test2 | 0.5782 [0.4419, 0.7109] | 0.6119 [0.4854, 0.7312] | 0.9225 [0.8766, 0.9599] | 0.6356 [0.4842, 0.7885] |

### Core Ablations

The table uses each experiment's documented, validation-selected protocol. Thresholds are shown because not every historical branch used 0.85.

| Experiment | Threshold | Test1 MCC | Test1 AUPRC | Test2 F1 | Test2 MCC | Test2 AUPRC | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Default learned-confidence fusion | 0.85 | 0.8458 | 0.9426 | 0.6602 | 0.6320 | 0.6969 | Retained default |
| ESM2 sequence-only | 0.85 | 0.8177 | 0.9367 | 0.6038 | 0.5709 | 0.7077 | Strong single branch; lower decision metrics |
| ProtT5 fusion | 0.85 | 0.8234 | 0.9426 | 0.6038 | 0.5709 | 0.6953 | ESM2 retained |
| Supervised contrastive loss | 0.85 | 0.8143 | 0.9303 | 0.6038 | 0.5709 | 0.6427 | Rejected |
| Focal BCE | 0.85 | 0.8159 | 0.9419 | 0.6471 | 0.6171 | 0.7273 | Ranking-oriented auxiliary result |
| Chemical structure features | 0.85 | 0.8037 | 0.9398 | 0.6383 | 0.6067 | 0.7243 | Ranking-oriented auxiliary result |
| ESM2 + atom-graph cross-attention | 0.25 | 0.8297 | 0.9311 | 0.6429 | 0.6183 | 0.6622 | Best atom-graph branch; below default |
| Small atom residual on default 3D | 0.43 | 0.8241 | 0.9339 | 0.6018 | 0.5728 | 0.6339 | Rejected |
| C-alpha local-frame geometry | 0.50 | 0.7953 | 0.9265 | 0.5200 | 0.4770 | 0.6440 | Rejected |
| Full-backbone geometry | 0.85 | 0.7626 | 0.9225 | 0.5814 | 0.5497 | 0.6641 | Rejected |
| High-pLDDT-only training | 0.85 | 0.7232 | 0.8925 | 0.5854 | 0.5593 | 0.5923 | Rejected |

No tested architecture exceeded the default model on the adoption criterion of test2 MCC while preserving F1.

### Structure-Quality Analysis

The frozen default model performs better on samples with mean pLDDT >= 0.70:

| Evaluation subset | N | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test1 full | 1126 | 0.8885 | 0.8458 | 0.9699 | 0.9426 |
| Test1 high-pLDDT | 671 | 0.9146 | 0.8878 | 0.9807 | 0.9633 |
| Test2 full | 582 | 0.6602 | 0.6320 | 0.9316 | 0.6969 |
| Test2 high-pLDDT | 333 | 0.7368 | 0.7221 | 0.9575 | 0.8022 |

Filtering training data to high-pLDDT samples reduced full-test performance, so confidence is used for analysis and dynamic fusion rather than hard training exclusion.

### CD-HIT 0.80 Protocol

Native CD-HIT 4.8.1 reduced training data from 6387 to 5183 representatives. Corrected train+test combined clustering produced strict sets of 771 test1 peptides and 425 test2 peptides.

| Model | Evaluation set | F1 | MCC | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: |
| Full-data default | Test1 strict | 0.8456 | 0.7925 | 0.9543 | 0.9064 |
| CD-HIT 0.80 retrained | Test1 strict | 0.8118 | 0.7562 | 0.9454 | 0.8852 |
| Full-data default | Test2 strict | 0.6279 | 0.5941 | 0.9274 | 0.6606 |
| CD-HIT 0.80 retrained | Test2 strict | 0.5278 | 0.4843 | 0.8950 | 0.5421 |

The full-data model remains stronger on identical strict subsets. CD-HIT retraining is retained as a protocol-alignment experiment and does not replace the default.

### One-Dimensional Structure-Free Backup

A leakage-controlled study used five fixed stratified sequence-group folds and compared the existing ESM2 Transformer sequence branch against multi-scale CNN and BiLSTM alternatives. The retained backup is ESM2-650M residue embeddings plus BiLSTM; multi-scale CNN variants were rejected.

| Model | OOF threshold | OOF MCC | OOF AUROC | OOF AUPRC | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| ESM2 Transformer | 0.5594 | 0.8365 | 0.9537 | 0.9139 | Control |
| ESM2 BiLSTM | 0.5000 | 0.8408 | 0.9629 | 0.9247 | Retained backup |

Frozen three-seed BiLSTM results:

| Dataset | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Test1 | 0.8768 +/- 0.0039 | 0.8285 +/- 0.0036 | 0.9709 +/- 0.0021 | 0.9432 +/- 0.0032 |
| Test2 | 0.6051 +/- 0.0182 | 0.5773 +/- 0.0222 | 0.9361 +/- 0.0122 | 0.6965 +/- 0.0178 |

The backup does not replace the 3D default because test2 MCC/F1 do not improve. It is retained because it requires no ESMFold inference, has stronger test2 ranking metrics, and performs comparatively well on low-pLDDT samples. The retained checkpoints and machine-readable external predictions are under `runs/sequence_1d_bilstm_full_seed*/` and `runs/external_toxinpred3_1d/`.

### Reporting Guidance

- Use the three-seed mean +/- standard deviation as the primary reproducibility result.
- Show the frozen seed-42 result as the selected single-run checkpoint, not as the expected performance of every run.
- Report AUROC and AUPRC together with MCC/F1 because test2 is highly imbalanced.
- Include bootstrap intervals and CD-HIT strict results when comparing against papers with redundancy-controlled protocols.
- Do not claim direct superiority over published models unless datasets and split protocols are identical.

Machine-readable supporting results are retained under `runs/bootstrap_ci/`, `runs/sequence_similarity/`, `runs/complementarity/`, `runs/external_toxinpred3_*/`, and the corresponding frozen checkpoint directories. This document is the single consolidated result record; superseded result Markdown files have been removed.


---

## 18. 已尝试路线、停止决定与后续优先级

以下进度记录用于防止重复实验。所有新工作必须遵守“不再根据 Test1/Test2 调参”的边界。

### 7. 当前最合理的新创新方向

残差专家 P1 已经失败，因此不建议直接继续在其上叠加复杂的收益路由。后续创新应分为“当前数据可安全推进”和“获得新外部数据后推进”两类。

#### 7.1 已执行并停止：冻结分类器的风险感知与选择性预测

执行结果：学习风险分数的 AURC 为 `0.018917`，弱于现有三通路置信度的 `0.018631`，相对变化 `-1.54%`。虽然 5 折中有 3 折获胜，且 80% 覆盖率下准确率较高，但没有达到“相对最强基线 AURC 至少提高 5%”的预设标准。该实现和运行目录已删除，不继续调参。

不再修改默认分类器，只学习：

```text
这个样本是否可信？
模型是否应该拒绝预测？
```

候选可靠度特征：

- 预测概率和熵；
- 多随机种子或 MC Dropout 方差；
- mean/min pLDDT；
- 低 pLDDT 残基比例；
- 与训练集最近序列的一致性；
- ESM2 embedding kNN 距离；
- 序列长度和组成异常度；
- 1D、2D、3D 预测分歧。

这一方向不改变原始毒性概率，只输出：

```text
accepted prediction
low-confidence warning
abstain / requires review
```

优势：

- 不会因为路由权重错误降低默认 MCC；
- 可以利用当前动态路由已证明有效的 Brier/ECE 和风险分层能力；
- 相对参考论文更容易形成“风险可控预测”的创新；
- 可以在 nested calibration 中完成，不需要继续查看 Test1/Test2。

#### 7.2 已执行并停止：Conformal 最终预测集合

执行结果：90% 目标下总体覆盖率 `0.8907`，非毒/毒性覆盖率 `0.8858/0.9032`，singleton rate `0.9789`，但空预测集率 `0.02114` 超过预设上限 `0.02`。未根据结果事后放宽门槛；实现和运行目录已删除。

可直接对冻结默认模型或冻结三模型均值进行 conformal calibration，输出：

- `{toxic}`；
- `{non-toxic}`；
- `{toxic, non-toxic}`，表示不确定；
- 极端 OOD 警告。

主要评价：

- 90% 目标覆盖率下的实际覆盖率；
- 固定覆盖率下的错误率；
- AURC；
- 高/低同源性分层覆盖；
- 高/低 pLDDT 分层覆盖。

这比继续追求小幅 MCC 提升更符合毒性预测的安全属性。

#### 7.3 获得更多独立外部集后：重新设计边际收益专家

当前残差模型训练时序列基础和残差分支共同更新，可能导致基础表示被残差训练干扰。新的版本可采用：

1. 先冻结训练完成的一维基础模型；
2. 固定 `z_1D`；
3. 只训练 `delta_2D` 和 `delta_3D`；
4. 对残差施加零均值、稀疏和正交约束；
5. 使用 leave-one-modality-out 收益作为监督；
6. 仅在新的 nested 协议和更多独立来源外部集上选择。

只有该残差基础稳定非劣，才继续边际收益路由。

#### 7.4 获得更多结构资源后：多构象不确定性

可以比较：

- ESMFold 不同扰动；
- ESMFold、OmegaFold、ColabFold；
- 多构象距离矩阵方差；
- 不同结构预测下 3D 分类概率方差。

多构象分歧比单一 pLDDT 更接近结构 epistemic uncertainty，但计算成本较高，应放在风险感知与 conformal 之后。

#### 7.5 数据创新优先于继续加深模型

如果目标是超过参考论文，最重要的改进之一不是增加网络层，而是建立新的独立外部验证集：

- 按文献来源或发布时间切分；
- 按实验平台切分；
- 按物种或毒性测定方式切分；
- 保证与训练集低同源；
- 保留足够正样本；
- 在任何模型开发前冻结标签和评估脚本。

---

### 8. 推荐的后续优先级

#### 第一优先级：论文与结果可信度

1. 明确区分历史冻结默认结果和 nested 创新筛选结果；
2. 报告多随机种子和 bootstrap 区间；
3. 不直接比较不同数据集上的 PeptiTox/ToxiPep 数值；
4. 固化已完成的 ToxinPred3 未使用外部验证结果；
5. 将负结果纳入论文消融和讨论。

#### 第二优先级：严格外部验证收尾

1. 使用 ToxinPred3 官方独立集，不使用其训练集扩充当前训练；
2. 排除与当前 train、Test1、Test2 的精确重复和 `>=0.8` 同源序列；
3. [已完成] 在原生 CD-HIT 4.8.1 环境复核当前 Biopython 回退筛查；
4. [已完成] 冻结评估 1D、2D、3D，未根据外部标签调阈值；
5. [已完成] 无论结果好坏均报告完整指标和 bootstrap 区间。

#### 第三优先级：新数据条件下重做收益专家

1. 冻结一维基础；
2. 独立训练 2D/3D 增量；
3. 收益监督；
4. OOD 感知启用；
5. conformal 收益下界。

#### 第四优先级：高成本几何扩展

1. 多构象；
2. 多结构预测器；
3. SE(3)-equivariant 方向特征；
4. 真实主链局部框架；
5. 构象不确定性传播。

---

### 9. 当前可用于论文的创新表述

在不夸大结果的前提下，可以将当前工作的贡献概括为：

1. 构建了结合 PLM 序列、原子拓扑和预测三维几何的多层次肽毒性预测框架；
2. 引入 pLDDT-aware 几何融合，并系统分析结构质量对预测性能的影响；
3. 建立无结构一维备份和二维原子图对照，分析多模态错误互补；
4. 构建 group-aware OOF 和严格 nested group CV，分离训练、模型选择、校准和外层测试；
5. 系统报告动态路由、残差专家和多种结构/训练改进的正负结果；
6. 证明多模态信息能够改善排序和校准，但决策指标提升存在明显分布依赖；
7. 为后续风险可控、可拒识的多肽毒性预测提供实验基础。

暂时不应声称：

- 已达到普遍 SOTA；
- 已显著超过所有参考论文；
- Test1/Test2 是从未参与开发的独立测试集；
- 当前 3D 分支是纯几何模型；
- 动态路由或残差专家已经稳定提升最终分类性能。

---

### 10. 最终结论

当前项目已经完成从一维序列基线到 2D 原子图、3D 几何、多通路可靠度、严格 nested 验证和 ToxinPred3 冻结外部验证的完整探索。冻结默认 3D 模型仍是历史 Test1/Test2 上的主要决策模型；一维 BiLSTM 在新的严格外部集上迁移最好，是重要的无结构备份；二维原子图作为互补消融；动态路由保留为可靠度分析；残差专家、选择性风险学习器和 conformal 均因未通过预设门槛而停止。

当前论文阶段不再继续微调 Test1/Test2。已经形成并应在论文中凝练的证据链是：

```text
冻结分类器
+ ToxinPred3 严格低同源独立集
+ 已完成的原生 CD-HIT 复核
+ 已完成的 1D/2D/3D 一次性外部评估
+ 论文结果可信度
```

原生 CD-HIT 复核、1D/2D/3D 冻结外部评估和结果整合均已完成。严格外部集显示 1D 优于 2D/3D，3D 的高 precision 伴随明显 recall 损失。当前下一步是完成全评估集的分层生物学解释和论文交付，不重启边际收益专家；只有获得更多独立来源数据，或提出不重复现有消融的明确新假设时，才考虑“冻结一维基础 + 独立 2D/3D 边际收益专家”。
