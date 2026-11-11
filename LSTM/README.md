# RNN / LSTM 原理与蛋白二级结构预测实战

> 本 README 整理自一次完整的"从概念到实验"学习路径：RNN/LSTM 的架构原理 → 在 CB513 蛋白二级结构预测上的两个完整对照实验（序列谱特征 vs ESM2 预训练 embedding）。所有代码可直接运行，实验结果可复现。

## 0. 模型选择：LSTM 适合哪些生物学任务

**判断标准一句话：数据有"顺序依赖"，就有 LSTM 的用武之地。** 生物学里的序列和时间过程无处不在，按任务模式分四类：

### 四类适用任务

| 任务模式 | 生物学任务举例 |
|---|---|
| **逐位置标注**（多对多） | 蛋白二级结构预测 [1]；磷酸化/糖基化等修饰位点；剪接位点、启动子/增强子定位；信号肽与跨膜区；抗原表位 |
| **整条序列分类/回归**（多对一） | 蛋白溶解度/稳定性/表达量；酶-非酶与蛋白家族分类；突变致病性预测 |
| **真正的时间过程**（时间步 = 采样时刻） | 构象动力学与 MD 轨迹建模 [2, 3]；ECG/EEG/ICU 生理信号；纵向临床数据与疾病进展；时序组学与细胞分化轨迹；药代动力学曲线 |
| **序列生成**（自回归） | SMILES 药物分子生成；蛋白/抗体序列设计；启动子/元件设计 |

### 什么时候不用 LSTM

| 情形 | 更合适的工具 |
|---|---|
| 超长序列（基因组级上下文） | Transformer / 长上下文架构 |
| 位置**对**之间的交互（共进化、接触图） | 2D CNN / Attention |
| 图结构数据（PPI 网络、代谢通路） | 图神经网络（GNN） |
| 静态图像（病理切片、显微图像） | CNN / ViT |
| 极小数据集 + 需要可解释概率 | HMM / 贝叶斯方法 |

### 实践建议

中小规模序列任务（几百到几万条样本、序列长几十到几百位），LSTM/GRU 是便宜且强的基线。动手前先问一句：**打乱序列顺序后，标签还成立吗？** 若成立（任务只依赖组成信息、不依赖顺序），RNN 的结构优势用不上，普通网络就够了。

## 目录

0. [模型选择：LSTM 适合哪些生物学任务](#0-模型选择lstm-适合哪些生物学任务)
1. [RNN 基础](#1-rnn-基础)
2. [LSTM 原理](#2-lstm-原理)
3. [实战：蛋白二级结构预测](#3-实战蛋白二级结构预测q3-三态分类)
4. [特征工程：从序列谱到 ESM2](#4-特征工程从序列谱到-esm2)
5. [实践清单：常见坑](#5-实践清单常见坑)
6. [文件清单与复现方法](#6-文件清单与复现方法)
7. [延伸阅读](#7-延伸阅读)

---

## 1. RNN 基础

### 1.1 什么是循环神经网络

循环神经网络（RNN）是处理**序列数据**的网络：当前时刻的输出不仅取决于当前输入，还取决于之前所有输入留下的"记忆"（隐状态）：

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

**核心性质——权重沿时间共享**：同一个隐层（同一组权重）在每个时间步被重复调用。序列再长也不增加参数；一个模式在任何时间位置学到，对所有位置生效。

### 1.2 三个最容易混淆的维度

| 参数 | 含义 | 谁决定 |
|---|---|---|
| `seq_len` | 序列的时间步数（隐层被调用的次数） | **数据**：每条样本的窗口长度 |
| `hidden_size` | 隐状态向量的维度（隐层神经元个数） | 超参数 |
| `num_layers` | 纵向堆叠的隐层个数 | 超参数 |

关键区分（初学最常见的混淆）：

- **时间步 ≠ 层**。时间步是"同一层被横向调用多少次"（权重共享，参数量不变）；层是"纵向叠了几个不同的隐层"（各有各的权重，参数量翻倍）
- `hidden_size=32` 是**一层里有 32 个神经元**（h 是 32 维向量），不是 32 层
- 层内神经元之间**不**共享权重；共享只发生在时间步之间
- 一共有多少个 h？每步一个，共 `seq_len` 个；每个 h 多大？`hidden_size` 维

### 1.3 输入格式

PyTorch 的 RNN/LSTM/GRU 输入是三维张量，三连问定形状：

```
(batch_size, seq_len, input_size)
  多少条序列   每条多长    每步几个特征
```

| 数据类型 | input_size | 说明 |
|---|---|---|
| 单变量时间序列 | 1 | 别忘了 unsqueeze 特征维 |
| 多变量时间序列 | 特征数 | 直接拼在最后一维 |
| 文本 | embedding_dim | 词 ID 先过 `nn.Embedding`；不是词表大小 |
| 蛋白残基 | 21 / 43 / 480 | one-hot / one-hot+谱 / ESM2 embedding |

对应的标签（target）形状：多对一任务 `(batch,)`；多对多（逐残基/逐步）任务 `(batch, seq_len)`。

### 1.4 变长序列处理

不同长度的序列要拼进同一个 batch，标准三件套：

```python
# ① Dataset 只取真实长度
# ② collate 里补齐：特征补 0，标签补 -100
X = pad_sequence(xs, batch_first=True)                       # (B, Lmax, D)
Y = pad_sequence(ys, batch_first=True, padding_value=-100)   # (B, Lmax)
# ③ pack 让 RNN 跳过 padding；损失忽略 -100
packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

两个机制缺一不可：`pack_padded_sequence` 让 padding 不进 RNN（不污染隐状态、不浪费计算）；`ignore_index=-100` 让 padding 不进损失（不产生梯度）。

### 1.5 训练：权重从哪来，BPTT 是什么

隐层权重不是人设定的：**随机初始化 → 前向算损失 → 反向传播算梯度 → 梯度下降更新**，循环迭代。对 RNN，反向传播沿时间展开（BPTT）：因为权重在所有时间步共享，所有时间步的梯度**累加**到同一组权重上——每条序列提供 seq_len 份学习信号。

前向是逐步顺序走的（不能并行，这是 RNN 慢于 Transformer 的根源）；反向是一次性从最后一步传回第一步。训练中变化的是权重，不是序列长度。

---

## 2. LSTM 原理

### 2.1 门控机制与细胞状态

LSTM 在隐状态 $h_t$ 之外增加一条**细胞状态** $c_t$（记忆传送带），用三个门控制信息流：

```python
f = sigmoid(W_f @ [h_prev, x] + b_f)   # 遗忘门：旧记忆保留多少（0 擦除 ~ 1 全留）
i = sigmoid(W_i @ [h_prev, x] + b_i)   # 输入门：新信息写入多少
g = tanh(W_g @ [h_prev, x] + b_g)      # 候选新记忆
c = f * c_prev + i * g                 # ★ 细胞状态：加法更新（关键！）
o = sigmoid(W_o @ [h_prev, x] + b_o)   # 输出门：暴露多少给外界
h = o * tanh(c)
```

### 2.2 为什么能缓解梯度消失

普通 RNN 的梯度每步要乘一个矩阵 $W_h^\top \mathrm{diag}(\tanh')$，连乘几十步指数消失。LSTM 细胞状态的梯度因子是：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{（逐元素的门控值）}$$

遗忘门是**网络可以学着调成 ≈1 的**——网络学会"这段历史重要，别忘"，梯度就沿加法传送带近乎无损地穿越长距离（恒定误差传送带，Hochreiter & Schmidhuber, 1997）。

结构上，$c_t = \underbrace{f_t \odot c_{t-1}}_{\text{恒等通路}} + \underbrace{i_t \odot g_t}_{\text{学习增量}}$ 与 ResNet 残差连接 $y = x + F(x)$ 是同一思想：给信息留一条不经过变换的高速路。

**边界**：LSTM 是缓解而非消除——门关小时梯度照样衰减；梯度爆炸仍需裁剪；上千步的长依赖仍是 Transformer 的主场。

### 2.3 RNN / GRU / LSTM 速查

| | 输入格式 | 返回 | 参数量（相对） | 适用 |
|---|---|---|---|---|
| `nn.RNN` | 相同 | `out, h_n` | 1× | 教学/极短序列 |
| `nn.GRU` | 相同 | `out, h_n` | ~3× | 中小数据的默认起点 |
| `nn.LSTM` | 相同 | `out, (h_n, c_n)` | ~4× | 长依赖、精度优先 |

三者输入接口完全一致，换模型只改一行构造。

---

## 3. 实战：蛋白二级结构预测（Q3 三态分类）

### 3.1 任务与数据

**任务**：给定蛋白中每个残基，预测其二级结构——H（α 螺旋）/ E（β 折叠）/ C（无规卷曲）。这是"多对多"逐残基分类任务。

**数据**：CB513 数据集（Zhou & Troyanskaya, ICML 2014），514 条蛋白、84,765 个残基，每条蛋白存为 700×57 矩阵（不足 700 的用 NoSeq 标记补齐）：

| 特征列 | 内容 |
|---|---|
| [0, 22) | 氨基酸 one-hot（20 标准残基 + X + NoSeq） |
| [22, 31) | Q8 二级结构标签（8 态 + NoSeq），实验中映射为 Q3 |
| [31, 33) | N/C 末端标记 |
| [33, 35) | 溶剂可及度（本实验未用） |
| [35, 57) | 序列谱 profile（PSI-BLAST，22 维） |

数据来源：Zenodo record 7764556（数据集作者维护的官方镜像；注意文件名为 .gz 但实为未压缩 .npy，md5 `a6ca35acd2a453b0186e3fe4af23011f`）。

### 3.2 五步流程（LSTM 使用模板）

```
① 编码        每个时间步 → 定长向量（残基 = one-hot + 谱；文本 = Embedding；时序 = 归一化特征）
② 变长处理    Dataset 截取真实长度 → collate 补齐 → pack_padded_sequence
③ 模型        nn.LSTM(..., bidirectional=True) + 作用在每个时间步上的输出头（多对多）
④ 训练循环    loss.backward()（= BPTT）→ clip_grad_norm_ → opt.step()；按验证集选最优
⑤ 评估        总体准确率 + 每类 P/R/F1 + 混淆矩阵
```

### 3.3 关键代码

模型（两个实验共用，仅输入维度不同）：

```python
class BiLSTMSS(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)   # 双向 → 2*hidden

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.fc(self.dropout(out))                   # (B, L, 3) 逐残基 logits
```

训练循环核心（BPTT + 防爆炸 + 忽略 padding）：

```python
crit = nn.CrossEntropyLoss(ignore_index=-100)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1, 16):
    for X, Y, lengths in train_loader:
        logits = model(X, lengths)
        loss = crit(logits.reshape(-1, 3), Y.reshape(-1))
        opt.zero_grad(); loss.backward()                    # BPTT
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    # 每轮在验证集上评估，保留最优模型（早停思想）
```

### 3.4 实验设置与结果

统一协议：380/50/84 划分（seed 42）、BiLSTM（hidden 64、2 层、dropout 0.5）、Adam 1e-3、15 epochs、梯度裁剪 5.0、按验证集准确率选最优模型。测试集 13,393 个残基。

| 输入特征 | 维度 | 测试集 Q3 | H F1 | E F1 | C F1 |
|---|---|---|---|---|---|
| one-hot + 序列谱 | 43 | 74.5% | 0.785 | 0.668 | 0.756 |
| **ESM2 t12_35M embedding** | 480 | **77.1%** | **0.822** | **0.719** | **0.759** |

结论：**ESM2 全面胜出（+2.6 个百分点），三类全部提升**，提升最大的是原本最难的 E 折叠类。下游 BiLSTM 完全相同，差异全部来自输入表示。参考基线：Heffernan et al. 2017 的 BiLSTM（更复杂特征 + 全量训练集）Q3 ≈ 84% [1]。

---

## 4. 特征工程：从序列谱到 ESM2

### 4.1 序列谱（profile / PSSM）

序列谱 = 每个位置上"进化投票"的统计：用 PSI-BLAST 搜出同源序列、做多序列比对，在每个位置统计 20 种氨基酸的出现频率。它回答"进化上这个位置**倾向于**是什么残基"（one-hot 回答"实际是什么"）。

对二级结构预测特别有用，因为**二级结构比序列本身保守**：α 螺旋的疏水残基每 3~4 位周期出现、β 折叠交替出现，这种模式在几百条同源序列的谱里反复强化。谱特征能把 Q3 从 ~70% 拉到 ~80%。

注意：序列谱是**进化保守性**（单位置统计），严格说的**共进化**指位置对之间的关联（DCA/Potts 耦合，L×L 矩阵）——后者塞不进逐残基的 LSTM 输入，需要二维网络或注意力机制处理。

### 4.2 ESM2 预训练 embedding

ESM2 是在海量蛋白序列上以掩码语言模型目标预训练的模型，其逐残基 embedding 把进化信息压缩进稠密向量。用法三步：

```python
import esm  # fair-esm
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()   # 权重自动下载（134 MB）
converter = alphabet.get_batch_converter()
_, _, tokens = converter([("p1", "MKTAYIAK...")])        # (B, L+2) 含 BOS/EOS
out = model(tokens, repr_layers=[model.num_layers])
emb = out["representations"][model.num_layers][:, 1:L+1] # 去掉 BOS，取前 L 个残基 → (B, L, 480)
```

工程要点：按长度排序分批推理减少 padding 浪费；embedding 缓存到磁盘（npz），重跑训练不重复推理；**缓存文件名包含数据规模、加载时校验长度**（否则会踩到"冒烟测试缓存污染完整实验"的坑）。

### 4.3 特征拼接范式与线性投影层

经典范式（2015-2019）：`输入 = [one-hot | PSSM | 其他预测特征]` 沿特征维拼接。现代范式（2020-）：打底编码换成预训练 embedding，拼接结构不变。拼接时注意：各块特征尺度差异大，先各自标准化；ESM embedding 已隐含残基身份，再拼 one-hot 增益通常很小（建议消融实验）。

特征拼接后维度暴涨（如 523 维）时，可在 LSTM 前加**线性投影层** `nn.Linear(523, 128)`。要点：它是**可学习的、任务导向的降维**（与 LSTM 端到端一起训练，不是 PCA 那样的训练前预处理）；它只做线性重新混合，**不做抽象**——抽象化是 LSTM 的职责；收益是 LSTM 输入权重随 input_size 线性增长，先降维能省参数、让 LSTM 专注时间模式。

---

## 5. 实践清单：常见坑

1. **形状**：`print(x.shape)` 确认 `(batch, seq_len, input_size)`；不设 `batch_first=True` 时前两维含义颠倒，不报错但结果错
2. **归一化**：数值特征务必标准化（本实验的 one-hot/ESM 例外，但拼接多源特征时各块要分别标准化）
3. **时间切分**：时间序列按时间顺序切训练/测试集；窗口样本可打乱，但窗口内部时序不能乱
4. **双向的使用条件**：`bidirectional=True` 会看到"未来"，只适合文本/逐残基标注这类非因果任务；预测未来的时序任务禁用
5. **梯度裁剪**：RNN/LSTM 训练标配 `clip_grad_norm_`，防梯度爆炸
6. **hidden_size 选择**：从 32/64 起步，看验证曲线调——训练/验证损失都高 → 加大；训练低验证高 → 减小或加正则
7. **padding 双保险**：`pack_padded_sequence` + `ignore_index=-100` 缺一不可
8. **缓存一致性**：缓存中间结果（如 embedding）时，文件名带数据规模、加载时校验，防止不同规模实验互相污染

---

## 6. 文件清单与复现方法

```
ss_lstm_tutorial/
├── README.md                  # 本文件
├── download_data.sh           # 数据下载脚本（含 md5 校验与 .gz 改名处理）
├── train_ss_bilstm.py         # 实验一：one-hot + 序列谱基线（Q3 74.5%）
├── train_ss_bilstm_esm.py     # 实验二：ESM2 embedding（Q3 77.1-77.4%）
├── ss_lstm_tutorial.ipynb     # Notebook 版：两部分教程 + 全部图表
├── training_curve.*           # 实验一训练曲线
├── confusion_matrix.*         # 实验一混淆矩阵
├── esm_training_curve.*       # 实验二训练曲线
└── esm_confusion_matrix.*     # 实验二混淆矩阵
```

依赖：`torch`、`numpy`、`matplotlib`（实验二另需 `fair-esm`）。

```bash
# 一键复现（自动下载数据，CPU 约 5 分钟 / 实验）
python train_ss_bilstm.py
python train_ss_bilstm_esm.py          # 首次运行自动下载 ESM2 权重并提取 embedding

# 快速试跑 / 自定义
python train_ss_bilstm_esm.py --epochs 3 --max-proteins 100
python train_ss_bilstm_esm.py --esm-model esm2_t33_650M_UR50D   # 更强但更慢
```

升级到文献标准协议：用 Zenodo 同一 record 下的 2 GB `cullpdb+profile_6133.npy.gz` 训练、CB513 仅作测试，预期 Q3 80%+。

---

## 7. 延伸阅读

- [1] Heffernan et al., Bioinformatics 2017 —— BiLSTM 二级结构预测（本实验的文献基线，Q3 ≈ 84%）
- [4] Zhou & Troyanskaya, ICML 2014 数据集（Zenodo 镜像）—— CB513 数据与格式文档
- [2] López-Correa et al., Scientific Reports 2023 —— LSTM 直接预测 GPCR 分子动力学轨迹（构象时间序列建模）
- [5] Mardt et al., Nature Communications 2017 —— VAMPnets：深度学习从 MD 轨迹学动力学模型
- [3] Shen et al., NeurIPS 2025 —— 自回归模型同时建模蛋白构象与动力学
- [6] Lewis et al., Science 2025 —— BioEmu：生成式蛋白构象系综仿真

## References

1. Rhys Heffernan, Yuedong Yang, K. Paliwal, et al. Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility. Bioinformatics. 2017. doi:10.1093/bioinformatics/btx218.
2. Juan Manuel López-Correa, Caroline König, Alfredo Vellido. GPCR molecular dynamics forecasting using recurrent neural networks. Scientific Reports. 2023. doi:10.1038/s41598-023-48346-4.
3. Shen Y, Wang L, Yuan H, Wang Y, Yang B, Gu Q. Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression. Advances in Neural Information Processing Systems 38. 2025. doi:10.52202/085713-3666.
4. Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction, 2014 | Zenodo. [cited 2026 Nov 10]. Available from: https://zenodo.org/records/7764556
5. Andreas Mardt, Luca Pasquali, Hao Wu, et al. VAMPnets for deep learning of molecular kinetics. Nature Communications. 2017. doi:10.1038/s41467-017-02388-1.
6. Lewis S, Hempel T, Jiménez-Luna J, Gastegger M, Xie Y, Foong AYK, et al. Scalable emulation of protein equilibrium ensembles with generative deep learning. Science. 2025. doi:10.1126/science.adv9817.
