# One-Hot Encoding 实战：心脏病预测中的类别特征编码

> 一个面向初学者的端到端项目：用 UCI Cleveland 心脏病数据集，通过对照实验展示
> "类别特征如何编码"对模型性能与可解释性的影响。

## 项目动机

机器学习模型只能做数学运算，而真实临床数据里充满了类别变量（胸痛类型、扫描结论、性别）。
编码方式选错——比如把无序类别当数字直接喂给模型——会强加虚假的顺序假设。
本项目用同一模型、同一评估协议，对比三种编码策略，量化这个决策的影响。

## 目录结构

```
One-Hot/
├── README.md                                # 本文件
├── 01.Data/                                 # 全部数据
│   ├── processed.cleveland.data             # UCI 原始数据（无表头，缺失值以 ? 伪装）
│   └── warmup_patients.csv                  # 教学用热身小数据
└── 02.Code/
    └── heart_encoding_portfolio.ipynb       # 完整分析 Notebook（已执行，含输出）
```

## 数据

- 来源：[UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- 303 例患者、14 列；删除 6 例含缺失值后 297 例
- 目标二值化：`num` 0 → 无病（160 例），1~4 → 有病（137 例）
- 特征分组：连续数值 5 列（StandardScaler），类别 8 列（OneHotEncoder）

## 实验设计

控制变量：同一逻辑回归（`max_iter=5000`）、同一 5 折交叉验证（`KFold(shuffle=True, random_state=42)`）、
同一指标（ROC AUC）。唯一变量是编码方式。

| 实验 | 编码方式 | 泄漏风险 | 5 折平均 AUC |
|---|---|---|---|
| A | Label encoding（类别当数字） | 无 | 0.8910 |
| B | `pd.get_dummies`（CV 前编码） | 轻微（类别清单） | 0.9023 |
| C | `OneHotEncoder` + `ColumnTransformer` + `Pipeline` | 无 | **0.9067** |

## 主要发现

1. **One-hot 优于 label encoding，但差距诚实地说并不大**（+0.011~0.016，小于折间标准差 0.03~0.04）。
   原因：类别数最多只有 4，逻辑回归能部分补偿强加的顺序。方向与理论一致，小样本上不显著。
2. **B ≈ C**：两者数学上是同一种编码，C 的优势在工程——零数据泄漏、
   `handle_unknown="ignore"` 兜底未见类别、可整体部署。
3. **系数解读验证了编码理论**：one-hot 学到的胸痛类型效应呈"锯齿形"
   （cp_1: -0.70 → cp_2: +0.18 → cp_3: -0.60 → cp_4: +1.11），
   无症状型（cp_4）比典型心绞痛（cp_1）危险 1.81 个 log-odds——
   这种非单调模式是 label encoding 的单系数直线**原理上无法表达**的。
4. **标准化的实验证据**：不标准化时逻辑回归报 `ConvergenceWarning`；
   Pipeline 中加入 `StandardScaler` 后警告消失。

## 关键教训（初学者向）

- dtype 是数字 ≠ 变量是数值：本数据 13 列全是 float64，其中 8 列是类别。
- 判断类别的标准不是"长什么样"，而是"对它做算术有没有意义"。
- 所有需要"学习"的步骤（编码器、标准化器、模型）只准看训练数据——
  `Pipeline` 把这条纪律变成了结构保证。
- 高基数（类别数过多）是 one-hot 的天花板：列爆炸 + 每类样本过少 → 过拟合。

## 核心代码：模型训练模块

完整训练流程分四步，全部代码可在 `02.Code/heart_encoding_portfolio.ipynb` 中运行。

### 第 1 步：数据加载与准备

```python
import pandas as pd

COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# 读取数据：无表头手动命名；缺失值在原始文件中以 ? 伪装，翻译成 NaN
df = pd.read_csv("../01.Data/processed.cleveland.data",
                 header=None, names=COLS, na_values="?")

# 清洗：删除 6 例缺失（约 2%）；目标二值化：num 0→无病，1~4→有病
df_clean = df.dropna().copy()
df_clean["target"] = (df_clean["num"] > 0).astype(int)

NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]   # 连续数值 → 标准化
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]  # 类别 → one-hot

X = df_clean[NUM_COLS + CAT_COLS]
y = df_clean["target"]
```

### 第 2 步：构建预处理 + 模型的完整 Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# 两路预处理：数值列标准化，类别列 one-hot 编码
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),  # ignore 兜底未见类别
])

# 预处理与模型捆成一个整体：交叉验证时每一折自动重新 fit，杜绝数据泄漏
pipe = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=5000)),
])
```

### 第 3 步：5 折交叉验证评估（ROC AUC）

```python
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

print(f"5 折平均 AUC: {scores.mean():.4f} ± {scores.std():.4f}")
# 输出：5 折平均 AUC: 0.9067 ± 0.0269
```

### 第 4 步：训练最终模型并解读 one-hot 系数

```python
pipe.fit(X, y)

# 提取编码后的特征名与对应系数
feature_names = pipe.named_steps["prep"].get_feature_names_out()
coefs = pipe.named_steps["model"].coef_[0]

coef_table = pd.DataFrame({"feature": feature_names, "coef": coefs})
print(coef_table.sort_values("coef", ascending=False).to_string(index=False))

# 关键解读（相对基准的 log-odds 偏移）：
#   cat__cp_4.0  +1.11   无症状型胸痛最危险（临床共识）
#   cat__thal_7.0 +0.83  铊扫描可逆性缺损提示缺血
#   num__thalach -0.41   最大心率越高，心肺储备越好，患病概率越低
```

## 局限（诚实声明）

- 单一数据集、297 例小样本，结果为 discovery-only，无外部验证队列。
- 实验 B 存在轻微的类别清单泄漏（`get_dummies` 在 CV 前执行），已在实验设计中注明；
  实验 C 从机制上消除了该泄漏。
- `slope`、`ca` 按类别编码处理；`ca` 作为计数值也可按数值特征处理，两种路线均合理。

## 复现

```bash
pip install pandas scikit-learn
jupyter notebook 02.Code/heart_encoding_portfolio.ipynb
```

Notebook 通过相对路径 `../01.Data/processed.cleveland.data` 读取数据，
保持目录结构不变即可直接运行。
