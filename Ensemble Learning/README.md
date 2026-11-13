# Ensemble Learning Toolkit — 集成学习工具包

可复用的树模型集成学习代码库：**随机森林（Bagging）+ GBDT 家族（Boosting）**，统一接口，专为生物学数据分类任务设计。

## 生物学中的常用任务

| 任务领域 | 典型问题 | 推荐算法 |
|----------|----------|----------|
| 变异致病性预测（VUS 分类） | 判断 SNP/突变是否致病 | XGBoost |
| ADMET / QSAR | 化合物吸收、毒性、活性预测 | LightGBM、XGBoost |
| 大规模虚拟筛选 | 百万级化合物库排序 | LightGBM |
| 临床风险预测 | 疾病风险分层、生物标志物面板 | XGBoost、LightGBM |
| 细胞类型注释 | scRNA-seq 高维表达 → 细胞标签 | LightGBM |

## 优缺点对比

### Bagging vs Boosting

| | 随机森林 | GBDT 家族 |
|---|---|---|
| 准确率 | 好 | **通常更高**（表格数据常胜） |
| 过拟合风险 | 低（对噪声稳健） | 较高（对离群值敏感） |
| 超参数 | 少，默认即可 | 多，需要交叉验证调参 |
| 训练速度 | 可完全并行 | 串行本质（LightGBM 已大幅缓解） |
| 内置评估 | OOB 免费评估 | 需验证集做早停 |

### 家族内部

| 算法 | 优点 | 缺点 |
|------|------|------|
| XGBoost | 精度高、文档全、工具丰富 | 超参数多，调参成本高 |
| LightGBM | 大数据极快、内存省 | 小数据上 leaf-wise 易过拟合 |
| CatBoost | 类别特征免编码、默认参数强 | 训练通常最慢 |

## 文件结构

```
ensemble-learning-toolkit/
├── README.md               # 本文件：算法总结 + 使用指南
├── ensemble_models.py      # 可复用模块（5 种模型统一接口）
├── demo_breast_cancer.py   # 演示脚本：乳腺癌数据全流程
├── model_comparison.csv    # 演示输出：五模型指标对比表
└── figures/                # 演示输出：评估图 + 特征重要性图
```

## 安装

```bash
pip install scikit-learn xgboost lightgbm catboost pandas matplotlib seaborn
```

## 快速上手

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ensemble_models import (
    get_model, train_and_evaluate, compare_models,
    plot_confusion_and_roc, plot_feature_importance,
)

# 1. 数据（示例：乳腺癌良恶性分类，0=恶性，1=良性）
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. 一行训练 + 评估（可选 GridSearchCV 调参）
model, metrics = train_and_evaluate(
    'xgboost', X_train, y_train, X_test, y_test,
    tune=True,               # 5 折交叉验证调参
    scoring='recall',        # 癌症场景重点看漏诊率
    positive_label=0,        # 恶性(0)为正类
)
print(metrics)               # accuracy / recall / precision / f1 / auroc

# 3. 可视化
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
plot_confusion_and_roc(y_test, y_pred, y_proba, 'xgboost',
                       class_names=('Malignant', 'Benign'))
plot_feature_importance(model, X_test, y_test, data.feature_names, top_k=10)

# 4. 多模型对比
results = {}
for name in ['rf', 'gbdt', 'xgboost', 'lightgbm', 'catboost']:
    _, results[name] = train_and_evaluate(name, X_train, y_train, X_test, y_test)
print(compare_models(results))   # 按 AUROC 排序的对比表
```

## 支持的模型

| 名称 | 类 | 路线 | 一句话特点 |
|------|-----|------|-----------|
| `rf` | RandomForestClassifier | **Bagging** | 并行种树 + 投票，稳健，默认参数就能用，自带 OOB 评估 |
| `gbdt` | GradientBoostingClassifier | **Boosting** | 串行拟合残差的原型，sklearn 内置 |
| `xgboost` | XGBClassifier | Boosting | 二阶泰勒 + 正则化，精度高，生态最成熟 |
| `lightgbm` | LGBMClassifier | Boosting | 直方图加速 + leaf-wise 生长，大数据集最快 |
| `catboost` | CatBoostClassifier | Boosting | 有序提升 + 原生类别特征，默认参数强 |

## 算法原理速查

### 两条集成路线

```
Bagging（随机森林）                    Boosting（GBDT 家族）
──────────────────                    ──────────────────
并行训练 N 棵独立的树                  串行训练，第 k 棵树纠正前 k-1 棵
每棵树看不同的 bootstrap 子集          每棵树拟合当前预测的残差（负梯度）
最终决策 = 多数投票                    最终决策 = 所有树的加权和
降低 → 方差（靠多样性）                降低 → 偏差（逐步逼近）
```

### GBDT 家族的演进（相对手写原型）

| 算法 | 核心改进 |
|------|----------|
| GBDT (2001) | 原型：每轮拟合残差 `F += lr * tree.predict(X)` |
| XGBoost (2016) | + 二阶泰勒展开（收敛更快）+ 树复杂度正则项 + 缺失值自动处理 |
| LightGBM (2017) | + 直方图分桶找分裂点（快几个数量级）+ 按叶子生长 |
| CatBoost (2018) | + 有序提升（对抗过拟合）+ 类别特征免编码 |

## 选型建议

1. **小数据 + 快速出结果** → `rf`（默认参数 + OOB）
2. **榨取最后几个点准确率** → `xgboost` / `lightgbm` + 调参（`tune=True`）
3. **数据量很大（百万行）** → `lightgbm`
4. **特征含大量类别变量**（突变类型、染色体等）→ `catboost`

## 实践要点（来自学习笔记）

- **随机森林不需要标准化**：树模型基于阈值分裂，与特征尺度无关（SVM/KNN 才需要）
- **测试集只能用一次**：调参用训练集内的交叉验证，测试集留到最后
- **癌症场景重点看 Recall**：漏诊（恶性→良性）比误诊后果严重
- **AUROC 用 `predict_proba`**：不要用 `predict` 的硬标签
- **特征重要性 ≠ 因果**：报告写"与良恶性相关"，因果需实验验证
- **排列重要性更可靠**：比内置 Gini 重要性更客观，且对 5 种模型接口统一

## 复现演示

```bash
python demo_breast_cancer.py
```

输出：五模型对比表（`model_comparison.csv`）+ 最优模型的混淆矩阵/ROC 图和 Top-10 特征重要性图（`figures/`）。
