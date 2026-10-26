# GPR Enzyme Kinetics Tutorial

用高斯过程回归（GPR）从 8 个带噪声的酶动力学测量点拟合 Michaelis-Menten 曲线，
并进行诚实的不确定性量化。面向零基础初学者的完整入门项目。

## 项目亮点

- **小数据建模**：仅 8 个测量点，展示核函数先验在小样本下的价值
- **不确定性量化**：95% 置信带随数据覆盖自动伸缩——插值区窄、外推区宽
- **对比实验**：GPR vs 4 阶多项式，外推区预测 -748 vs 诚实置信带
- **超参数敏感性**：亲手看长度尺度和噪声参数如何塑造曲线行为
- **完全可复现**：固定随机种子（seed=42），数据 CSV 随仓库提供

## 快速开始

```bash
pip install -r requirements.txt
jupyter lab notebooks/gpr_enzyme_kinetics.ipynb
```

## 核心结果

| 模型 | S = 4 mM 处预测（真实值 10.9） | 行为 |
|---|---|---|
| 4 阶多项式 | **-747.8** | 自信地错，毫无预警 |
| GPR | 3.6 ± 11.4（95%） | 均值回落到先验，置信带诚实变宽 |

> 一个不带不确定度的预测值，只有一半信息。
> GPR 的置信带不是装饰——它是模型在告诉你：哪里在插值（窄），哪里在猜（宽）。

## 图表预览

| 图 | 内容 |
|---|---|
| `figures/project_step1_data.png` | 模拟数据 vs 真实曲线 |
| `figures/project_step2_gpr_fit.png` | GPR 拟合 + 95% 置信带 |
| `figures/project_step3_gpr_vs_poly.png` | GPR vs 多项式外推对比 |
| `figures/project_step4_hyperparams.png` | 长度尺度与噪声的四面板敏感性实验 |

## 仓库结构

```
├── README.md                     # 本文件
├── requirements.txt              # 依赖
├── data/
│   └── enzyme_kinetics_data.csv  # 8 点模拟数据（seed=42）
├── notebooks/
│   └── gpr_enzyme_kinetics.ipynb # 完整分析，逐行中文注释
├── figures/                      # 4 张核心图
└── docs/
    └── gpr_scope_of_applicability.md  # GPR 适用范围详解
```

## 学习路线

本项目按"10 题测验 → 动手项目"的导师制流程设计，覆盖：

1. 回归与不确定性思维
2. 高斯分布与条件化
3. 核函数直觉与常见核（RBF / 周期 / 线性）
4. 先验 → 后验的贝叶斯更新
5. 超参数与边际似然自动调参
6. 预测区间解读
7. GPR 的适用边界（见 `docs/gpr_scope_of_applicability.md`）

## License

MIT
