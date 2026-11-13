"""
demo_breast_cancer.py
=====================
演示：用 ensemble_models 工具包在乳腺癌数据上跑全部 5 种集成学习模型，
输出对比表 + 每个模型的评估图 + 特征重要性图。

运行：python demo_breast_cancer.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ensemble_models import (
    train_and_evaluate, compare_models,
    plot_confusion_and_roc, plot_feature_importance,
)

# --------------------------------------------------------------------------
# 1. 数据准备
# --------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # 0=恶性, 1=良性
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"训练集 {X_train.shape[0]} | 测试集 {X_test.shape[0]}")

os.makedirs('figures', exist_ok=True)

# --------------------------------------------------------------------------
# 2. 依次训练 5 种模型（tune=True 开启 GridSearchCV 调参）
# --------------------------------------------------------------------------
results, models = {}, {}
for name in ['rf', 'gbdt', 'xgboost', 'lightgbm', 'catboost']:
    print(f"\n===== {name} =====")
    model, metrics = train_and_evaluate(
        name, X_train, y_train, X_test, y_test,
        tune=True,                      # 5 折交叉验证调参
        scoring='recall',
        positive_label=0,               # 恶性(0)为正类 → Recall = 恶性召回率
        n_jobs=1,                       # 沙箱/容器环境并行开销大，串行反而快
    )
    results[name], models[name] = metrics, model
    bp = metrics.pop('best_params')
    print(metrics)
    if bp:
        print(f"best_params: {bp}")

# --------------------------------------------------------------------------
# 3. 五模型对比表
# --------------------------------------------------------------------------
print("\n===== 模型对比（按 AUROC 排序）=====")
comparison = compare_models(results)
print(comparison.to_string(index=False))
comparison.to_csv('model_comparison.csv', index=False)

# --------------------------------------------------------------------------
# 4. 最优模型的评估图 + 特征重要性图
# --------------------------------------------------------------------------
best_name = comparison.iloc[0]['model']
best_model = models[best_name]
print(f"\n最优模型: {best_name}")

y_pred = np.asarray(best_model.predict(X_test)).ravel()
y_proba = best_model.predict_proba(X_test)[:, 1]

plot_confusion_and_roc(
    y_test, y_pred, y_proba, model_name=best_name,
    class_names=('Malignant', 'Benign'),
    save_path=f'figures/confusion_roc_{best_name}.png',
)

importance, std = plot_feature_importance(
    best_model, X_test, y_test, feature_names,
    top_k=10, scoring='recall',
    save_path=f'figures/feature_importance_{best_name}.png',
)

top5 = np.argsort(importance)[-5:][::-1]
print("\nTop 5 features (permutation importance, recall drop):")
for i, idx in enumerate(top5):
    print(f"  {i+1}. {feature_names[idx]:25s} {importance[idx]:.4f}")
