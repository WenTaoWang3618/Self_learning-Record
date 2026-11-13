"""
ensemble_models.py
==================
可复用的集成学习工具包：随机森林（Bagging）+ GBDT 家族（Boosting）

统一接口支持 5 种模型：
    rf        : RandomForestClassifier       随机森林（Bagging，并行 + 投票）
    gbdt      : GradientBoostingClassifier   GBDT 原型（Boosting，串行拟合残差）
    xgboost   : XGBClassifier                二阶泰勒 + 正则化
    lightgbm  : LGBMClassifier               直方图加速，大数据首选
    catboost  : CatBoostClassifier           原生类别特征，默认参数强

快速上手：
    from ensemble_models import get_model, train_and_evaluate

    model, metrics = train_and_evaluate(
        'xgboost', X_train, y_train, X_test, y_test,
        positive_label=0,          # 癌症数据中 0=恶性，作为"正类"算 Recall
    )
    print(metrics)

    # 调参模式（GridSearchCV，5 折交叉验证）：
    model, metrics = train_and_evaluate(
        'lightgbm', X_train, y_train, X_test, y_test, tune=True,
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# --------------------------------------------------------------------------
# 1. 模型工厂：统一创建入口
# --------------------------------------------------------------------------

def get_model(name, random_state=42, **kwargs):
    """按名称创建模型。

    Parameters
    ----------
    name : str
        'rf' | 'gbdt' | 'xgboost' | 'lightgbm' | 'catboost'
    random_state : int
        随机种子，保证可复现。
    **kwargs :
        透传给底层模型，如 n_estimators=200, max_depth=5。
    """
    base = {
        'rf':       RandomForestClassifier,
        'gbdt':     GradientBoostingClassifier,
        'xgboost':  XGBClassifier,
        'lightgbm': LGBMClassifier,
        'catboost': CatBoostClassifier,
    }
    if name not in base:
        raise ValueError(f"未知模型 '{name}'，可选: {list(base)}")
    cls = base[name]
    if cls is None:
        raise ImportError(f"模型 '{name}' 的依赖库未安装，请先 pip install")

    # 各库的静默/兼容参数
    if name == 'catboost':
        kwargs.setdefault('verbose', 0)
        # 禁止 CatBoost 写 catboost_info 临时目录（网络盘/只读目录下会报错）
        kwargs.setdefault('allow_writing_files', False)
    if name == 'lightgbm':
        kwargs.setdefault('verbose', -1)   # 关闭训练日志
    if name == 'xgboost':
        kwargs.setdefault('eval_metric', 'logloss')
    kwargs.setdefault('random_state', random_state)
    return cls(**kwargs)


# --------------------------------------------------------------------------
# 2. 默认调参网格（每个算法的常用搜索空间）
# --------------------------------------------------------------------------

DEFAULT_PARAM_GRIDS = {
    'rf': {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt', 'log2'],
    },
    'gbdt': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [2, 3, 5],
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
    },
    'lightgbm': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [15, 31, 63],
    },
    'catboost': {
        'iterations': [100, 200],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6, 8],
    },
}


# --------------------------------------------------------------------------
# 3. 训练 + 评估：一条命令完成
# --------------------------------------------------------------------------

def train_and_evaluate(name, X_train, y_train, X_test, y_test,
                       tune=False, cv=5, scoring='recall',
                       positive_label=1, random_state=42,
                       n_jobs=-1, **model_kwargs):
    """训练模型并在测试集上评估，返回 (model, metrics_dict)。

    Parameters
    ----------
    name : str
        模型名称，见 get_model。
    tune : bool
        True 时用 GridSearchCV（cv 折交叉验证）在训练集上调参。
    scoring : str
        调参时的选优标准，默认 'recall'（癌症场景重点看漏诊率）。
    positive_label : int or str
        计算 Recall/Precision/F1 时的"正类"。
        癌症数据 load_breast_cancer 中 0=恶性，想算"恶性召回率"就传 0。
    n_jobs : int
        GridSearchCV 的并行度。-1 用全部核心；受限环境（容器/沙箱）
        建议设 1，并行开销可能远大于收益。
    """
    if tune:
        grid = dict(DEFAULT_PARAM_GRIDS[name])
        search = GridSearchCV(
            get_model(name, random_state=random_state, **model_kwargs),
            param_grid=grid, cv=cv, scoring=scoring, n_jobs=n_jobs,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = get_model(name, random_state=random_state, **model_kwargs)
        model.fit(X_train, y_train)
        best_params = None

    y_pred = model.predict(X_test)
    # XGBoost/LightGBM/CatBoost 的 predict 已返回原始标签；统一转 numpy
    y_pred = np.asarray(y_pred).ravel()
    y_proba = _predict_proba(model, X_test)

    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, pos_label=positive_label),
        'precision': precision_score(y_test, y_pred, pos_label=positive_label),
        'f1': f1_score(y_test, y_pred, pos_label=positive_label),
        'auroc': roc_auc_score(y_test, y_proba),
        'best_params': best_params,
    }
    # 随机森林额外报告 OOB（需要 oob_score=True）
    if name == 'rf' and getattr(model, 'oob_score', False):
        metrics['oob_score'] = model.oob_score_
    return model, metrics


def _predict_proba(model, X):
    """统一取正类概率。"""
    proba = model.predict_proba(X)
    return proba[:, 1]


# --------------------------------------------------------------------------
# 4. 可视化：混淆矩阵 + ROC 曲线
# --------------------------------------------------------------------------

def plot_confusion_and_roc(y_test, y_pred, y_proba, model_name='model',
                           class_names=('Negative', 'Positive'),
                           save_path=None):
    """画混淆矩阵 + ROC 曲线并排图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title(f'Confusion Matrix ({model_name})')

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auroc = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color='#0279EE', lw=2, label=f'AUROC = {auroc:.3f}')
    axes[1].plot([0, 1], [0, 1], color='gray', ls='--', label='Random Guess')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve ({model_name})')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# --------------------------------------------------------------------------
# 5. 可视化：特征重要性（排列重要性，统一支持全部 5 种模型）
# --------------------------------------------------------------------------

def plot_feature_importance(model, X_test, y_test, feature_names,
                            top_k=10, scoring='recall',
                            save_path=None):
    """排列重要性 Top-K 水平条形图（含误差线）。

    用排列重要性而非内置 feature_importances_，因为前者更可靠，
    且对 5 种模型接口完全统一。
    """
    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=10, random_state=42,
                                  scoring=scoring)
    importance = perm.importances_mean
    std = perm.importances_std

    order = np.argsort(importance)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_k), importance[order], color='#0279EE',
            alpha=0.85, xerr=std[order], capsize=3)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(np.asarray(feature_names)[order], fontsize=10)
    ax.set_xlabel(f'Permutation Importance ({scoring} drop)')
    ax.set_title(f'Top {top_k} Features')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return importance, std


# --------------------------------------------------------------------------
# 6. 多模型对比表
# --------------------------------------------------------------------------

def compare_models(results):
    """把 train_and_evaluate 返回的多个 metrics 汇总成对比表。

    用法：
        results = {}
        for name in ['rf', 'gbdt', 'xgboost', 'lightgbm', 'catboost']:
            _, results[name] = train_and_evaluate(name, ...)
        df = compare_models(results)
    """
    import pandas as pd
    rows = []
    for name, m in results.items():
        rows.append({
            'model': name,
            'accuracy': round(m['accuracy'], 4),
            'recall': round(m['recall'], 4),
            'precision': round(m['precision'], 4),
            'f1': round(m['f1'], 4),
            'auroc': round(m['auroc'], 4),
        })
    return pd.DataFrame(rows).sort_values('auroc', ascending=False)
