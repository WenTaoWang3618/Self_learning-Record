#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
蛋白二级结构预测 BiLSTM 教程（Q3 三态分类：H=α螺旋 / E=β折叠 / C=卷曲）
====================================================================
数据: CB513 数据集（Zhou & Troyanskaya, ICML 2014）
      Zenodo 官方镜像: https://zenodo.org/records/7764556
      每条蛋白 = (700 残基 × 57 特征) 矩阵，padding 到 700
      特征布局: [0,22) 氨基酸 one-hot | [22,31) Q8 标签 | [31,33) 末端 |
                [33,35) 溶剂可及度 | [35,57) 序列谱 profile

用法:
    python train_ss_bilstm.py                       # 完整训练（CPU 约 5 分钟）
    python train_ss_bilstm.py --epochs 3            # 快速试跑
    python train_ss_bilstm.py --max-proteins 100    # 小规模冒烟测试
    python train_ss_bilstm.py --data-dir data --outdir figures

流程五步: 编码 → 变长处理 → 模型 → 训练循环 → 评估
"""
import argparse
import hashlib
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# 0. 数据下载（已下载则跳过）
# ----------------------------------------------------------------------------
URL = "https://zenodo.org/records/7764556/files/cb513%2Bprofile_split1.npy.gz?download=1"
MD5 = "a6ca35acd2a453b0186e3fe4af23011f"


def download_data(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cb513+profile_split1.npy")
    if os.path.exists(path):
        return path
    print("下载数据（约 164 MB，视网速 5-10 分钟）...")
    tmp = path + ".tmp"
    urllib.request.urlretrieve(URL, tmp)
    md5 = hashlib.md5(open(tmp, "rb").read()).hexdigest()
    assert md5 == MD5, f"md5 校验失败: {md5}"
    os.rename(tmp, path)  # 注意: 文件名为 .gz 但实为未压缩 .npy
    print("下载完成:", path)
    return path


# ----------------------------------------------------------------------------
# 1. 解析: 原始矩阵 → 特征 X / 标签 Y / 长度
# ----------------------------------------------------------------------------
Q8_TO_Q3 = np.array([2, 1, 1, 0, 0, 0, 2, 2, -100])  # L,B,E,G,I,H,S,T,NoSeq
#                                                      Q3: 0=H, 1=E, 2=C, -100=忽略


def load_dataset(npy_path: str):
    data = np.load(npy_path).reshape(-1, 700, 57).astype(np.float32)

    aa = data[:, :, 0:22]      # 氨基酸 one-hot（第 21 列 = NoSeq 填充标记）
    q8 = data[:, :, 22:31]     # Q8 标签（第 8 列 = NoSeq）
    prof = data[:, :, 35:57]   # 序列谱

    is_real = aa[:, :, 21] == 0
    lengths = is_real.sum(axis=1).astype(np.int64)

    X = np.concatenate([aa[:, :, :21], prof], axis=-1)          # (N, 700, 43)
    Y = Q8_TO_Q3[q8.argmax(axis=-1)].astype(np.int64)           # (N, 700)

    # 数据检查
    assert (aa[is_real][:, :21].sum(axis=1) == 1).all(), "one-hot 校验失败"
    assert (Y[~is_real] == -100).all() and (Y[is_real] >= 0).all(), "mask 校验失败"
    print(f"蛋白条数: {len(X)} | 平均序列长度: {lengths.mean():.1f} | "
          f"总残基数: {(Y != -100).sum()}")
    return X, Y, lengths


# ----------------------------------------------------------------------------
# 2. 数据管线: 划分 + Dataset/DataLoader + 变长处理
# ----------------------------------------------------------------------------
class ProteinSS(Dataset):
    """逐残基分类数据集: 返回单条蛋白的 (特征, 标签)，只取真实残基部分"""

    def __init__(self, X, Y, lengths, idx):
        self.X, self.Y, self.L, self.idx = X, Y, lengths, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j, L = self.idx[i], self.L[self.idx[i]]
        return (torch.from_numpy(np.ascontiguousarray(self.X[j, :L])),   # (L, 43)
                torch.from_numpy(np.ascontiguousarray(self.Y[j, :L])))   # (L,)


def collate(batch):
    """不同长度的蛋白拼成一个 batch: 特征补 0，标签补 -100（损失中忽略）"""
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    X = pad_sequence(xs, batch_first=True)                       # (B, Lmax, 43)
    Y = pad_sequence(ys, batch_first=True, padding_value=-100)   # (B, Lmax)
    return X, Y, lengths


def make_loaders(X, Y, lengths, batch_size=32, seed=42):
    """按比例划分 train/val/test；全量 514 条时为 380/50/84（与教程 notebook 一致）"""
    N = len(X)
    n_train = max(1, round(N * 380 / 514))
    n_val = max(1, round(N * 50 / 514))
    perm = np.random.default_rng(seed).permutation(N)
    splits = {"train": perm[:n_train],
              "val": perm[n_train:n_train + n_val],
              "test": perm[n_train + n_val:]}
    loaders = {k: DataLoader(ProteinSS(X, Y, lengths, idx), batch_size=batch_size,
                             shuffle=(k == "train"), collate_fn=collate)
               for k, idx in splits.items()}
    return loaders


# ----------------------------------------------------------------------------
# 3. 模型: 2 层双向 LSTM + 全连接头（多对多，逐残基分类）
# ----------------------------------------------------------------------------
class BiLSTMSS(nn.Module):
    def __init__(self, input_size=43, hidden_size=64, num_layers=2,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)  # 双向 → 2*hidden

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)                 # RNN 不处理 padding 位置
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.fc(self.dropout(out))          # (B, L, 3)


# ----------------------------------------------------------------------------
# 4. 训练循环: BPTT + 梯度裁剪 + 按验证集选最优模型
# ----------------------------------------------------------------------------
def run_epoch(model, loader, crit, opt=None):
    model.train() if opt else model.eval()
    tot_loss, tot_res, correct = 0.0, 0, 0
    with torch.set_grad_enabled(opt is not None):
        for X, Yb, L in loader:
            logits = model(X, L)
            loss = crit(logits.reshape(-1, 3), Yb.reshape(-1))
            if opt:
                opt.zero_grad()
                loss.backward()                              # BPTT
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 防爆炸
                opt.step()
            m = Yb != -100
            tot_loss += loss.item() * m.sum().item()
            correct += (logits.argmax(-1)[m] == Yb[m]).sum().item()
            tot_res += m.sum().item()
    return tot_loss / tot_res, correct / tot_res


def train(model, loaders, epochs=15, lr=1e-3):
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = 0.0, None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, _ = run_epoch(model, loaders["train"], crit, opt)
        va_loss, va_acc = run_epoch(model, loaders["val"], crit)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        if va_acc > best_acc:  # 早停思想: 保留验证集最优模型
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print("epoch %2d | train loss %.4f | val loss %.4f | val Q3 acc %.3f"
              % (epoch, tr_loss, va_loss, va_acc))

    model.load_state_dict(best_state)
    print("训练完成，用时 %.1f 分钟 | 最佳验证 Q3 准确率 %.3f"
          % ((time.time() - t0) / 60, best_acc))
    return history


# ----------------------------------------------------------------------------
# 5. 评估: 总体 Q3 准确率 + 每类 P/R/F1 + 混淆矩阵（手工计算，透明可查）
# ----------------------------------------------------------------------------
def evaluate(model, loader, outdir):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, Yb, L in loader:
            m = Yb != -100
            preds.append(model(X, L).argmax(-1)[m].numpy())
            trues.append(Yb[m].numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)

    names = ["H (helix)", "E (strand)", "C (coil)"]
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1

    print("测试集: %d 个残基 | 总体 Q3 准确率: %.3f" % (cm.sum(), np.trace(cm) / cm.sum()))
    print("\n类\t精确率\t召回率\tF1")
    for k in range(3):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print("%s\t%.3f\t%.3f\t%.3f" % (names[k], prec, rec, f1))

    try:
        plot_confusion(cm, names, outdir)
    except Exception as e:
        print("（绘图跳过: %s）" % e)
    return cm


def plot_confusion(cm, names, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%d\n(%.0f%%)" % (cm[i, j], 100 * norm[i, j]),
                    ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (residues)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{outdir}/confusion_matrix.svg")
    fig.savefig(f"{outdir}/confusion_matrix.png", dpi=200)
    print("混淆矩阵已保存到", outdir)


def plot_history(history, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    matplotlib.rcParams["svg.fonttype"] = "none"

    os.makedirs(outdir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(epochs, history["train_loss"], "o-", color="#0279EE", label="Train")
    axes[0].plot(epochs, history["val_loss"], "s-", color="#FF9400", label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, history["val_acc"], "s-", color="#75A025")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Q3 accuracy")
    axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/training_curve.svg")
    fig.savefig(f"{outdir}/training_curve.png", dpi=200)
    print("训练曲线已保存到", outdir)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="蛋白二级结构预测 BiLSTM 教程")
    ap.add_argument("--data-dir", default="data", help="数据目录（自动下载）")
    ap.add_argument("--outdir", default="figures", help="图表输出目录")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-proteins", type=int, default=0, help=">0 时只用前 N 条（冒烟测试）")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    npy = download_data(args.data_dir)
    X, Y, lengths = load_dataset(npy)
    if args.max_proteins > 0:
        X, Y, lengths = X[:args.max_proteins], Y[:args.max_proteins], lengths[:args.max_proteins]

    loaders = make_loaders(X, Y, lengths, batch_size=args.batch_size)
    model = BiLSTMSS()
    print("可训练参数量: %d\n" % sum(p.numel() for p in model.parameters()))

    history = train(model, loaders, epochs=args.epochs)
    evaluate(model, loaders["test"], args.outdir)
    plot_history(history, args.outdir)


if __name__ == "__main__":
    main()
