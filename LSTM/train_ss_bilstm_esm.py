#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 ESM2 embedding 替换序列谱的蛋白二级结构预测（Q3 三态分类）
================================================================
与 train_ss_bilstm.py（profile 基线，Q3 74.5%）唯一的区别是输入特征：
    基线:  21 维氨基酸 one-hot + 22 维序列谱 = 43 维
    本版:  ESM2 最后一层逐残基 embedding（t12_35M 为 480 维）
其余（数据划分、模型结构、训练协议）完全一致，保证可比。

用法:
    python train_ss_bilstm_esm.py                          # 完整流程（CPU 约 20-30 分钟）
    python train_ss_bilstm_esm.py --esm-model esm2_t6_8M_UR50D   # 更快的小模型
    python train_ss_bilstm_esm.py --max-proteins 20 --epochs 2   # 冒烟测试
    python train_ss_bilstm_esm.py --extract-only           # 只提取 embedding（缓存后复用）
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
# 数据与模型规格
# ----------------------------------------------------------------------------
CB513_URL = "https://zenodo.org/records/7764556/files/cb513%2Bprofile_split1.npy.gz?download=1"
CB513_MD5 = "a6ca35acd2a453b0186e3fe4af23011f"

# ESM2 规格名 → 逐残基 embedding 维度（fair-esm 预训练函数同名）
ESM_MODELS = {
    "esm2_t6_8M_UR50D": 320,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t33_650M_UR50D": 1280,
}

# CB513 one-hot 的氨基酸顺序（官方 readme）: 20 标准残基 + X（未知）+ NoSeq（填充）
AA_ORDER = "ACEDGFIHKMLNQPSRTWVYX"
Q8_TO_Q3 = np.array([2, 1, 1, 0, 0, 0, 2, 2, -100])  # L,B,E,G,I,H,S,T,NoSeq → 0=H,1=E,2=C


def download_data(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cb513+profile_split1.npy")
    if os.path.exists(path):
        return path
    print("下载数据（约 164 MB）...")
    tmp = path + ".tmp"
    urllib.request.urlretrieve(CB513_URL, tmp)
    md5 = hashlib.md5(open(tmp, "rb").read()).hexdigest()
    assert md5 == CB513_MD5, f"md5 校验失败: {md5}"
    os.rename(tmp, path)  # 文件名为 .gz 但实为未压缩 .npy
    return path


def load_cb513(npy_path: str):
    """返回: 序列列表, 逐残基标签 (N,700), 真实长度 (N,)"""
    data = np.load(npy_path).reshape(-1, 700, 57).astype(np.float32)
    aa = data[:, :, 0:22]
    q8 = data[:, :, 22:31]

    is_real = aa[:, :, 21] == 0
    lengths = is_real.sum(axis=1).astype(np.int64)

    # one-hot → 氨基酸字符串（含 X = 未知残基）
    seqs = []
    for i in range(len(data)):
        idx = aa[i].argmax(axis=-1)
        seq = "".join(AA_ORDER[k] for k in idx[: lengths[i]])
        seqs.append(seq)

    Y = Q8_TO_Q3[q8.argmax(axis=-1)].astype(np.int64)
    Y[~is_real] = -100
    return seqs, Y, lengths


# ----------------------------------------------------------------------------
# ESM2 逐残基 embedding 提取（带磁盘缓存）
# ----------------------------------------------------------------------------
def extract_embeddings(seqs, model_name: str, cache_path: str):
    """返回 (N, 700, D) float16 的 embedding 矩阵（padding 位置为 0）"""
    my_lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if np.array_equal(cached["lengths"], my_lengths):  # 防止误用不同规模数据的缓存
            print("使用缓存的 embedding:", cache_path)
            return cached["emb"], cached["lengths"]
        print("缓存与当前数据不匹配，重新提取...")
        os.remove(cache_path)

    import esm  # fair-esm

    assert model_name in ESM_MODELS, f"未知模型 {model_name}，可选: {list(ESM_MODELS)}"
    print(f"加载 {model_name}（首次运行会自动下载权重）...")
    model, alphabet = getattr(esm.pretrained, model_name)()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    layer = model.num_layers

    # 按长度排序分批，减少 padding 浪费；结束后恢复原顺序
    order = np.argsort([-len(s) for s in seqs])
    emb_padded = np.zeros((len(seqs), 700, ESM_MODELS[model_name]), dtype=np.float16)

    t0, batch_size = time.time(), 8
    for start in range(0, len(seqs), batch_size):
        idx = order[start:start + batch_size]
        data = [(f"p{j}", seqs[j]) for j in idx]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer])
        rep = out["representations"][layer]          # (B, Lmax+2, D)，含 BOS/EOS
        for row, j in enumerate(idx):
            L = len(seqs[j])
            # 第 0 位是 BOS，第 1..L 位才是逐残基表示
            emb_padded[j, :L] = rep[row, 1:L + 1].to(torch.float16).numpy()
        done = start + len(idx)
        print("  embedding %d/%d（用时 %.1f 分钟）" % (done, len(seqs), (time.time() - t0) / 60))

    lengths = my_lengths
    np.savez_compressed(cache_path, emb=emb_padded, lengths=lengths)
    print("embedding 已缓存:", cache_path)
    return emb_padded, lengths


# ----------------------------------------------------------------------------
# 数据管线 / 模型 / 训练 / 评估（与 profile 基线完全一致，仅输入维度不同）
# ----------------------------------------------------------------------------
class ProteinSS(Dataset):
    def __init__(self, X, Y, lengths, idx):
        self.X, self.Y, self.L, self.idx = X, Y, lengths, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j, L = self.idx[i], self.L[self.idx[i]]
        return (torch.from_numpy(np.ascontiguousarray(self.X[j, :L].astype(np.float32))),
                torch.from_numpy(np.ascontiguousarray(self.Y[j, :L])))


def collate(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    X = pad_sequence(xs, batch_first=True)                       # (B, Lmax, D)
    Y = pad_sequence(ys, batch_first=True, padding_value=-100)   # (B, Lmax)
    return X, Y, lengths


def make_loaders(X, Y, lengths, batch_size=32, seed=42):
    N = len(X)
    n_train = max(1, round(N * 380 / 514))
    n_val = max(1, round(N * 50 / 514))
    perm = np.random.default_rng(seed).permutation(N)
    splits = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val],
              "test": perm[n_train + n_val:]}
    return {k: DataLoader(ProteinSS(X, Y, lengths, idx), batch_size=batch_size,
                          shuffle=(k == "train"), collate_fn=collate)
            for k, idx in splits.items()}


class BiLSTMSS(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.fc(self.dropout(out))


def run_epoch(model, loader, crit, opt=None):
    model.train() if opt else model.eval()
    tot_loss, tot_res, correct = 0.0, 0, 0
    with torch.set_grad_enabled(opt is not None):
        for X, Yb, L in loader:
            logits = model(X, L)
            loss = crit(logits.reshape(-1, 3), Yb.reshape(-1))
            if opt:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print("epoch %2d | train loss %.4f | val loss %.4f | val Q3 acc %.3f"
              % (epoch, tr_loss, va_loss, va_acc), flush=True)

    model.load_state_dict(best_state)
    print("训练完成，用时 %.1f 分钟 | 最佳验证 Q3 准确率 %.3f" % ((time.time() - t0) / 60, best_acc))
    return history


def evaluate(model, loader, outdir, tag="esm"):
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

    q3 = np.trace(cm) / cm.sum()
    print("测试集: %d 个残基 | 总体 Q3 准确率: %.3f" % (cm.sum(), q3))
    print("\n类\t精确率\t召回率\tF1")
    for k in range(3):
        tp = cm[k, k]
        prec = tp / cm[:, k].sum() if cm[:, k].sum() else 0.0
        rec = tp / cm[k, :].sum() if cm[k, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print("%s\t%.3f\t%.3f\t%.3f" % (names[k], prec, rec, f1))

    try:
        plot_confusion(cm, names, outdir, tag)
        return q3
    except Exception as e:
        print("（绘图跳过: %s）" % e)
        return q3


def plot_confusion(cm, names, outdir, tag):
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
    fig.savefig(f"{outdir}/{tag}_confusion_matrix.svg")
    fig.savefig(f"{outdir}/{tag}_confusion_matrix.png", dpi=200)
    print("混淆矩阵已保存到", outdir)


def plot_history(history, outdir, tag="esm"):
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
    fig.savefig(f"{outdir}/{tag}_training_curve.svg")
    fig.savefig(f"{outdir}/{tag}_training_curve.png", dpi=200)
    print("训练曲线已保存到", outdir)


def main():
    ap = argparse.ArgumentParser(description="ESM2 embedding + BiLSTM 二级结构预测")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--esm-model", default="esm2_t12_35M_UR50D", choices=list(ESM_MODELS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-proteins", type=int, default=0, help=">0 时只用前 N 条（冒烟测试）")
    ap.add_argument("--extract-only", action="store_true", help="只提取 embedding 不训练")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    npy = download_data(args.data_dir)
    seqs, Y, lengths = load_cb513(npy)
    if args.max_proteins > 0:
        seqs, Y, lengths = seqs[:args.max_proteins], Y[:args.max_proteins], lengths[:args.max_proteins]

    cache = os.path.join(args.data_dir,
                         f"{args.esm_model}_embeddings_N{len(seqs)}.npz")  # 缓存按数据规模区分
    emb, lengths = extract_embeddings(seqs, args.esm_model, cache)
    if args.extract_only:
        return

    loaders = make_loaders(emb, Y, lengths, batch_size=args.batch_size)
    model = BiLSTMSS(input_size=ESM_MODELS[args.esm_model])
    print("可训练参数量: %d\n" % sum(p.numel() for p in model.parameters()))

    history = train(model, loaders, epochs=args.epochs)
    q3 = evaluate(model, loaders["test"], args.outdir)
    plot_history(history, args.outdir)

    print("\n===== 对比 =====")
    print("profile 基线 (train_ss_bilstm.py):  测试 Q3 = 0.745")
    print("ESM2 (%s): 测试 Q3 = %.3f" % (args.esm_model, q3))


if __name__ == "__main__":
    main()
