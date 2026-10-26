"""W-P1-W：真实架构上的 Δ_mech 验证（4 架构 × 2 臂 × 4 数据集 × 3 体制）

预注册：`W-P1-W真实架构验证预注册规格-v1.0.md`（判据先于训练写定）

臂定义（逐架构，来自源码核验，见预注册 §1.1）
------------------------------------------------
  DLinear    full: individual=False   indep: individual=True
  PatchTST   full: 全通道拼接序列上 patch（人造，跨通道可见）
             indep: 原版（patch 逐通道，天然 CI）
  iTransformer  full: 原版（enc_in=n_vars，通道注意力）
                indep: enc_in=1，逐通道训练后拼接
  TimeMixer  full: channel_independence=0   indep: channel_independence=1（原生开关）

统一训练协议（预注册 §2，不逐架构调参）
--------------------------------------
  seq_len=96 pred_len=96 d_model=64 d_ff=128 e_layers=2 n_heads=4
  dropout=0.1 batch=32 lr=1e-3 epochs=20 patience=5
  数据切分完全复用 tsf_loader.split / tsf_external.split_external

用法
----
  python scripts/run_p1w_arch.py --arch DLinear --seeds 0
  python scripts/run_p1w_arch.py --all --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIB = Path(r"C:/paper-QuantMo/论文模板2的代码-PIR-main")
sys.path.insert(0, str(LIB))

import _depshim                                              # noqa: E402
_depshim.install()

from tsf_loader import BENCHMARKS, split as split_inner     # noqa: E402

RESULTS = ROOT / "results"
RUNS = RESULTS / "p1w_runs.csv"
RESULTS.mkdir(exist_ok=True)

REGIMES = ("raw", "shift", "noise")
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")

SEQ_LEN, PRED_LEN = 96, 96
D_MODEL, D_FF, E_LAYERS, N_HEADS = 64, 128, 2, 4
DROPOUT, BATCH, LR = 0.1, 32, 1e-3
EPOCHS, PATIENCE = 20, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Cfg:
    """最小 configs 对象，字段名与 Time-series-Library 一致。"""
    def __init__(self, **kw):
        self.task_name = "long_term_forecast"
        self.seq_len = SEQ_LEN
        self.label_len = SEQ_LEN // 2
        self.pred_len = PRED_LEN
        self.d_model = D_MODEL
        self.d_ff = D_FF
        self.e_layers = E_LAYERS
        self.n_heads = N_HEADS
        self.dropout = DROPOUT
        self.factor = 1
        self.embed = "timeF"
        self.freq = "h"
        self.activation = "gelu"
        self.output_attention = False
        self.use_norm = 1
        self.moving_avg = 25
        self.down_sampling_layers = 1
        self.down_sampling_window = 2
        self.down_sampling_method = "avg"
        self.decomp_method = "moving_avg"
        self.top_k = 5
        self.channel_independence = 1
        self.enc_in = 1
        self.dec_in = 1
        self.c_out = 1
        self.num_class = 2
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------- 数据 ----------------

def build_dataset(name: str, regime: str, offset: int = 0):
    """返回 (X_train, Y_train, X_val, Y_val, X_test, Y_test)，已在 train 段标准化。

    完全复用 p32 的切分语义：train 800 / calib 400 / test 500 + lookback 前缀。
    calib 再对半切成 train/val（与 delta_res_real 的 honest 口径一致）。
    """
    s = split_inner(name, regime, lookback=SEQ_LEN, n_train=800,
                    n_calib=400, n_test=500, offset=offset)
    tr = s.train
    ca = s.calib
    te = s.test

    def windows(arr, n):
        X = np.stack([arr[i: i + SEQ_LEN] for i in range(n)])
        Y = np.stack([arr[i + SEQ_LEN: i + SEQ_LEN + PRED_LEN] for i in range(n)])
        return X, Y

    n_tr = tr.shape[0] - SEQ_LEN - PRED_LEN
    n_ca = ca.shape[0] - SEQ_LEN - PRED_LEN
    n_te = te.shape[0] - SEQ_LEN - PRED_LEN
    if min(n_tr, n_ca, n_te) <= 0:
        raise ValueError(f"{name}/{regime}: 序列太短 n_tr={n_tr} n_ca={n_ca} n_te={n_te}")

    Xt, Yt = windows(tr, n_tr)
    Xc, Yc = windows(ca, n_ca)
    Xe, Ye = windows(te, n_te)

    h = len(Xc) // 2
    return (Xt, Yt, Xc[:h], Yc[:h], Xe, Ye)


# ---------------- 架构包装 ----------------

class PatchTSTPosOnly(nn.Module):
    """PatchTST 的通道独立简化封装（对齐官方 patch 逻辑，去掉 x_mark）。

    官方 PatchTST.Model.forecast 需要 x_mark；这里用一个只吃 x 的等价实现，
    保证 indep 臂与 full 臂共享同一套 patch/attention 结构，唯一差别是
    patch 的输入维度（逐通道 vs 全通道拼接）。
    """

    def __init__(self, n_vars: int, cross_channel: bool):
        super().__init__()
        from layers.Transformer_EncDec import Encoder, EncoderLayer
        from layers.SelfAttention_Family import FullAttention, AttentionLayer
        from layers.Embed import PatchEmbedding

        self.n_vars = n_vars
        self.cross = cross_channel
        self.patch_len, self.stride = 16, 8
        padding = self.stride
        c_len = n_vars * SEQ_LEN if cross_channel else SEQ_LEN

        # cross=True：把 C 条通道首尾拼成 C*T 的长序列（1 个 patch 序列，输出 C 列）
        # cross=False：原版逐通道 patch（C 个 patch 序列，每个只输出 1 列）
        self.patch_in_len = c_len
        self.out_dim = n_vars if cross_channel else 1
        self.patch_embedding = PatchEmbedding(
            D_MODEL, self.patch_len, self.stride, padding, DROPOUT)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, 1, attention_dropout=DROPOUT,
                                             output_attention=False),
                               D_MODEL, N_HEADS),
                D_MODEL, D_FF, dropout=DROPOUT, activation="gelu")
             for _ in range(E_LAYERS)],
            norm_layer=torch.nn.BatchNorm1d(D_MODEL))
        self.patch_num = int((self.patch_in_len - self.patch_len) / self.stride + 2)
        self.head_nf = D_MODEL * self.patch_num
        self.head = nn.Linear(self.head_nf, PRED_LEN * self.out_dim)

    def forward(self, x):                      # x: (B, T, C)
        B, T, C = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        if self.cross:
            # 全通道在时间轴上首尾拼接成 (1, C*T)，patch 可跨通道边界
            x2 = x.permute(0, 2, 1).reshape(B, 1, -1)
        else:
            # 原版 PatchTST：通道维作为 n_vars，patch 逐通道
            x2 = x.permute(0, 2, 1)
        enc, nv = self.patch_embedding(x2)
        enc, _ = self.encoder(enc)
        enc = enc.reshape(-1, nv, enc.shape[-2], enc.shape[-1])
        enc = enc.permute(0, 1, 3, 2).reshape(B * nv, -1)
        out = self.head(enc)                                 # (B*nv, pred*out_dim)
        out = out.reshape(B, nv * self.out_dim, PRED_LEN).permute(0, 2, 1)
        out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return out


def build_model(arch: str, arm: str, n_vars: int):
    """返回 (model, 是否逐通道拼接)。"""
    if arch == "DLinear":
        from models.DLinear import Model as M
        cfg = Cfg(enc_in=n_vars, c_out=n_vars, seq_len=SEQ_LEN, pred_len=PRED_LEN)
        return M(cfg, individual=(arm == "indep")), False

    if arch == "iTransformer":
        from models.iTransformer import Model as M
        if arm == "full":
            cfg = Cfg(enc_in=n_vars, dec_in=n_vars, c_out=n_vars,
                      use_norm=1, factor=1)
            return M(cfg), False
        cfg = Cfg(enc_in=1, dec_in=1, c_out=1)
        return M(cfg), True                      # 逐通道

    if arch == "TimeMixer":
        from models.TimeMixer import Model as M
        # channel_independence=1 时模型内部自行按通道 reshape（TimeMixer.py:417），
        # 因此两臂都喂全通道输入，per_channel 一律 False。
        ci = 1 if arm == "indep" else 0
        cfg = Cfg(enc_in=n_vars, dec_in=n_vars, c_out=n_vars,
                  channel_independence=ci, use_norm=1)
        return M(cfg), False

    if arch == "PatchTST":
        return PatchTSTPosOnly(n_vars, cross_channel=(arm == "full")), False

    raise KeyError(arch)


def forward_batch(model, x, per_channel: bool):
    """统一前向：真实模型可能要求 x_mark，也可能不需要。"""
    if per_channel:
        B, T, C = x.shape
        outs = []
        for j in range(C):
            xj = x[:, :, j: j + 1]
            try:
                o = model(xj)
            except TypeError:
                o = model(xj, None, None, None)
            outs.append(o)
        return torch.cat(outs, dim=2)
    try:
        o = model(x)
    except TypeError:
        o = model(x, None, None, None)
    return o


# ---------------- 训练 ----------------

def train_eval(arch: str, arm: str, dataset: str, regime: str,
               seed: int, offset: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Ytr, Xva, Yva, Xte, Yte = build_dataset(dataset, regime, offset)
    n_vars = Xtr.shape[2]

    model, per_ch = build_model(arch, arm, n_vars)
    model = model.to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    Xtr_t, Ytr_t = to_t(Xtr), to_t(Ytr)
    Xva_t, Yva_t = to_t(Xva), to_t(Yva)
    Xte_t, Yte_t = to_t(Xte), to_t(Yte)

    n = len(Xtr_t)
    best_val, best_state, bad = np.inf, None, 0
    epochs_run = 0

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i: i + BATCH]
            opt.zero_grad()
            out = forward_batch(model, Xtr_t[idx], per_ch)
            loss = lossf(out, Ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            v = lossf(forward_batch(model, Xva_t, per_ch), Yva_t).item()
        epochs_run = ep + 1
        if v < best_val - 1e-6:
            best_val, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = forward_batch(model, Xte_t, per_ch)
        te_mse = lossf(pred, Yte_t).item()

    return dict(arch=arch, arm=arm, dataset=dataset, regime=regime, seed=seed,
                mse=te_mse, val_mse=best_val, n_param=n_param,
                epochs_run=epochs_run, n_vars=n_vars, offset=offset)


def append(row: dict):
    new = not RUNS.exists()
    with RUNS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def already_done(arch, arm, dataset, regime, seed, offset=0) -> bool:
    if not RUNS.exists():
        return False
    with RUNS.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r["arch"] == arch and r["arm"] == arm and r["dataset"] == dataset
                    and r["regime"] == regime and int(r["seed"]) == seed
                    and int(float(r["offset"])) == offset):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    args = ap.parse_args()

    archs = ["DLinear", "iTransformer", "TimeMixer", "PatchTST"] if args.all else [args.arch]
    if archs == [None]:
        print("需指定 --arch 或 --all")
        return 1
    seeds = [int(s) for s in args.seeds.split(",")]
    dsets = [d for d in args.datasets.split(",") if d]

    print(f"device={DEVICE}  archs={archs}  seeds={seeds}  datasets={dsets}", flush=True)
    t0 = time.time()
    for arch in archs:
        for ds in dsets:
            for reg in REGIMES:
                for arm in ("full", "indep"):
                    for sd in seeds:
                        tag = f"{arch}/{ds}/{reg}/{arm}/s{sd}"
                        if already_done(arch, arm, ds, reg, sd):
                            print(f"  [skip] {tag}", flush=True)
                            continue
                        try:
                            r = train_eval(arch, arm, ds, reg, sd)
                        except Exception as e:              # noqa: BLE001
                            print(f"  [FAIL] {tag}: {type(e).__name__}: {e}",
                                  flush=True)
                            continue
                        append(r)
                        print(f"  {tag:44s} mse={r['mse']:.5f} "
                              f"val={r['val_mse']:.5f} ep={r['epochs_run']:2d} "
                              f"P={r['n_param']}", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
