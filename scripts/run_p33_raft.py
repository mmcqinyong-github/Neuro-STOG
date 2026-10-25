"""W-P3-3 层 2 复现驱动脚本（RAFT 官方代码，零改写）。

设计原则（对应 W-P3-3层2复现预注册规格-v1.0.md）：
  * 官方源码 `C:/tmp/raft` **一行不改**，用 monkeypatch 注入消融开关；
  * 依赖缺失（sktime / matplotlib）用 shim 目录 `C:/tmp/raft_shim` 注入；
  * 档位通过全局开关 `RAFT_ARM` 控制：
      - "on"  : 完整 RAFT（官方行为）
      - "A"   : 结构不变，前向期把 retrieval_pred_list 置零（检索分支净贡献）
      - "B"   : 只保留线性预测器 linear_x（等价论文 Table 8 的 "Without Retrieval"）
  * 每个 (dataset, arm, seed) 一次独立进程，避免显存/状态串扰。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

RAFT_DIR = Path(r"C:\tmp\raft")
SHIM_DIR = Path(r"C:\tmp\raft_shim")

T0 = time.time()


def _mark(label: str) -> None:
    """时间埋点：用于定位各阶段耗时（写入 stdout，不影响官方输出格式）"""
    print(f"[TIMING] {label}: {time.time() - T0:.1f}s", flush=True)

sys.path.insert(0, str(RAFT_DIR))
sys.path.insert(0, str(SHIM_DIR))

import torch  # noqa: E402

ARM = {"value": "on"}


# --------------------------------------------------------------------------
# 消融注入：不触碰官方源码，只在运行时替换 Model.encoder
# --------------------------------------------------------------------------
def patch_arm(arm: str) -> None:
    """按档位替换 RAFT.Model.encoder。

    官方前向（models/RAFT.py）：
        x_pred_from_x   = linear_x(x_norm)                      # B,P,C
        retrieval_list  = sum_g retrieval_pred[g](...)           # B,P,C
        pred            = linear_pred(cat([x_pred_from_x, retrieval_list])) + x_offset

    档 A：保留全部结构与参数量，仅把 retrieval_list 前向期置零。
           因为置零发生在前向期，linear_pred 必须重新训练才能适配，
           否则测到的是"训练后剪枝"的虚假贡献（预注册 §6 陷阱 1）。
    档 B：只用线性预测器，等价论文 Table 8 的 "Without Retrieval"。
    """
    from models import RAFT as RAFT_mod

    original_encoder = RAFT_mod.Model.encoder

    def encoder(self, x, index, mode):
        if ARM["value"] == "on":
            return original_encoder(self, x, index, mode)

        index = index.to(self.device)
        bsz, seq_len, channels = x.shape
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        if ARM["value"] == "B":
            # 只保留线性预测器
            return x_pred_from_x + x_offset

        # 档 A：检索分支前向期真置零
        zero = torch.zeros_like(x_pred_from_x)
        pred = torch.cat([x_pred_from_x, zero], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        pred = pred.reshape(bsz, self.pred_len, self.channels)
        return pred + x_offset

    RAFT_mod.Model.encoder = encoder


def patch_device_placement() -> None:
    """修复官方代码在新版 PyTorch 下的设备放置不兼容。

    官方 `prepare_dataset` 把检索库存在 CPU（`retrievals.append(...cpu())`），
    而 `encoder` 用 `.to(self.device)` 后的 index 去索引它，新版 PyTorch 直接报错：
        RuntimeError: indices should be either on cpu or on the same device
    官方作者当时的 torch 版本允许 CPU 张量被 GPU index 索引。

    本补丁**只改设备位置，不改任何数值逻辑**：把检索库搬到模型所在设备，
    使索引在同设备上进行，结果与官方语义完全一致（官方紧接着本来就有
    `pred_from_retrieval = pred_from_retrieval.to(self.device)`）。
    """
    from models import RAFT as RAFT_mod

    original_prepare = RAFT_mod.Model.prepare_dataset

    def prepare_dataset(self, train_data, valid_data, test_data):
        original_prepare(self, train_data, valid_data, test_data)
        # 检索库搬到 GPU，消除设备不匹配
        self.retrieval_dict = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in self.retrieval_dict.items()
        }

    RAFT_mod.Model.prepare_dataset = prepare_dataset


def patch_retrieval_cache(cache_key: str, cache_dir: Path) -> None:
    """检索库磁盘缓存（纯性能优化，数值必须逐位一致）。

    官方 `RetrievalTool.retrieve_all` 用 `shuffle=False`、无随机性，
    且检索库只取决于 (数据集, seq_len, pred_len, topm, n_period)，
    **与消融档位、随机种子无关**。但官方每个进程都重新构建一遍，
    而构建是全流程最慢的一步（约占 4/6）。

    本补丁在 `Model.prepare_dataset` 外层加缓存：
      * 命中 ⇒ 直接载入已算好的 retrieval_dict；
      * 未命中 ⇒ 走官方逻辑，算完后落盘。

    **不变性保证**：缓存的是官方 `prepare_dataset` 的输出张量本身，
    不做任何数值变换；命中与未命中走的是同一份数据、同一个官方函数。
    验证方式：同一配置跑一次冷启动与一次热启动，比对 MSE 是否逐位相同。
    """
    from models import RAFT as RAFT_mod

    original_prepare = RAFT_mod.Model.prepare_dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"retrieval_{cache_key}.pt"

    def prepare_dataset(self, train_data, valid_data, test_data):
        if cache_file.exists():
            blob = torch.load(cache_file, map_location="cpu", weights_only=False)
            self.retrieval_dict = {k: v for k, v in blob.items()}
            # 缓存路径绕过了 patch_device_placement，必须同样把检索库搬到 GPU
            self.retrieval_dict = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in self.retrieval_dict.items()
            }
            return
        original_prepare(self, train_data, valid_data, test_data)
        torch.save({k: v for k, v in self.retrieval_dict.items()}, cache_file)

    RAFT_mod.Model.prepare_dataset = prepare_dataset


def patch_vali_mode() -> None:
    """修正官方 `exp.vali()` 中 `mode='valid'` 的硬编码（上游 bug）。

    官方 `exp/exp_long_term_forecasting.py` 的 `vali()` 里写死：

        outputs = self.model(batch_x, index, mode='valid')

    而 `exp.train()` 用**同一个** `vali()` 分别评估 val 段与 test 段：

        vali_loss = self.vali(vali_data, vali_loader, criterion)
        test_loss = self.vali(test_data, test_loader, criterion)   # ← 仍用 mode='valid'

    `mode` 决定从哪个检索库取数：`self.retrieval_dict[mode][:, index]`。
    检索库的第二个维度是**该段自己的样本数**：
      * ETT 系列：官方固定 12/4/4 月切分 ⇒ |val| == |test| ⇒ 用 'valid' 库
        去索引 test 的 index **恰好不越界**（数值上取的是 valid 段检索结果，
        属于官方既有行为，本复现如实沿用，不做"修正"）。
      * custom 系列（weather / exchange_rate）：官方 7:1:2 切分 ⇒ |val| ≠ |test|。
        weather 的 |val|=5175、|test|=10444 ⇒ 索引 10443 行进入只有 5175 行的
        'valid' 库 ⇒ `CUDA error: device-side assert triggered`（异步报错，
        栈顶落在无关的 `.to(self.device)` 上）。

    本补丁把 `mode` 由"写死 valid"改为"**按被评估的那一段取对应的库**"：
      * 评估 val 段 ⇒ mode='valid'（与官方完全一致）
      * 评估 test 段 ⇒ mode='test'（与官方 `exp.test()` 内部的写法一致）
    因此：
      * 对 ETT 系列**数值完全不变**（|val|==|test| 时两库同长，官方原行为保留）；
      * 对 custom 系列只是让 test 段用上本该属于它的检索库，使官方代码
        在该数据集上**能够跑完**，从而与论文 Table 8 的 weather 数字可比。

    这是对上游 bug 的最小修复，**不修改官方源码一行**，
    与本工作区其余注入（device placement / arm / cache）性质相同。
    """
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    original_vali = Exp_Long_Term_Forecast.vali

    def vali(self, vali_data, vali_loader, criterion):
        # 通过被评估段的样本数判断当前是 val 段还是 test 段：
        # 官方 train() 依次评估 val 段与 test 段，两者用同一个 vali()。
        # 检索库形状 (G, N_seg, P, C)，N_seg 就是该段的样本数，
        # 与 len(该段 dataset) 相同（检索库由 retrieve_all(该段 dataset) 构成）。
        mode = "valid"
        try:
            n = len(vali_data)
            seg_sizes = {k: int(v.shape[1]) for k, v in self.model.retrieval_dict.items()}
            # 若该段长度不等于 valid 段，则按 test 段处理
            if n != seg_sizes.get("valid", -1) and n == seg_sizes.get("test", -2):
                mode = "test"
        except Exception:
            mode = "valid"
        # 属性必须设在 Model 实例上：encoder 的 self 是 Model，不是 Exp
        self.model._p33_vali_mode = mode
        try:
            return original_vali(self, vali_data, vali_loader, criterion)
        finally:
            self.model._p33_vali_mode = None

    Exp_Long_Term_Forecast.vali = vali


def patch_vali_mode_forward() -> None:
    """把 `vali()` 选出的 mode 传到 `Model.encoder`。

    官方 `vali()` 内部写死 `mode='valid'`，我们不能改它；
    但 `Model.encoder` 是我们已经在 patch 的入口（消融注入就替换了它），
    在那里按 `_p33_vali_mode` 覆盖 mode 参数即可，官方源码仍然零改写。

    调用顺序：`patch_arm` 先替换 encoder（保存官方原版为 original_encoder），
    本函数再包一层。因此本函数的 original_encoder 实际是 patch_arm 的版本，
    覆盖后的 mode 会继续向下传递，消融逻辑不受影响。
    """
    from models import RAFT as RAFT_mod

    original_encoder = RAFT_mod.Model.encoder

    def encoder(self, x, index, mode):
        override = getattr(self, "_p33_vali_mode", None)
        if override is not None and mode == "valid":
            mode = override
        return original_encoder(self, x, index, mode)

    RAFT_mod.Model.encoder = encoder


# --------------------------------------------------------------------------
# 参数表：与论文 Table 6 (pred_len=96) 完全一致
# --------------------------------------------------------------------------
PAPER_HP = {
    "ETTh1": dict(seq_len=720, learning_rate=1e-3, topm=20),
    "ETTh2": dict(seq_len=720, learning_rate=1e-2, topm=10),
    "ETTm1": dict(seq_len=720, learning_rate=1e-2, topm=1),
    "ETTm2": dict(seq_len=720, learning_rate=1e-3, topm=5),
    "weather": dict(seq_len=720, learning_rate=1e-2, topm=1),
    "exchange_rate": dict(seq_len=720, learning_rate=1e-4, topm=1),
}

DATA_FILES = {
    "ETTh1": ("data/ETT/ETTh1.csv", "ETTh1", 7),
    "ETTh2": ("data/ETT/ETTh2.csv", "ETTh2", 7),
    "ETTm1": ("data/ETT/ETTm1.csv", "ETTm1", 7),
    "ETTm2": ("data/ETT/ETTm2.csv", "ETTm2", 7),
    "weather": ("data/custom/weather.csv", "custom", 21),
    "exchange_rate": ("data/custom/exchange_rate.csv", "custom", 8),
}


def build_args(dataset: str, seed: int, pred_len: int, train_epochs: int,
               num_workers: int) -> argparse.Namespace:
    """复刻 run.py 的默认参数，仅覆盖数据集相关项。"""
    hp = PAPER_HP[dataset]
    data_path, data_kind, enc_in = DATA_FILES[dataset]

    ns = argparse.Namespace()
    ns.task_name = "long_term_forecast"
    ns.is_training = 1
    ns.model_id = "p33"
    ns.model = "RAFT"
    ns.data = data_kind if data_kind != "custom" else "custom"
    ns.root_path = "./data/" + ("ETT/" if data_kind == "ETTh1" or data_kind == "ETTh2"
                                or data_kind == "ETTm1" or data_kind == "ETTm2" else "custom/")
    ns.data_path = os.path.basename(data_path)
    ns.features = "M"
    ns.target = "OT"
    ns.freq = "h"
    ns.checkpoints = "./checkpoints_p33/"
    ns.seq_len = hp["seq_len"]
    ns.label_len = 48
    ns.pred_len = pred_len
    ns.seasonal_patterns = "Monthly"
    ns.inverse = False
    ns.mask_rate = 0.25
    ns.anomaly_ratio = 0.25
    ns.expand = 2
    ns.d_conv = 4
    ns.top_k = 5
    ns.num_kernels = 6
    ns.enc_in = enc_in
    ns.dec_in = enc_in
    ns.c_out = enc_in
    ns.d_model = 512
    ns.n_heads = 8
    ns.e_layers = 2
    ns.d_layers = 1
    ns.d_ff = 2048
    ns.moving_avg = 25
    ns.factor = 1
    ns.distil = True
    ns.dropout = 0.1
    ns.embed = "timeF"
    ns.activation = "gelu"
    ns.channel_independence = 1
    ns.decomp_method = "moving_avg"
    ns.use_norm = 1
    ns.down_sampling_layers = 0
    ns.down_sampling_method = None
    ns.seg_len = 48
    ns.output_attention = False
    ns.n_period = 3
    ns.topm = hp["topm"]
    ns.num_workers = num_workers
    ns.itr = 1
    ns.train_epochs = train_epochs
    ns.batch_size = 32
    ns.patience = 10
    ns.learning_rate = hp["learning_rate"]
    ns.des = f"p33_{dataset}_{ARM['value']}_s{seed}"
    ns.loss = "MSE"
    ns.lradj = "type1"
    ns.use_amp = False
    ns.use_gpu = torch.cuda.is_available()
    ns.use_multi_gpu = False
    ns.devices = "0"
    ns.device_ids = [0]
    ns.gpu = 0
    ns.p_hidden_dims = [128, 128]
    ns.p_hidden_layers = 2
    ns.use_dtw = False
    ns.augmentation_ratio = 0
    ns.seed = seed
    for flag in ("jitter", "scaling", "permutation", "randompermutation",
                 "magwarp", "timewarp", "windowslice", "windowwarp",
                 "rotation", "spawner", "dtwwarp", "shapedtwwarp", "wdba",
                 "discdtw", "discsdtw"):
        setattr(ns, flag, False)
    ns.extra_tag = ""
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True, choices=["on", "A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--train_epochs", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # out 路径必须在 chdir 之前转成绝对路径——脚本稍后会 chdir 到 RAFT 目录，
    # 相对路径会落到 C:\tmp\raft\results\...（不存在）⇒ FileNotFoundError。
    out_path = Path(args.out).resolve()

    ARM["value"] = args.arm

    # 种子必须在任何 torch 随机调用前设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 官方 run.py 的确定性设置（本驱动绕过 run.py，必须自行继承，
    # 否则同一配置两次运行的 MSE 会有 ~0.02% 的漂移，无法做逐位比对）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.chdir(RAFT_DIR)
    patch_device_placement()
    patch_arm(args.arm)
    # 修正官方 vali() 的 mode 硬编码（上游 bug）：ETT 的 val/test 等长故不触发，
    # custom 系列（weather/exchange_rate）|val| != |test| 会索引越界而崩溃。
    patch_vali_mode()
    patch_vali_mode_forward()

    # 官方结果文件是全局追加的，运行时先记录基线长度，只解析本次新增的片段
    result_txt = Path("result_long_term_forecast.txt")
    base_len = result_txt.stat().st_size if result_txt.exists() else 0

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    ns = build_args(args.dataset, args.seed, args.pred_len,
                    args.train_epochs, args.num_workers)
    ns.is_training = 1

    # 检索库磁盘缓存（纯性能优化）。缓存键只含决定检索库内容的量：
    # 数据集 / seq_len / pred_len / topm / n_period。
    # **不含 arm 与 seed**——检索库与消融档位、随机种子无关。
    cache_key = f"{args.dataset}_sl{ns.seq_len}_pl{ns.pred_len}_topm{ns.topm}_np{ns.n_period}"
    patch_retrieval_cache(cache_key=cache_key,
                          cache_dir=Path(r"C:\tmp\p33_retrieval_cache"))

    exp = Exp_Long_Term_Forecast(ns)
    _mark("model+data init done")
    setting = f"{ns.task_name}_{ns.model_id}_{ns.model}_{ns.data}_ft{ns.features}" \
              f"_sl{ns.seq_len}_ll{ns.label_len}_pl{ns.pred_len}_{ns.des}"
    exp.train(setting)
    _mark("train done")
    exp.test(setting)
    _mark("test done")

    # 官方 test() 会把指标追加到 result_long_term_forecast.txt。
    # 官方同时调用 np.save(folder_path + 'metrics.npy', ...) 但在 Windows 下
    # 路径拼接后未落盘（目录存在但为空），因此改读官方 txt 输出——同一份数字。
    result_txt = Path("result_long_term_forecast.txt")
    if not result_txt.exists():
        raise FileNotFoundError("官方结果文件未生成")
    raw = result_txt.read_bytes()[base_len:].decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    mse = mae = None
    for i, ln in enumerate(lines):
        if ln.strip() == setting and i + 1 < len(lines) and "mse:" in lines[i + 1]:
            parts = lines[i + 1].split(",")
            for p in parts:
                k, _, v = p.partition(":")
                if k.strip() == "mse":
                    mse = float(v)
                elif k.strip() == "mae":
                    mae = float(v)
            break
    if mse is None:
        raise RuntimeError(f"未能在官方结果文件中定位 {setting}")

    rec = dict(dataset=args.dataset, arm=args.arm, seed=args.seed,
               pred_len=args.pred_len, seq_len=ns.seq_len,
               topm=ns.topm, learning_rate=ns.learning_rate,
               train_epochs=args.train_epochs,
               mae=mae, mse=mse, setting=setting)
    Path(out_path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(rec))


if __name__ == "__main__":
    main()
