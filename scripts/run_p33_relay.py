"""W-P3-3 层 2：接力执行器。

按预注册 v1.3 的最终矩阵串行执行：
  主矩阵：6 数据集 × {on, B} × 3 种子 = 36 次
  补充档：ETTh1 + exchange_rate × A × 2 种子 = 4 次
共 40 次。

串行的理由（实测）：3 并发时单次运行从 5.5 分钟劣化到 26 分钟
（GPU 显存带宽争用），并发有害。

以 results/p33_raw/*.json 是否存在作为幂等标记，可随时中断续跑。
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Paper-Neuro-STOG")
PY = r"C:\code\T\Scripts\python.exe"
OUT = ROOT / "results" / "p33_raw"

MAIN_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange_rate"]
MAIN_ARMS = ["on", "B"]
MAIN_SEEDS = [0, 1, 2]

# 补充档 A：只跑对照性最强的两个数据集
EXTRA_ARMS = ["A"]
EXTRA_DATASETS = ["ETTh1", "exchange_rate"]
EXTRA_SEEDS = [0, 1]


def jobs():
    out = []
    for ds in MAIN_DATASETS:
        for arm in MAIN_ARMS:
            for s in MAIN_SEEDS:
                out.append((ds, arm, s))
    for ds in EXTRA_DATASETS:
        for arm in EXTRA_ARMS:
            for s in EXTRA_SEEDS:
                out.append((ds, arm, s))
    return out


def main() -> None:
    log_dir = Path(r"C:\tmp\p33_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs()
            if not (OUT / f"{j[0]}_{j[1]}_s{j[2]}.json").exists()]
    print(f"总任务 {len(jobs())}，待跑 {len(todo)}", flush=True)
    t_start = time.time()
    for i, (ds, arm, seed) in enumerate(todo, 1):
        out_json = OUT / f"{ds}_{arm}_s{seed}.json"
        tmp_json = out_json.with_suffix(".json.tmp")
        log = log_dir / f"{ds}_{arm}_s{seed}.log"
        cmd = [PY, str(ROOT / "scripts" / "run_p33_raft.py"),
               "--dataset", ds, "--arm", arm, "--seed", str(seed),
               "--train_epochs", "10", "--out", str(tmp_json)]
        t0 = time.time()
        with log.open("w", encoding="utf-8", errors="ignore") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=5400).returncode
        dt = time.time() - t0
        if rc == 0 and tmp_json.exists():
            tmp_json.replace(out_json)
            import json
            rec = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[{i}/{len(todo)}] ok  {ds:14s} arm={arm:2s} s={seed}  "
                  f"mse={rec['mse']:.4f}  mae={rec['mae']:.4f}  {dt:.0f}s  "
                  f"总耗时 {(time.time()-t_start)/60:.1f}min", flush=True)
        else:
            if tmp_json.exists():
                tmp_json.unlink()
            print(f"[{i}/{len(todo)}] FAIL {ds:14s} arm={arm:2s} s={seed}  "
                  f"rc={rc}  {dt:.0f}s  日志 {log}", flush=True)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
