"""W-P3-3 层 2 复现：批量运行器。

按 (dataset, arm, seed) 逐个起独立进程（显存/状态隔离），
结果落到 results/p33_raw/<dataset>_<arm>_s<seed>.json。

用法：
    python scripts/run_p33_batch.py --datasets ETTh1 ETTh2 --arms on A B --seeds 0 1 2 3 4
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Paper-Neuro-STOG")
PY = r"C:\code\T\Scripts\python.exe"
OUT_DIR = ROOT / "results" / "p33_raw"


def run_one(dataset: str, arm: str, seed: int, train_epochs: int,
            timeout: int, log_dir: Path) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{dataset}_{arm}_s{seed}.json"
    if out_json.exists():
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            out_json.unlink()

    # 原子写：先写临时文件再改名，避免并发实例读到半截文件
    tmp_json = out_json.with_suffix(".json.tmp")

    log = log_dir / f"{dataset}_{arm}_s{seed}.log"
    cmd = [PY, str(ROOT / "scripts" / "run_p33_raft.py"),
           "--dataset", dataset, "--arm", arm, "--seed", str(seed),
           "--train_epochs", str(train_epochs), "--out", str(tmp_json)]
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="ignore") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              timeout=timeout)
    dt = time.time() - t0
    if proc.returncode != 0 or not tmp_json.exists():
        if tmp_json.exists():
            tmp_json.unlink()
        return dict(dataset=dataset, arm=arm, seed=seed, status="failed",
                    returncode=proc.returncode, seconds=round(dt, 1),
                    log=str(log))
    tmp_json.replace(out_json)
    rec = json.loads(out_json.read_text(encoding="utf-8"))
    rec["status"] = "ok"
    rec["seconds"] = round(dt, 1)
    rec["log"] = str(log)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["on", "A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train_epochs", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--log_dir", default=r"C:\tmp\p33_logs")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发子进程数（检索库构建是 CPU 密集，GPU 利用率低）")
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(ds, arm, seed) for ds in args.datasets
            for arm in args.arms for seed in args.seeds]
    pending = [j for j in jobs
               if not (OUT_DIR / f"{j[0]}_{j[1]}_s{j[2]}.json").exists()]
    print(f"总任务 {len(jobs)}，待跑 {len(pending)}，并发 {args.workers}", flush=True)

    results = {}
    if args.workers <= 1:
        for ds, arm, seed in pending:
            rec = run_one(ds, arm, seed, args.train_epochs, args.timeout, log_dir)
            results[(ds, arm, seed)] = rec
            _report(rec)
    else:
        from concurrent.futures import ThreadPoolExecutor
        # 每个 job 自身是独立子进程，线程池只做调度
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, ds, arm, seed, args.train_epochs,
                               args.timeout, log_dir): (ds, arm, seed)
                    for ds, arm, seed in pending}
            for fut in futs:
                pass
            for fut, key in futs.items():
                rec = fut.result()
                results[key] = rec
                _report(rec)

    summary = list(results.values())
    out = OUT_DIR / "_batch_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n汇总写入 {out}")


def _report(rec: dict) -> None:
    tag = f"{rec.get('status'):>6s} {rec['dataset']:12s} arm={rec['arm']:2s} s={rec['seed']}"
    if rec.get("status") == "ok":
        tag += f"  mse={rec['mse']:.4f}  mae={rec['mae']:.4f}  {rec['seconds']}s"
    else:
        tag += f"  rc={rec.get('returncode')}  {rec.get('seconds')}s"
    print(tag, flush=True)


if __name__ == "__main__":
    main()
