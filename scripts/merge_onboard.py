"""Merge the sharded H6 onboarding outputs into one ``h6_onboarding.json``.

``run_onboard.py`` writes a single JSON per invocation, so a sharded run
(one shard per group of domains) produces several files that must be
concatenated before the statistics stage reads them.  This script performs
exactly that merge and nothing else, so the recorded protocol stays
reproducible:

    python scripts/run_onboard.py --domains <shard> --out results/ob1
    ...
    python scripts/merge_onboard.py            # -> results/synth/h6_onboarding.json

It is deliberately tolerant of missing shards: any shard directory that is
absent or empty is skipped, and the script reports how many rows came from
where.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*",
                    default=sorted(glob.glob(os.path.join(ROOT, "results", "ob*"))),
                    help="shard directories (default: results/ob*)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "synth",
                                                  "h6_onboarding.json"))
    a = ap.parse_args()

    rows, sources = [], []
    for d in a.shards:
        p = os.path.join(d, "h6_onboarding.json")
        if not os.path.exists(p):
            continue
        try:
            part = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- shard may be mid-write
            continue
        if not isinstance(part, list) or not part:
            continue
        rows.extend(part)
        sources.append((os.path.basename(d.rstrip("/\\")), len(part)))

    # de-duplicate on (domain, seed, expert): a repeated shard must not double count
    seen, uniq = set(), []
    for r in rows:
        key = (r.get("domain"), r.get("seed"), r.get("expert"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(uniq, open(a.out, "w", encoding="utf-8"), indent=1, default=str)

    for name, n in sources:
        print(f"  {name:<8} {n:>4} rows")
    print(f"merged {len(rows)} rows -> {len(uniq)} unique "
          f"({len(rows) - len(uniq)} duplicates dropped)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
