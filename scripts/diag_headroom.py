"""How much headroom does per-day expert selection actually have?

Compares, on the same cached pool:
  oracle        - per-day best expert (upper bound, not attainable)
  static        - best fixed combination (B1)
  ridge-select  - offline regression from the fingerprint to each expert's loss,
                  then pick the arg-min (an *offline* upper bound for any
                  fingerprint-driven selector; uses test-day losses for fitting
                  so it is only a ceiling, never a baseline)
  ridge-select-causal - the same model fitted on training days only (attainable)
  last-day      - pick yesterday's best expert
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cerebrocast.fingerprints import Fingerprint  # noqa: E402

POOL = os.path.join(os.path.dirname(__file__), "..", "results", "pool")


def headroom(domain, seed):
    z = np.load(os.path.join(POOL, f"{domain}__seed{seed}.npz"), allow_pickle=True)
    P, Y, X = z["P"], z["Y"], z["X"]
    itr, ite = z["idx_train"], z["idx_test"]
    fp = Fingerprint(K=32); fp.fit(X[itr])
    PHI = fp.transform(X)
    lm = np.mean((P - Y[:, None, :]) ** 2, axis=2)              # (D, M)
    D, M = lm.shape

    def pick(idx_pred):
        idx_pred = np.asarray(idx_pred)
        return float(np.mean(lm[ite, idx_pred]))

    oracle = float(np.mean(lm[ite].min(axis=1)))
    # offline ceiling: fit on the test days themselves
    Zte = np.column_stack([PHI[ite], np.ones(len(ite))])
    Wte = np.linalg.lstsq(Zte, lm[ite], rcond=None)[0]
    pred_ceiling = Zte @ Wte
    ceiling = pick(np.argmin(pred_ceiling, axis=1))

    # causal: fit on the training days
    Ztr = np.column_stack([PHI[itr], np.ones(len(itr))])
    Wtr = np.linalg.lstsq(Ztr, lm[itr], rcond=None)[0]
    pred_c = Zte @ Wtr
    causal = pick(np.argmin(pred_c, axis=1))

    # last-day winner
    last = np.zeros(len(ite), dtype=int)
    lmprev = np.vstack([lm[itr][-1:], lm[ite[:-1]]])
    last = np.argmin(lmprev, axis=1)

    # soft combination from the causal predictor
    tau = 0.15
    pr = pred_c - pred_c.min(axis=1, keepdims=True)
    w = np.exp(-pr / (tau * (np.abs(pr).mean() + 1e-9)))
    w = w / w.sum(axis=1, keepdims=True)
    soft = float(np.mean([np.mean((w[i] @ P[ite][i] - Y[ite][i]) ** 2)
                          for i in range(len(ite))]))

    return dict(domain=domain, oracle=oracle, ceiling=ceiling, causal=causal,
                last_day=pick(last), soft_causal=soft,
                mean_expert=float(np.mean(lm[ite])))


if __name__ == "__main__":
    for dom in sys.argv[1:]:
        r = headroom(dom, 7)
        print(f"{dom:9s} oracle={r['oracle']:.4f} ceiling={r['ceiling']:.4f} "
              f"causal={r['causal']:.4f} last_day={r['last_day']:.4f} "
              f"soft_causal={r['soft_causal']:.4f} mean={r['mean_expert']:.4f}")
