"""Global configuration for the CerebroCast experiment suite.

Every constant that can influence a reported number lives here so that a run is
fully described by (config snapshot + random seed).  Nothing in the code path
reads the test split for calibration: the five free scalars that are tuned
(theta1, theta2, gamma, eligibility half-life, k range) are fitted on the
validation split only, then frozen (see scripts/run_main.py --stage).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEEDS: List[int] = [7, 42, 2021, 3407, 12345]
CONFIRM_SEEDS: List[int] = [7, 12345]

# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------
# NOTE: the split actually used by the pipeline is the four-stage growing-block
# protocol in scripts/build_pool.py:
#     fit 55% | warm 10% | val 10% | test 25%
# It replaced the former three-stage 0.7/0.1/0.2 constant below, which was left
# here unused and had become a misleading "text says one thing, code does
# another" artefact (review item B3/C11).  It is kept only as a documented
# historical value, never read.
LEGACY_SPLIT_UNUSED = (0.7, 0.1, 0.2)
LOOKBACK_DAYS = 7                # window length = 7 days
SEASON_DAYS = 7                  # weekly seasonal lags retained

# ----------------------------------------------------------------------------
# Architecture hyper-parameters (see paper Section 4)
# ----------------------------------------------------------------------------
@dataclass
class SystemConfig:
    # (1) fingerprint encoder
    n_spec_bins: int = 32           # K, spectral shape bins
    spec_log: bool = True           # log-periodogram before binning

    # (2) episodic memory
    episodic_cap: int = 4000        # hard cap on stored episodes
    episodic_top_s: int = 12        # neighbours retrieved each day

    # (3) slow affinity graph / cluster field
    k0_clusters: int = 6            # clusters seeded from the training split
    tau_kappa: float = 0.35         # temperature of the affinity kernel
    graph_alpha: float = 0.30       # message-passing strength on the slow graph

    # (4) change detection + thalamic gating
    detect_window: int = 28         # w, days in the reference sliding window
    hold_gain: float = 0.35         # plasticity gain while the gate is in HOLD
    theta1: float = 2.00            # HOLD -> ADAPT threshold (robust z units)
    theta2: float = 3.20            # ADAPT -> RESET threshold (robust z units)
    detect_persist: int = 2         # consecutive days above theta2 to fire RESET

    # (5) LC-NE reset
    reset_decay: float = 0.35       # prior retained after a reset
    reset_lr_boost: float = 4.0     # transient learning-rate multiplier
    reset_temp_boost: float = 1.60  # transient exploration temperature
    reset_horizon: int = 7          # days the transient lasts

    # (6) responsibility system
    beta_forward: float = 1.00      # weight of the forward error model in gating
    topk: int = 5                   # k_max of the sparse responsibility set
    lambda_temp: float = 1.00       # softmax temperature of the responsibility

    # (7) three-factor plasticity + eligibility traces
    eta: float = 0.02               # base learning rate
    trace_decay: float = 0.99       # recency of the accumulated logit (~100 days)
    elig_halflife: int = 3          # rho, days
    winsor_q: float = 0.95          # rolling winsorising quantile (frozen)

    # (8) astrocytic metaplasticity
    astro_up: float = 0.15          # rate of the slow variable upward drift
    astro_down: float = 0.60        # rate of the slow variable decay
    astro_floor: float = 0.50
    astro_ceil: float = 2.00

    # (9) sleep phase
    gamma0: float = 3e-3            # synaptic downscaling per day
    replay_frac: float = 0.03       # fraction of the day buffer replayed (≈3-4 days)
    replay_lr: float = 0.30         # learning-rate multiplier during replay

    # (9a) reconsolidation -- the slow weight source
    recall_halflife: float = 10.0   # recency half-life of the second moments (days)
    diff_q: float = 0.25            # quantile of the day's expert losses = difficulty
    consol_ridge: float = 1e-2      # relative ridge on the second-moment matrix
    consol_shrink: float = 2.0      # shrinkage of a cluster towards the population
    consol_shrink_u: float = 0.0    # shrinkage of the population solution to uniform
    adapt_forget_boost: float = 1.5  # forgetting multiplier while the gate is ADAPT
    reset_forget_boost: float = 4.0  # forgetting multiplier while the gate is RESET

    grow_r_cov: float = 1.60        # fingerprint distance that triggers growth
    grow_persist: int = 3           # days a novel fingerprint must persist
    prune_mass: float = 1e-3        # responsibility mass below which a cluster dies
    prune_idle: int = 45            # days of neglect before a cluster dies
    cluster_momentum: float = 0.05  # slow drift of a cluster centre

    # admin
    seed: int = 7
    loss_cap_mult: float = 8.0      # winsorising envelope on per-expert losses
    w_cap: float = 60.0             # homeostatic bound on synaptic efficacy


@dataclass
class Ablation:
    """One arm of the mechanism-pricing design.

    Every flag removes exactly one mechanism from the full architecture.  The
    'placebo' flag marks arms that remove the *structure* of a mechanism while
    preserving its *rate/capacity*, which is what separates "this mechanism
    carries information" from "this mechanism just does something often".
    """
    name: str
    no_spectral: bool = False      # A1 : fingerprint drops the spectral-shape factor
    no_episodic: bool = False      # A2 : no episodic retrieval into the affinities
    no_clusters: bool = False      # A3 : one population combination, no cluster field
    no_detect: bool = False        # A4 : gating pinned to ADAPT
    no_reset: bool = False         # A5 : RESET demoted to ADAPT
    no_forward: bool = False       # A6 : forward error model gate disabled
    no_three_factor: bool = False  # A7 : updates written uniformly to every cluster
    no_astro: bool = False         # A8 : constant learning rate
    no_replay: bool = False        # A9 : replay and cross-day linking disabled
    no_downscale: bool = False     # A10 : no synaptic downscaling
    no_growth: bool = False        # A11 : no cluster growth / pruning
    placebo: str = ""              # P1..P4 markers
    description: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


def default_arms() -> List[Ablation]:
    """The 9 ablation arms + 4 placebo arms + the full system."""
    arms = [
        Ablation("FULL", description="complete architecture"),
        Ablation("A1_no_spectral", no_spectral=True,
                 description="fingerprint reduced to amplitude + regime factors"),
        Ablation("A2_no_episodic", no_episodic=True,
                 description="episodic retrieval disabled (slow graph only)"),
        Ablation("A3_no_clusters", no_clusters=True,
                 description="one population combination; cluster field not used"),
        Ablation("A4_no_change_detection", no_detect=True,
                 description="gating pinned to ADAPT (no HOLD, no RESET)"),
        Ablation("A5_no_reset", no_reset=True,
                 description="RESET demoted to ADAPT with unchanged parameters"),
        Ablation("A6_no_forward_model", no_forward=True,
                 description="predicted-error gate disabled"),
        Ablation("A7_no_three_factor", no_three_factor=True,
                 description="updates written uniformly to every cluster"),
        Ablation("A8_no_metaplasticity", no_astro=True,
                 description="constant learning rate (slow variable frozen at 1)"),
        Ablation("A9_no_replay", no_replay=True,
                 description="sleep replay and cross-day linking disabled"),
        Ablation("A10_no_downscaling", no_downscale=True,
                 description="synaptic downscaling disabled"),
        Ablation("A11_no_growth", no_growth=True,
                 description="cluster field frozen (no growth, no pruning)"),
        Ablation("P1_random_reset", placebo="reset",
                 description="RESET fires at the same rate but at random days"),
        Ablation("P2_random_replay", placebo="replay",
                 description="replay samples uniformly instead of by priority"),
        Ablation("P3_shuffled_responsibility", placebo="responsibility",
                 description="lambda permuted across experts each day, BEFORE the "
                             "mixture: prices 'sound mix vs random mix' (engine "
                             "evidence), not the responsibility ordering"),
        Ablation("P3b_shuffled_credit", placebo="responsibility_learn",
                 description="fusion weights left intact; only the lambda that "
                             "carries credit into the replay path is permuted -- "
                             "the clean placebo for the responsibility ordering"),
        Ablation("P4_random_growth", placebo="growth",
                 description="cluster growth fires at random days"),
    ]
    return arms


# ----------------------------------------------------------------------------
# Pre-registered thresholds (SI-B)
# ----------------------------------------------------------------------------
PREREG = {
    "H1_main_gain_pct": 8.0,       # >= 8% cumulative MSE gain vs best baseline
    "H1a_domains": ("EPF-NP", "EPF-PJM", "EPF-BE", "EPF-FR", "EPF-DE"),
    "H2_recovery_days": 10.0,      # <= 10 days to recover after a break
    "H3_universality_pct": 5.0,    # median relative degradation <= 5%
    "H4_detect_macro_f1": 0.70,    # three-tier vs truth, macro F1
    # H5 denominator: 11 module ablations are run (A1..A11); A1 is a *feature*
    # ablation and is reported separately, so the module count is 10 while the
    # pre-registered 6-of-9 threshold is kept verbatim for the record.  Both are
    # reported (review item C11/24: the arm count changed without revisiting the
    # threshold -- v2 reports both readings side by side).
    "H5_pricing_modules": 6,
    "H5_denominator_arms": 11,
    "H5_denominator_module_arms": 10,
    "H6_onboard_days": 30.0,
    "H7_cost_fraction": 0.10,      # incremental compute < 10% of monthly retrain
    "equivalence_margin_pct": 1.0,  # TOST margin = 1% of reference MSE
}
