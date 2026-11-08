"""Summarise batch CSVs written by scripts/batch_matches.py into the JSON/CSV the site build reads.

  pixi run python scripts/summarise_batch.py \
      --main results/batch2/main_current.csv results/batch2/main_baseline.csv \
      --sweep results/batch2/sweep_current.csv results/batch2/sweep_baseline.csv \
      --out results/batch2/summary.json --csv results/batch2/summary.csv

The two teams of a same-brain match are not independent (one match, one ball, one network), so each metric is reduced to
one value per match, the mean of the home and away team values (a team without a value is skipped); n = matches. goal_diff is
identically 0 for that reduction and is not reported here (head-to-head: --h2h). Brain column
`current` -> `new`, `baseline` -> `pre`. Statistics are pure Python (no scipy): Student t 95% CI, Mann-Whitney U two-sided
(normal approximation, tie and continuity corrected), pct_change = (new - pre) / pre * 100 (null if pre is 0).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

BRAIN_LABEL = {"current": "new", "baseline": "pre"}
METRICS = [
    "chase_flips_per_min", "role_flips_per_min", "lead_changes_per_min", "decision_flips_per_min",
    "duplicate_decision_frac", "duplicate_lead_frac", "duplicate_striker_frac", "chaser_present_frac",
    "reaction_median_s", "reaction_p90_s", "reaction_censored", "fused_ball_err_m", "local_ball_err_m",
    "mode_healthy", "mode_degraded", "mode_isolated",
]
SEG_METRICS = ["role_flips_per_min", "lead_changes_per_min", "duplicate_striker_frac",
               "reaction_median_s", "reaction_p90_s", "reaction_censored"]
SIGNAL_COL = {"role_flips_per_min": "role_signal", "duplicate_striker_frac": "duplicate_signal"}
ALL_METRICS = METRICS + ["goals"] + [f"{s}_{m}" for s in ("bo", "out") for m in SEG_METRICS]
SWEEP_RE = re.compile(r"^lat(\d+(?:\.\d+)?)_loss(\d+(?:\.\d+)?)$")


# ---------------------------------------------------------------- statistics
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    p = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - p if t > 0 else p


def t_quantile(p, df):
    """Quantile of Student t (bisection on the CDF); p in (0, 1)."""
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_ci(xs, level=0.95):
    """(mean, lo, hi); lo/hi None for n < 2."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = t_quantile(0.5 + level / 2.0, n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mann_whitney_p(a, b):
    """Two-sided Mann-Whitney U p-value, normal approximation with tie and continuity correction."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    n = n1 + n2
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0.0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    z = max(z, 0.0)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def binom_two_sided_p(k, n):
    """Exact two-sided sign-test p (p0 = 0.5) for k successes in n trials."""
    if n == 0:
        return None
    pmf = [math.comb(n, i) / 2.0 ** n for i in range(n + 1)]
    return min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9)))


def wilcoxon_signed_rank_p(xs):
    """Two-sided Wilcoxon signed-rank p (zeros dropped), normal approximation with tie and continuity correction."""
    d = [x for x in xs if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, tie_term, i = [0.0] * n, 0.0, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = 0.5 * (i + j) + 1.0
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return 1.0
    z = max(abs(w - mu) - 0.5, 0.0) / math.sqrt(var)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- head-to-head
def h2h(paths, perspective="home"):
    """Per (profile, brains, team profiles) group: goal difference and goals per match from one side's view.
    perspective 'home' = the CSV's home team; a brain name ('current'/'baseline') = that brain's team, whichever side it
    played (so a side-swapped second run pools with the first). n = matches."""
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                gh, ga = num(row.get("goals_home")), num(row.get("goals_away"))
                hb, ab = (row.get("home_brain") or row.get("brain") or "").strip(), (row.get("away_brain") or row.get("brain") or "").strip()
                if gh is None or ga is None or not hb or not ab:
                    continue
                hp, ap = (row.get("home_profile") or row.get("profile") or "").strip(), (row.get("away_profile") or row.get("profile") or "").strip()
                if perspective == "home" or hb == perspective:
                    own, opp, ob, pb, op, pp = gh, ga, hb, ab, hp, ap
                elif ab == perspective:
                    own, opp, ob, pb, op, pp = ga, gh, ab, hb, ap, hp
                else:
                    continue
                groups[((row.get("profile") or "").strip(), ob, pb, op, pp)].append((own, opp))
    out = []
    for (prof, ob, pb, op, pp), g in groups.items():
        gd = [a - b for a, b in g]
        wins, losses = sum(x > 0 for x in gd), sum(x < 0 for x in gd)
        m, lo, hi = mean_ci(gd)
        om, olo, ohi = mean_ci([a for a, _ in g])
        pm, plo, phi = mean_ci([b for _, b in g])
        out.append({"profile": prof, "own_brain": ob, "opp_brain": pb, "own_network": op, "opp_network": pp,
                    "n": len(g), "goal_diff_mean": m, "goal_diff_ci_lo": lo, "goal_diff_ci_hi": hi,
                    "goals_for_mean": om, "goals_for_ci_lo": olo, "goals_for_ci_hi": ohi,
                    "goals_against_mean": pm, "goals_against_ci_lo": plo, "goals_against_ci_hi": phi,
                    "wins": wins, "draws": len(gd) - wins - losses, "losses": losses,
                    "p_sign": binom_two_sided_p(wins, wins + losses), "p_wilcoxon": wilcoxon_signed_rank_p(gd)})
    return out


def fmt_h2h(rows):
    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = []
    for r in rows:
        lines.append(f"{r['profile']}: {r['own_brain']}({r['own_network']}) vs {r['opp_brain']}({r['opp_network']})  n={r['n']}  "
                     f"GD {f(r['goal_diff_mean'])} [{f(r['goal_diff_ci_lo'])}, {f(r['goal_diff_ci_hi'])}]  "
                     f"GF {f(r['goals_for_mean'])} [{f(r['goals_for_ci_lo'])}, {f(r['goals_for_ci_hi'])}]  "
                     f"GA {f(r['goals_against_mean'])} [{f(r['goals_against_ci_lo'])}, {f(r['goals_against_ci_hi'])}]  "
                     f"W/D/L {r['wins']}/{r['draws']}/{r['losses']}  p_sign {f(r['p_sign'])}  p_wilcoxon {f(r['p_wilcoxon'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------- per game (full games, batch_matches --half-minutes)
GAME_METRICS = ["goals", "chase_flips_per_min", "duplicate_decision_frac", "chaser_present_frac",
                "reaction_median_s", "reaction_p90_s", "fused_ball_err_m"]


def games(paths):
    """One entry per game and team, and per network profile a summary. A team's value is its own (home_* / away_* column,
    goals = its goals). Per profile and side (home/away): mean with a Student t 95% CI over games. Per profile: Mann-Whitney U
    between the new-brain teams' values and the pre-change teams' values (current -> new, baseline -> pre, whichever side they
    played), with the mean of each and pct_change. Games of same-brain runs have no such pair: the test needs both brains."""
    teams = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                prof = (row.get("profile") or "").strip()
                if not prof or num(row.get("goals_home")) is None or num(row.get("goals_away")) is None:
                    continue
                game = {"profile": prof, "match": row.get("match"), "seed": row.get("seed"), "play_s": num(row.get("play_s"))}
                for side in ("home", "away"):
                    brain = (row.get(f"{side}_brain") or row.get("brain") or "").strip()
                    net = (row.get(f"{side}_profile") or prof).strip()
                    t = {**game, "side": side, "brain": brain, "label": BRAIN_LABEL.get(brain, brain), "network": net,
                         "goals": num(row.get(f"goals_{side}")), "goals_against": num(row.get(f"goals_{'away' if side == 'home' else 'home'}"))}
                    for m in GAME_METRICS[1:]:
                        t[m] = num(row.get(f"{side}_{m}"))
                    teams.append(t)
    sides, tests = [], []
    for prof in dict.fromkeys(t["profile"] for t in teams):
        mine = [t for t in teams if t["profile"] == prof]
        for side in ("home", "away"):
            for brain in dict.fromkeys(t["brain"] for t in mine if t["side"] == side):
                grp = [t for t in mine if t["side"] == side and t["brain"] == brain]
                for m in GAME_METRICS:
                    xs = [t[m] for t in grp if t[m] is not None]
                    if xs:
                        mean, lo, hi = mean_ci(xs)
                        sides.append({"profile": prof, "side": side, "brain": brain, "metric": m, "n": len(xs),
                                      "mean": mean, "ci_lo": lo, "ci_hi": hi})
        for m in GAME_METRICS:
            new = [t[m] for t in mine if t["label"] == "new" and t[m] is not None]
            pre = [t[m] for t in mine if t["label"] == "pre" and t[m] is not None]
            if new and pre:
                mn, mp = sum(new) / len(new), sum(pre) / len(pre)
                tests.append({"profile": prof, "metric": m, "n_new": len(new), "n_pre": len(pre), "mean_new": mn, "mean_pre": mp,
                              "pct_change": None if mp == 0 else (mn - mp) / mp * 100.0, "p_mwu": mann_whitney_p(new, pre)})
    return {"teams": teams, "sides": sides, "tests": tests}


def fmt_games(res):
    f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    lines = ["per game and team:"]
    for t in res["teams"]:
        lines.append(f"  {t['profile']} game {t['match']} {t['side']} {t['brain']}({t['network']}): goals {f(t['goals'])}  "
                     + "  ".join(f"{m} {f(t[m])}" for m in GAME_METRICS[1:]))
    lines.append("per profile and side (mean [95% CI] over games):")
    for r in res["sides"]:
        lines.append(f"  {r['profile']} {r['side']} {r['brain']} {r['metric']}: n={r['n']} {f(r['mean'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}]")
    lines.append("new vs pre-change team values (Mann-Whitney U):")
    for r in res["tests"]:
        lines.append(f"  {r['profile']} {r['metric']}: new {f(r['mean_new'])} (n={r['n_new']}) pre {f(r['mean_pre'])} (n={r['n_pre']}) "
                     f"change {f(r['pct_change'])}%  p {f(r['p_mwu'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------- loading
def num(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


class Store:
    """samples[(profile, brain, metric)] -> list of per-match values (mean of the two teams); signals[(profile, brain, metric)] -> set of strings."""

    def __init__(self):
        self.samples = defaultdict(list)
        self.signals = defaultdict(set)
        self.matches = defaultdict(int)       # brain -> matches
        self.play_s = []
        self.profiles = []

    def add_csv(self, path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                brain = BRAIN_LABEL.get((row.get("brain") or "").strip())
                prof = (row.get("profile") or "").strip()
                if brain is None or not prof:
                    continue
                if prof not in self.profiles:
                    self.profiles.append(prof)
                self.matches[brain] += 1
                ps = num(row.get("play_s"))
                if ps is not None:
                    self.play_s.append(ps)
                per_match = defaultdict(list)  # metric -> [home value, away value]
                for team, gcol in (("home", "goals_home"), ("away", "goals_away")):
                    g = num(row.get(gcol))
                    if g is not None:
                        per_match["goals"].append(g)
                    for m in METRICS:
                        self._take(row, team, prof, brain, m, m, per_match)
                    for seg in ("bo", "out"):
                        for m in SEG_METRICS:
                            self._take(row, team, prof, brain, f"{seg}_{m}", f"{seg}_{m}", per_match, sig_of=m)
                for metric, vals in per_match.items():
                    self.samples[(prof, brain, metric)].append(sum(vals) / len(vals))  # one value per match

    def _take(self, row, team, prof, brain, col, metric, per_match, sig_of=None):
        v = num(row.get(f"{team}_{col}"))
        if v is None:
            return
        per_match[metric].append(v)
        sig = SIGNAL_COL.get(sig_of or metric)
        if sig:
            s = (row.get(f"{team}_{sig}") or "").strip()
            if s:
                self.signals[(prof, brain, metric)].add(s)


def signal_str(store, prof, brain, metric):
    s = store.signals.get((prof, brain, metric))
    return "+".join(sorted(s)) if s else None


# ---------------------------------------------------------------- summary
def build(main_paths, sweep_paths, brains):
    main, sweep = Store(), Store()
    for p in main_paths:
        main.add_csv(p)
    for p in sweep_paths:
        sweep.add_csv(p)

    rows, tests, notes = [], [], []
    for prof in main.profiles:
        for metric in ALL_METRICS:
            per = {}
            for brain in ("new", "pre"):
                xs = main.samples.get((prof, brain, metric))
                if not xs:
                    continue
                m, lo, hi = mean_ci(xs)
                per[brain] = xs
                row = {"profile": prof, "brain": brain, "metric": metric, "n": len(xs),
                       "mean": m, "ci_lo": lo, "ci_hi": hi, "median": median(xs)}
                sig = signal_str(main, prof, brain, metric)
                if sig:
                    row["signal"] = sig
                rows.append(row)
            if "new" in per and "pre" in per:
                mn, mp = sum(per["new"]) / len(per["new"]), sum(per["pre"]) / len(per["pre"])
                t = {"profile": prof, "metric": metric,
                     "pct_change": None if mp == 0 else (mn - mp) / mp * 100.0,
                     "p_mwu": mann_whitney_p(per["new"], per["pre"])}
                sn, sp = signal_str(main, prof, "new", metric), signal_str(main, prof, "pre", metric)
                if sn and sp and sn != sp:
                    t["signal_note"] = f"new={sn}, pre={sp}"
                    notes.append(f"{prof}/{metric}: different signals (new={sn}, pre={sp})")
                tests.append(t)

    sweeps = []
    for prof in sweep.profiles:
        mt = SWEEP_RE.match(prof)
        if not mt:
            continue
        lat, loss = float(mt.group(1)), float(mt.group(2))
        axes = []
        if loss == 15:
            axes.append(("latency_ms", lat))
        if lat == 150:
            axes.append(("loss_pct", loss))
        for axis, val in axes:
            for brain in ("new", "pre"):
                for metric in ALL_METRICS:
                    xs = sweep.samples.get((prof, brain, metric))
                    if not xs:
                        continue
                    m, lo, hi = mean_ci(xs)
                    sweeps.append({"axis": axis, "value": int(val) if val == int(val) else val, "brain": brain,
                                   "metric": metric, "mean": m, "ci_lo": lo, "ci_hi": hi})

    n_matches = sum(main.matches.values()) + sum(sweep.matches.values())
    plays = main.play_s + sweep.play_s
    meta = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "brains": brains,
        "n_matches": n_matches,
        "unit": "match",  # every row's n counts matches: one value per match (mean of the two teams)
        "n": n_matches,
        "duration_s": round(sum(plays) / len(plays), 2) if plays else None,
        "profiles": main.profiles,
        "notes": ("One value per match (mean of the home and away team values), n = matches. CI: Student t 95%. "
                  "p_mwu: Mann-Whitney U two-sided on the per-match values. goals = mean goals per team per match. "
                  + ("; ".join(sorted(set(notes))) if notes else "No signal differences between brains.")),
    }
    return {"meta": meta, "rows": rows, "tests": tests, "sweeps": sweeps}


def write_csv(path, summary):
    cols = ["kind", "profile", "axis", "value", "brain", "metric", "n", "mean", "ci_lo", "ci_hi", "median",
            "pct_change", "p_mwu", "signal"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for kind, key in (("row", "rows"), ("test", "tests"), ("sweep", "sweeps")):
            for r in summary[key]:
                w.writerow({"kind": kind, **{k: ("" if v is None else v) for k, v in r.items()}})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h2h", nargs="+", default=None, metavar="CSV",
                    help="head-to-head mode: goal difference and goals per match from one side's view, with a sign test and "
                         "Wilcoxon (replaces --main/--sweep; --out optional JSON)")
    ap.add_argument("--games", nargs="+", default=None, metavar="CSV",
                    help="full-game mode: goals and metrics per game and team, per profile and side mean/95% CI over games, "
                         "and Mann-Whitney U new brain vs pre-change (--out optional JSON)")
    ap.add_argument("--perspective", default="home", choices=("home", "current", "baseline"),
                    help="--h2h: whose goals count as 'for' (a brain pools side-swapped runs)")
    ap.add_argument("--main", nargs="+", default=[], help="main batch CSVs (all brains)")
    ap.add_argument("--sweep", nargs="*", default=[], help="sweep batch CSVs (profiles lat<ms>_loss<pct>)")
    ap.add_argument("--out", default=None, help="summary JSON")
    ap.add_argument("--csv", default=None, help="tidy summary CSV")
    ap.add_argument("--new-hash", default="26b1193")
    ap.add_argument("--pre-hash", default="8d241a2")
    a = ap.parse_args(argv)
    if a.games:
        res = games(a.games)
        print(fmt_games(res) if res["teams"] else "no games")
        if a.out:
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        return 0
    if a.h2h:
        rows = h2h(a.h2h, a.perspective)
        print(fmt_h2h(rows) or "no head-to-head rows")
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"perspective": a.perspective, "rows": rows}, f, indent=1)
        return 0
    if not a.main or not a.out:
        ap.error("--main and --out are required (or use --h2h)")
    s = build(a.main, a.sweep, {"new": a.new_hash, "pre": a.pre_hash})
    with open(a.out, "w") as f:
        json.dump(s, f, indent=1)
    if a.csv:
        write_csv(a.csv, s)
    print(f"{len(s['rows'])} rows, {len(s['tests'])} tests, {len(s['sweeps'])} sweep rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
