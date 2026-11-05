import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "scripts" / "summarise_batch.py"
_spec = importlib.util.spec_from_file_location("summarise_batch", _p)
sb = importlib.util.module_from_spec(_spec)
sys.modules["summarise_batch"] = sb
_spec.loader.exec_module(sb)

COLS = ["profile", "brain", "match", "play_s", "goals_home", "goals_away", "goal_diff",
        "home_role_signal", "home_role_flips_per_min", "home_duplicate_signal", "home_duplicate_striker_frac",
        "home_duplicate_lead_frac", "home_bo_duplicate_striker_frac", "home_out_duplicate_striker_frac",
        "away_role_signal", "away_role_flips_per_min", "away_duplicate_signal", "away_duplicate_striker_frac",
        "away_duplicate_lead_frac", "away_bo_duplicate_striker_frac", "away_out_duplicate_striker_frac"]


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    return str(path)


def row(profile, brain, h, a, **kw):
    r = {"profile": profile, "brain": brain, "match": 0, "play_s": 60, "goals_home": 2, "goals_away": 1,
         "goal_diff": 1, "home_role_flips_per_min": h, "away_role_flips_per_min": a}
    r.update(kw)
    return r


def test_t_quantile_known_values():
    assert sb.t_quantile(0.975, 1) == pytest.approx(12.7062, abs=1e-3)
    assert sb.t_quantile(0.975, 9) == pytest.approx(2.2622, abs=1e-3)
    assert sb.t_quantile(0.975, 1000) == pytest.approx(1.9623, abs=1e-3)


def test_mean_ci():
    m, lo, hi = sb.mean_ci([1, 2, 3, 4, 5])
    assert m == 3
    h = 2.7764 * math.sqrt(2.5) / math.sqrt(5)
    assert (lo, hi) == (pytest.approx(3 - h, abs=1e-3), pytest.approx(3 + h, abs=1e-3))
    assert sb.mean_ci([7]) == (7, None, None)


def test_mwu_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    a = [1, 2, 2, 3, 5, 8, 8]
    b = [2, 4, 6, 8, 9, 9, 10, 12]
    ref = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue
    assert sb.mann_whitney_p(a, b) == pytest.approx(ref, abs=1e-6)
    assert sb.mann_whitney_p([1, 1, 1], [1, 1, 1]) == 1.0
    assert sb.mann_whitney_p([], [1]) is None
    assert sb.mann_whitney_p(list(range(20)), list(range(100, 120))) < 1e-5


def test_pooling_labels_pct_change_and_signal(tmp_path):
    cur = write(tmp_path / "c.csv", [
        row("good", "current", 4, 6, home_role_signal="player_role", away_role_signal="player_role"),
        row("good", "current", 8, 2, home_role_signal="player_role", away_role_signal="player_role")])
    base = write(tmp_path / "b.csv", [
        row("good", "baseline", 10, 10, home_role_signal="decision", away_role_signal="decision"),
        row("good", "baseline", 10, 10, home_role_signal="decision", away_role_signal="decision")])
    s = sb.build([cur, base], [], {"new": "n", "pre": "p"})
    r = {(x["brain"], x["metric"]): x for x in s["rows"] if x["profile"] == "good"}
    assert r[("new", "role_flips_per_min")]["n"] == 4          # 2 matches x 2 teams
    assert r[("new", "role_flips_per_min")]["mean"] == 5
    assert r[("new", "role_flips_per_min")]["median"] == 5
    assert r[("pre", "role_flips_per_min")]["mean"] == 10
    assert r[("new", "goals")]["n"] == 4 and r[("new", "goals")]["mean"] == 1.5
    assert r[("new", "goal_diff")]["mean"] == 0                # +1 home, -1 away
    t = [x for x in s["tests"] if x["metric"] == "role_flips_per_min"][0]
    assert t["pct_change"] == pytest.approx(-50)
    assert 0 <= t["p_mwu"] <= 1
    assert "player_role" in t["signal_note"] and "decision" in t["signal_note"]
    assert "role_flips_per_min" in s["meta"]["notes"]
    assert s["meta"]["n_matches"] == 4 and s["meta"]["brains"] == {"new": "n", "pre": "p"}


def test_pct_change_null_when_pre_zero_and_new_only_metric(tmp_path):
    cur = write(tmp_path / "c.csv", [row("good", "current", 3, 3, home_duplicate_lead_frac=0.1)])
    base = write(tmp_path / "b.csv", [row("good", "baseline", 0, 0)])
    s = sb.build([cur, base], [], {"new": "n", "pre": "p"})
    t = {x["metric"]: x for x in s["tests"]}
    assert t["role_flips_per_min"]["pct_change"] is None
    assert "duplicate_lead_frac" not in t                       # only one brain has it
    assert any(x["metric"] == "duplicate_lead_frac" and x["brain"] == "new" for x in s["rows"])


def test_missing_and_nan_cells_skipped(tmp_path):
    cur = write(tmp_path / "c.csv", [
        row("good", "current", "", "nan"),
        row("good", "current", 4, ""),
        row("good", "current", "inf", 6)])
    s = sb.build([cur], [], {"new": "n", "pre": "p"})
    r = [x for x in s["rows"] if x["metric"] == "role_flips_per_min"][0]
    assert r["n"] == 2 and r["mean"] == 5


def test_blackout_segments_separate_metrics(tmp_path):
    cur = write(tmp_path / "c.csv", [
        row("blackout:10@20", "current", 1, 1, home_bo_duplicate_striker_frac=0.4, away_bo_duplicate_striker_frac=0.2,
            home_out_duplicate_striker_frac=0.1, away_out_duplicate_striker_frac=0.1),
        row("good", "current", 1, 1)])
    s = sb.build([cur], [], {"new": "n", "pre": "p"})
    bo = [x for x in s["rows"] if x["metric"] == "bo_duplicate_striker_frac"]
    assert len(bo) == 1 and bo[0]["profile"] == "blackout:10@20" and bo[0]["mean"] == pytest.approx(0.3)
    assert any(x["metric"] == "out_duplicate_striker_frac" for x in s["rows"])
    assert not [x for x in s["rows"] if x["profile"] == "good" and x["metric"].startswith("bo_")]


def test_sweep_parsing_and_cli(tmp_path):
    rows = []
    for brain in ("current", "baseline"):
        for prof, v in (("lat50_loss15", 1), ("lat150_loss15", 2), ("lat300_loss15", 3),
                        ("lat150_loss0", 4), ("lat150_loss30", 5), ("lat50_loss30", 9), ("weird", 99)):
            rows.append(row(prof, brain, v, v + 2))
    sweep = write(tmp_path / "s.csv", rows)
    main = write(tmp_path / "m.csv", [row("good", "current", 1, 1), row("good", "baseline", 2, 2)])
    out, tidy = tmp_path / "o.json", tmp_path / "o.csv"
    assert sb.main(["--main", main, "--sweep", sweep, "--out", str(out), "--csv", str(tidy)]) == 0
    s = json.loads(out.read_text())
    assert s["meta"]["brains"] == {"new": "26b1193", "pre": "8d241a2"}
    sw = {(x["axis"], x["value"], x["brain"]): x for x in s["sweeps"] if x["metric"] == "role_flips_per_min"}
    assert sorted(v for a, v, b in sw if a == "latency_ms" and b == "new") == [50, 150, 300]
    assert sorted(v for a, v, b in sw if a == "loss_pct" and b == "new") == [0, 15, 30]
    assert sw[("latency_ms", 150, "new")]["mean"] == 3            # (2 + 4) / 2
    assert sw[("loss_pct", 30, "pre")]["mean"] == 6
    assert not any(x["value"] == 9 for x in s["sweeps"])           # lat50_loss30 is on neither axis
    assert "good" in s["meta"]["profiles"] and "lat50_loss15" not in s["meta"]["profiles"]
    assert tidy.read_text().splitlines()[0].startswith("kind,profile")


def _h2h_csv(path, rows):
    cols = ["profile", "brain", "goals_home", "goals_away", "home_brain", "away_brain", "home_profile", "away_profile"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_binomial_and_wilcoxon():
    assert sb.binom_two_sided_p(5, 5) == pytest.approx(0.0625)
    assert sb.binom_two_sided_p(2, 4) == pytest.approx(1.0)
    assert sb.binom_two_sided_p(0, 0) is None
    assert sb.wilcoxon_signed_rank_p([0, 0]) is None
    assert sb.wilcoxon_signed_rank_p([1, 2, 3, 4, 5, 6, 7, 8]) < 0.02
    assert sb.wilcoxon_signed_rank_p([1, -1, 2, -2]) == pytest.approx(1.0)


def test_h2h_home_perspective_and_swap_pooling(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    row = dict(profile="good", brain="current-vs-baseline", home_profile="good", away_profile="good")
    _h2h_csv(a, [{**row, "goals_home": 3, "goals_away": 1, "home_brain": "current", "away_brain": "baseline"},
                 {**row, "goals_home": 2, "goals_away": 2, "home_brain": "current", "away_brain": "baseline"}])
    _h2h_csv(b, [{**row, "goals_home": 0, "goals_away": 1, "home_brain": "baseline", "away_brain": "current"}])
    (r,) = sb.h2h([a], "home")
    assert (r["n"], r["wins"], r["draws"], r["losses"]) == (2, 1, 1, 0)
    assert r["goal_diff_mean"] == pytest.approx(1.0) and r["goals_for_mean"] == pytest.approx(2.5)
    assert r["p_sign"] == pytest.approx(1.0)
    (r,) = sb.h2h([a, b], "current")  # the swapped match counts for current: 1-0
    assert r["n"] == 3 and r["wins"] == 2 and r["goal_diff_mean"] == pytest.approx(1.0)
    assert "current(good) vs baseline(good)" in sb.fmt_h2h([r])


def test_h2h_cli_and_isolated_group(tmp_path, capsys):
    a = tmp_path / "a.csv"
    _h2h_csv(a, [dict(profile="good", brain="current", goals_home=g, goals_away=0, home_brain="current", away_brain="current",
                      home_profile="good", away_profile="blackout") for g in (1, 2, 3)])
    assert sb.main(["--h2h", str(a), "--out", str(tmp_path / "o.json")]) == 0
    assert "n=3" in capsys.readouterr().out
    assert json.loads((tmp_path / "o.json").read_text())["rows"][0]["opp_network"] == "blackout"
