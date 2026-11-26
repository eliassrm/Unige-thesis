"""
Data checks on DeepSense 6G scenarios 32 and 33, before any modelling:
  1. Each pass (seq_index) goes one way only: no U-turn or reversal inside it.
  2. How many passes go each way, per scenario (needed for a balanced split).
  3. Whether the two directions use different lanes.
  4. Whether the best-beam index moves the way GPS says the car moves.
  5. Basic integrity: required columns, time order, duplicated radio samples,
     gaps, unit1_beam vs argmax(power).

Run from the Code folder, either way:
    python tests/test_pass_properties.py              # report + pass/fail, writes CSVs
    python -m pytest tests/test_pass_properties.py -s

Data location: <Code>/../Data/scenario-XX, or set DEEPSENSE_DATA to the folder
that contains scenario-32 and scenario-33.

Variables used (columns of scenarioXX_dev.csv):
    index                        row id, 1-based, file order
    seq_index                    pass id: one recorded run of the car past the RSU
    time_stamp                   time of day, 'hh:mm:ss-microseconds'
    unit1_pwr_60ghz              path to the 64-beam power vector (linear, one value per line)
    unit1_beam                   best beam of the row, 1-based (1..64)
    unit1_max_pwr                power of that best beam (linear)
    unit1_loc                    path to the RSU GPS file: lat, lon (fixed)
    unit2_loc                    path to the vehicle GPS file: lat, lon
    unit2_interpolated_position  1 = vehicle position interpolated, 0 = measured fix
    unit2_spd_over_grnd_kmph, unit2_num_sats, unit2_hdop, ...  raw fix quality,
                                 filled only on rows with a real fix
Outputs: <Code>/outputs/data_checks/pass_properties.csv and scenario_summary.csv
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIOS = [32, 33]
CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DEEPSENSE_DATA", CODE_DIR.parent / "Data"))
OUT_DIR = CODE_DIR / "outputs" / "data_checks"

# Heuristic thresholds (metres / counts). Change here, not inside the code.
BACKTRACK_M = 5.0         # moving back this far against the pass direction = reversal
NET_FRACTION = 0.8        # |end - start| must cover this share of the travelled range
LANE_DIFF_M = 2.0         # median lateral offset difference suggesting different lanes
MIN_PASSES_PER_DIR = 3    # so each direction can appear in train, validation and test
GAP_FACTOR = 2.0          # time gap above this x median step = break inside a pass
SMOOTH_WIN = 5            # rolling-median window on along-road position (samples)
BEAM_MATCH_MIN = 0.95     # min share of rows where unit1_beam == argmax(power) + 1

REQUIRED = ["index", "seq_index", "time_stamp", "unit1_pwr_60ghz",
            "unit1_beam", "unit1_loc", "unit2_loc", "unit2_interpolated_position"]
R_EARTH = 6371000.0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _path(base, rel):
    rel = str(rel)
    if rel.startswith("./"):
        rel = rel[2:]
    return base / rel


def _read_vec(p):
    try:
        return np.loadtxt(p, dtype=float, ndmin=1)
    except (OSError, ValueError):
        return None


def _parse_time(ts):
    """'01:22:30-405771' -> seconds of day. Returns (seconds, micro_digits)."""
    hms, us = str(ts).split("-")
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s + int(us) / 1e6, len(us)


@lru_cache(maxsize=None)
def load_scenario(s):
    base = DATA_DIR / f"scenario-{s}"
    df = pd.read_csv(base / f"scenario{s}_dev.csv")
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        return dict(df=df, missing_cols=missing_cols)

    t, ndig = zip(*(_parse_time(v) for v in df["time_stamp"]))
    df["t_s"] = np.array(t)
    df["t_digits"] = np.array(ndig)

    veh = [_read_vec(_path(base, v)) for v in df["unit2_loc"]]
    df["lat"] = [v[0] if v is not None and v.size >= 2 else np.nan for v in veh]
    df["lon"] = [v[1] if v is not None and v.size >= 2 else np.nan for v in veh]
    rsu = _read_vec(_path(base, df["unit1_loc"].iloc[0]))

    P = [_read_vec(_path(base, v)) for v in df["unit1_pwr_60ghz"]]
    nb = max(p.size for p in P if p is not None)
    argmax1 = np.array([np.nanargmax(p) + 1 if p is not None and p.size == nb and np.isfinite(p).any()
                        else -1 for p in P])
    df["argmax_beam"] = argmax1
    df["pwr_hash"] = [hash(p.tobytes()) if p is not None else np.nan for p in P]

    # local metric frame centred on the RSU: x east, y north (metres)
    lat0, lon0 = rsu[0], rsu[1]
    df["x"] = np.radians(df["lon"] - lon0) * np.cos(np.radians(lat0)) * R_EARTH
    df["y"] = np.radians(df["lat"] - lat0) * R_EARTH

    # road axis = first principal direction of all vehicle positions
    xy = df[["x", "y"]].dropna().values
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    u = vt[0]
    if u[1] < 0 or (abs(u[1]) < 1e-9 and u[0] < 0):   # make +axis point north-ish
        u = -u
    n = np.array([-u[1], u[0]])                       # left of +axis
    df["along_m"] = (df[["x", "y"]].values - c) @ u
    df["lateral_m"] = (df[["x", "y"]].values - c) @ n
    rsu_lateral = float((np.zeros(2) - c) @ n)
    bearing = float(np.degrees(np.arctan2(u[0], u[1])) % 360)  # clockwise from north

    df = df.sort_values(["seq_index", "index"]).reset_index(drop=True)
    prev = df.groupby("seq_index")
    df["dup_time"] = prev["t_s"].diff().eq(0)
    df["dup_power"] = df["pwr_hash"].eq(prev["pwr_hash"].shift())
    return dict(df=df, missing_cols=[], n_beams=nb, axis_bearing_deg=bearing,
                rsu_lateral_m=rsu_lateral)


# --------------------------------------------------------------------------- #
# Per-pass properties
# --------------------------------------------------------------------------- #
def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


@lru_cache(maxsize=None)
def pass_table(s):
    S = load_scenario(s)
    df = S["df"]
    dt_all = df.groupby("seq_index")["t_s"].diff().dropna()
    med_dt = float(dt_all[dt_all > 0].median())
    rows = []
    for q, g in df.groupby("seq_index", sort=True):
        t = g["t_s"].values
        along = pd.Series(g["along_m"].values).rolling(SMOOTH_WIN, center=True, min_periods=1).median().values
        ok = np.isfinite(along)
        a = along[ok]
        k = min(3, len(a))
        start, end = np.median(a[:k]), np.median(a[-k:])
        net = end - start
        rng = a.max() - a.min()
        direction = "+axis" if net > 0 else "-axis"
        back = (np.maximum.accumulate(a) - a).max() if net > 0 else (a - np.minimum.accumulate(a)).max()
        dt = np.diff(t)
        beam = g["unit1_beam"].values.astype(float)
        rows.append(dict(
            scenario=s, seq_index=int(q), n_rows=len(g),
            duration_s=float(t[-1] - t[0]),
            time_ordered=bool((dt >= 0).all()),
            n_dup_time=int(g["dup_time"].sum()),
            n_dup_power=int(g["dup_power"].sum()),
            max_gap_s=float(dt.max()) if len(dt) else 0.0,
            n_gaps=int((dt > GAP_FACTOR * med_dt).sum()),
            gps_missing=int((~ok).sum()),
            interpolated_frac=float(g["unit2_interpolated_position"].fillna(0).astype(float).mean()),
            along_start_m=float(start), along_end_m=float(end),
            net_m=float(net), range_m=float(rng),
            direction=direction,
            backtrack_m=float(back),
            reversal=bool(back > BACKTRACK_M or abs(net) < NET_FRACTION * rng),
            lateral_median_m=float(np.nanmedian(g["lateral_m"].values)),
            lateral_iqr_m=float(np.subtract(*np.nanpercentile(g["lateral_m"].values, [75, 25]))),
            beam_start=float(np.median(beam[:k])), beam_end=float(np.median(beam[-k:])),
            beam_time_spearman=_spearman(beam, t),
            beam_matches_argmax=float((g["unit1_beam"].values == g["argmax_beam"].values).mean()),
        ))
    return pd.DataFrame(rows), med_dt


def scenario_summary(s):
    S = load_scenario(s)
    T, med_dt = pass_table(s)
    out = dict(scenario=s, n_rows=len(S["df"]), n_passes=len(T), n_beams=S["n_beams"],
               median_step_s=med_dt, road_axis_bearing_deg=S["axis_bearing_deg"],
               rsu_lateral_m=S["rsu_lateral_m"])
    seqs = sorted(T["seq_index"])
    out["missing_seq_ids"] = sorted(set(range(seqs[0], seqs[-1] + 1)) - set(seqs))
    for d in ["+axis", "-axis"]:
        Td = T[T["direction"] == d]
        out[f"passes_{d}"] = len(Td)
        out[f"lateral_{d}_m"] = float(Td["lateral_median_m"].median()) if len(Td) else np.nan
        sg = np.sign(Td["beam_time_spearman"].dropna())
        out[f"beam_trend_{d}"] = int(np.sign(sg.sum())) if len(sg) else 0
        out[f"beam_trend_agree_{d}"] = float((sg == np.sign(sg.sum())).mean()) if len(sg) else np.nan
    out["lane_diff_m"] = abs(out["lateral_+axis_m"] - out["lateral_-axis_m"])
    out["micro_digits_not_6"] = int((S["df"]["t_digits"] != 6).sum())
    out["beam_matches_argmax"] = float((S["df"]["unit1_beam"] == S["df"]["argmax_beam"]).mean())
    return out


# --------------------------------------------------------------------------- #
# Tests (pytest-compatible; also run by main())
# --------------------------------------------------------------------------- #
def test_required_columns():
    for s in SCENARIOS:
        assert not load_scenario(s)["missing_cols"], f"scenario {s}: missing {load_scenario(s)['missing_cols']}"


def test_time_never_goes_back():
    for s in SCENARIOS:
        T, _ = pass_table(s)
        bad = T.loc[~T["time_ordered"], "seq_index"].tolist()
        assert not bad, f"scenario {s}: time goes backwards in seq {bad}"


def test_no_duplicated_radio_samples():
    msgs = []
    for s in SCENARIOS:
        df = load_scenario(s)["df"]
        n_t, n_p = int(df["dup_time"].sum()), int(df["dup_power"].sum())
        both = int((df["dup_time"] & df["dup_power"]).sum())
        if n_t or n_p:
            msgs.append(f"scenario {s}: {n_t} rows repeat the previous timestamp, {n_p} repeat the previous "
                        f"power vector ({both} both) -> drop them before modelling "
                        f"(indices: {df.loc[df['dup_power'], 'index'].tolist()[:10]}...)")
    assert not msgs, "\n".join(msgs)


def test_no_reversal_within_pass():
    for s in SCENARIOS:
        T, _ = pass_table(s)
        bad = T.loc[T["reversal"], ["seq_index", "net_m", "range_m", "backtrack_m"]]
        assert bad.empty, f"scenario {s}: passes that reverse (cut them at the turn):\n{bad.to_string(index=False)}"


def test_both_directions_have_enough_passes():
    for s in SCENARIOS:
        T, _ = pass_table(s)
        c = T["direction"].value_counts().to_dict()
        for d in ["+axis", "-axis"]:
            assert c.get(d, 0) >= MIN_PASSES_PER_DIR, \
                f"scenario {s}: only {c.get(d, 0)} passes going {d} (need {MIN_PASSES_PER_DIR})"


def test_beam_trend_follows_direction():
    for s in SCENARIOS:
        m = scenario_summary(s)
        assert m["beam_trend_+axis"] != 0 and m["beam_trend_-axis"] != 0, f"scenario {s}: no clear beam trend"
        assert m["beam_trend_+axis"] == -m["beam_trend_-axis"], \
            f"scenario {s}: beam index moves the same way for both directions"


def test_beam_column_matches_power():
    for s in SCENARIOS:
        m = scenario_summary(s)
        assert m["beam_matches_argmax"] >= BEAM_MATCH_MIN, \
            f"scenario {s}: unit1_beam == argmax(power)+1 in only {m['beam_matches_argmax']:.1%} of rows"


TESTS = [test_required_columns, test_time_never_goes_back, test_no_duplicated_radio_samples,
         test_no_reversal_within_pass,
         test_both_directions_have_enough_passes, test_beam_trend_follows_direction,
         test_beam_column_matches_power]


def main():
    print(f"Data: {DATA_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables, sums = [], []
    for s in SCENARIOS:
        S = load_scenario(s)
        if S["missing_cols"]:
            print(f"scenario {s}: missing columns {S['missing_cols']}")
            continue
        T, _ = pass_table(s)
        m = scenario_summary(s)
        tables.append(T)
        sums.append(m)
        print(f"\n=== scenario {s} ===")
        print(f"rows {m['n_rows']}, passes {m['n_passes']}, beams {m['n_beams']}, "
              f"median step {m['median_step_s']*1000:.0f} ms, missing seq ids {m['missing_seq_ids']}")
        print(f"road axis bearing {m['road_axis_bearing_deg']:.0f} deg from north (+axis = north-ish); "
              f"RSU lateral offset {m['rsu_lateral_m']:.1f} m")
        print(f"passes +axis: {m['passes_+axis']}, -axis: {m['passes_-axis']}")
        print(f"median lateral offset +axis {m['lateral_+axis_m']:.1f} m, -axis {m['lateral_-axis_m']:.1f} m, "
              f"difference {m['lane_diff_m']:.1f} m -> "
              f"{'likely different lanes' if m['lane_diff_m'] > LANE_DIFF_M else 'no clear lane difference'}")
        print(f"beam index trend over time: +axis {m['beam_trend_+axis']:+d} "
              f"({m['beam_trend_agree_+axis']:.0%} of passes agree), "
              f"-axis {m['beam_trend_-axis']:+d} ({m['beam_trend_agree_-axis']:.0%} agree)")
        print(f"duplicated rows (same timestamp / same power as previous row): "
              f"{int(S['df']['dup_time'].sum())} / {int(S['df']['dup_power'].sum())}")
        print(f"unit1_beam == argmax(power)+1 in {m['beam_matches_argmax']:.1%} of rows; "
              f"timestamps with micro-digits != 6: {m['micro_digits_not_6']}")
        cols = ["seq_index", "n_rows", "duration_s", "direction", "net_m", "range_m", "backtrack_m",
                "reversal", "lateral_median_m", "beam_start", "beam_end", "n_gaps", "n_dup_power",
                "interpolated_frac"]
        print(T[cols].round(1).to_string(index=False))
    if tables:
        pd.concat(tables).to_csv(OUT_DIR / "pass_properties.csv", index=False)
        pd.DataFrame(sums).to_csv(OUT_DIR / "scenario_summary.csv", index=False)
        print(f"\nCSVs written to {OUT_DIR}")

    print("\n=== tests ===")
    failed = 0
    for f in TESTS:
        try:
            f()
            print(f"PASS  {f.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {f.__name__}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
