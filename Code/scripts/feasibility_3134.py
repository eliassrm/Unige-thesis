#!/usr/bin/env python3
"""
Feasibility check for best-beam selection on DeepSense 6G scenarios 31-34.

What it answers, per scenario:
  1. Data inventory   - samples, passes, samples per pass, whether the CSV is a
                        full continuous recording or a challenge window split.
  2. Beam dynamics    - how often the best beam changes, how far it jumps.
  3. Staleness cost   - dB lost if you hold a beam for k steps without checking.
                        (Is there a decision problem at all?)
  4. Cheap baselines  - full sweep, periodic sweep, local +-w tracking,
                        threshold-triggered sweep, GPS kNN lookup.
                        Loss vs. oracle and probing overhead.
                        (Is there headroom left for an active-inference policy?)
  5. Blockage events  - deep drops of the best-beam power, and whether switching
                        to another beam during the event actually helps.
                        Saves a before/during camera frame pair per event.

Usage (from the repo root):
  python Code/scripts/feasibility_3134.py ^
      --scenario-dirs Data/scenario-31 Data/scenario-32 Data/scenario-33 Data/scenario-34 ^
      --out Code/outputs/feasibility

Each scenario dir must contain the scenario CSV (e.g. scenario32.csv) and the
folders it references (unit1/mmWave_data/..., unit2/GPS_data/..., ...).
Paths inside the CSV are resolved relative to the scenario dir.

All thresholds are CLI arguments; the defaults are heuristics, not standards.

Known limits:
  - Events are found on the LOS-trend beam (rolling median of the best-beam
    index, window --med-win). A blockage in the first/last med_win/2 samples
    of a pass is not detected, and when the beam drifts fast the event window
    can run a few samples past the real blockage. Always confirm events on the
    camera frames listed in <scenario>_events.csv.
  - Noise floor defaults to the 5th percentile of all beam powers; pass the
    real value with --noise-floor-db once you have it. The DeepSense 31-34
    power only spans ~7-8 dB, so the event and useful-switch thresholds
    (--drop-db, --min-normal-db, --snr-min-db, --useful-gain-db) are set to a
    few dB, not the 6-10 dB you would use on calibrated power.
  - Overhead = beams measured per step; the serving beam is measured for free.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_cols(cols, include, exclude=()):
    out = []
    for c in cols:
        lc = c.lower()
        if all(s in lc for s in include) and not any(s in lc for s in exclude):
            out.append(c)
    return out


def pick_csv(d, csv_name):
    if csv_name:
        p = os.path.join(d, csv_name)
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        return p
    csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"no CSV in {d}")

    def rank(p):
        b = os.path.basename(p).lower()
        split = any(s in b for s in ("dev", "test", "train", "val", "challenge"))
        return (split, len(b))

    return sorted(csvs, key=rank)[0]


def resolve(d, rel):
    rel = str(rel).strip().strip("'\"[]")
    rel = rel.replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    p = os.path.join(d, rel)
    if os.path.isfile(p):
        return p
    # some CSVs carry an extra leading folder; try dropping it
    parts = rel.split("/")
    for i in range(1, len(parts)):
        q = os.path.join(d, *parts[i:])
        if os.path.isfile(q):
            return q
    return None


def load_vector(d, val):
    """A cell is either a path to a txt file, or an inline number."""
    try:
        return np.atleast_1d(float(val))
    except (TypeError, ValueError):
        pass
    p = resolve(d, val)
    if p is None:
        return None
    try:
        return np.atleast_1d(np.loadtxt(p, dtype=float, delimiter=None))
    except Exception:
        try:
            return np.atleast_1d(np.loadtxt(p, dtype=float, delimiter=","))
        except Exception:
            return None


def load_scenario(d, csv_name=None, verbose=True):
    path = pick_csv(d, csv_name)
    df = pd.read_csv(path)
    cols = list(df.columns)
    warn = []

    pwr_cols = find_cols(cols, ["pwr"], exclude=("max",))
    if not pwr_cols:
        raise ValueError(f"{path}: no power column (expected something like unit1_pwr_60ghz)")
    if len(pwr_cols) > 1:
        warn.append(
            f"{len(pwr_cols)} power columns ({pwr_cols[:3]}...): this looks like a challenge "
            "window split, not a continuous recording. Use the full scenarioXX.csv for "
            "pass-level analysis; only the first power column is used here."
        )
    c_pwr = pwr_cols[0]
    c_seq = (find_cols(cols, ["seq"]) or [None])[0]
    c_time = (find_cols(cols, ["time"]) or [None])[0]
    c_idx = (find_cols(cols, ["index"], exclude=("seq",)) or [None])[0]
    gps_cols = find_cols(cols, ["unit2", "loc"]) or find_cols(cols, ["unit2", "gps"])
    c_gps = gps_cols[0] if gps_cols else None
    c_rgb = (find_cols(cols, ["rgb"]) or [None])[0]
    c_lidar = (find_cols(cols, ["lidar"], exclude=("scr",)) or [None])[0]

    P, keep, missing = [], [], 0
    for i, v in enumerate(df[c_pwr].values):
        vec = load_vector(d, v)
        if vec is None or vec.size < 2:
            missing += 1
            continue
        P.append(vec)
        keep.append(i)
    if not P:
        raise ValueError(f"{path}: none of the power files could be read (check paths)")
    nb = int(np.median([p.size for p in P]))
    ok = [k for k, p in zip(keep, P) if p.size == nb]
    P = np.vstack([p for p in P if p.size == nb])
    df = df.iloc[ok].reset_index(drop=True)
    if missing:
        warn.append(f"{missing} power files missing/unreadable and dropped")
    # some files have a few NaN beams; fill them from neighbouring beams so
    # argmax and dB losses stay valid without breaking the time order of a pass
    nan_rows = np.where(np.isnan(P).any(1))[0]
    for r in nan_rows:
        m = np.isnan(P[r])
        P[r, m] = np.interp(np.where(m)[0], np.where(~m)[0], P[r, ~m])
    if len(nan_rows):
        warn.append(f"{len(nan_rows)} samples had NaN beam powers; filled by interpolating neighbouring beams")

    gps = None
    if c_gps is not None:
        g = [load_vector(d, v) for v in df[c_gps].values]
        if all(x is not None and x.size >= 2 for x in g):
            gps = np.vstack([x[:2] for x in g])
        else:
            warn.append("GPS column found but not all entries readable; GPS baseline skipped")

    # pass segmentation
    if c_seq is not None:
        seq = pd.to_numeric(df[c_seq], errors="coerce").ffill().fillna(0).astype(int).values
    else:
        warn.append("no seq_index column: treating the file as ONE pass (pass statistics unreliable)")
        seq = np.zeros(len(df), dtype=int)

    order_key = None
    if c_idx is not None:
        order_key = pd.to_numeric(df[c_idx], errors="coerce").values
    if order_key is None or np.isnan(order_key).all():
        order_key = np.arange(len(df))
    order = np.lexsort((order_key, seq))
    df, P, seq = df.iloc[order].reset_index(drop=True), P[order], seq[order]
    if gps is not None:
        gps = gps[order]

    info = dict(
        csv=os.path.basename(path), power_col=c_pwr, seq_col=c_seq, time_col=c_time,
        gps_col=c_gps, rgb_col=c_rgb, lidar_col=c_lidar, n_beams=nb,
    )
    if verbose:
        print(f"  csv: {info['csv']}  | columns used: {info}")
        for w in warn:
            print(f"  WARNING: {w}")
    return dict(df=df, P=P, seq=seq, gps=gps, info=info, warnings=warn,
                rgb=(df[c_rgb].values if c_rgb else None))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def to_db(P):
    return 10 * np.log10(np.maximum(P, EPS))


def passes(seq):
    idx = []
    for s in np.unique(seq):
        ii = np.where(seq == s)[0]
        idx.append(ii)
    return idx


def beam_dynamics(Pdb, pass_idx):
    best = Pdb.argmax(1)
    changes, jumps, runs, n = 0, [], [], 0
    for ii in pass_idx:
        b = best[ii]
        n += max(len(b) - 1, 0)
        d = np.diff(b)
        ch = d != 0
        changes += ch.sum()
        jumps += list(np.abs(d[ch]))
        # run lengths
        r = 1
        for k in range(1, len(b)):
            if b[k] == b[k - 1]:
                r += 1
            else:
                runs.append(r)
                r = 1
        runs.append(r)
    jumps = np.array(jumps) if jumps else np.array([0])
    return {
        "best_beam_change_rate": changes / max(n, 1),
        "median_run_len_samples": float(np.median(runs)) if runs else np.nan,
        "jump_within_1": float((jumps <= 1).mean()),
        "jump_within_2": float((jumps <= 2).mean()),
        "jump_within_4": float((jumps <= 4).mean()),
        "jump_p95": float(np.percentile(jumps, 95)),
    }


def staleness(Pdb, pass_idx, ks=(1, 2, 5, 10, 20)):
    out = {}
    for k in ks:
        losses = []
        for ii in pass_idx:
            X = Pdb[ii]
            b = X.argmax(1)
            bestp = X.max(1)
            for t in range(len(X) - k):
                losses.append(bestp[t + k] - X[t + k, b[t]])
        losses = np.array(losses) if losses else np.array([np.nan])
        out[f"hold{k}_mean_loss_dB"] = float(np.nanmean(losses))
        out[f"hold{k}_frac_gt3dB"] = float(np.nanmean(losses > 3))
    return out


# ---- policies: each returns (chosen beam per t, beams measured per t) ------ #
def pol_full(X):
    return X.argmax(1), np.full(len(X), X.shape[1])


def pol_periodic(X, T):
    nb = X.shape[1]
    ch, pr = np.zeros(len(X), int), np.zeros(len(X), int)
    cur = 0
    for t in range(len(X)):
        if t % T == 0:
            cur = int(X[t].argmax()); pr[t] = nb
        ch[t] = cur
    return ch, pr


def pol_local(X, w):
    nb = X.shape[1]
    ch, pr = np.zeros(len(X), int), np.zeros(len(X), int)
    cur = int(X[0].argmax()); pr[0] = nb; ch[0] = cur
    for t in range(1, len(X)):
        lo, hi = max(cur - w, 0), min(cur + w, nb - 1)
        cand = np.arange(lo, hi + 1)
        cur = int(cand[X[t, cand].argmax()])
        ch[t] = cur
        pr[t] = len(cand) - 1  # serving beam is measured for free
    return ch, pr


def pol_threshold(X, d_db):
    nb = X.shape[1]
    ch, pr = np.zeros(len(X), int), np.zeros(len(X), int)
    cur = int(X[0].argmax()); ref = X[0, cur]; pr[0] = nb; ch[0] = cur
    for t in range(1, len(X)):
        if X[t, cur] < ref - d_db:
            cur = int(X[t].argmax()); ref = X[t, cur]; pr[t] = nb
        ch[t] = cur
    return ch, pr


def gps_knn_predict(gps, best, pass_idx, k=5, topn=1):
    """Leave-one-pass-out kNN from user GPS to best beam."""
    pred = np.full((len(best), topn), -1, int)
    mu, sd = gps.mean(0), gps.std(0) + EPS
    G = (gps - mu) / sd
    for ii in pass_idx:
        train = np.setdiff1d(np.arange(len(best)), ii)
        if len(train) < k:
            continue
        for t in ii:
            dist = np.sum((G[train] - G[t]) ** 2, 1)
            nn = train[np.argsort(dist)[:k]]
            votes = np.bincount(best[nn], minlength=best.max() + 1)
            pred[t] = np.argsort(-votes)[:topn]
    return pred


def evaluate_policies(Pdb, pass_idx, gps=None, args=None):
    rows = []
    policies = [("full_sweep", lambda X: pol_full(X))]
    for T in args.periods:
        policies.append((f"periodic_T{T}", lambda X, T=T: pol_periodic(X, T)))
    for w in args.local_w:
        policies.append((f"local_pm{w}", lambda X, w=w: pol_local(X, w)))
    for d in args.thresholds:
        policies.append((f"threshold_{d}dB", lambda X, d=d: pol_threshold(X, d)))

    best_all = Pdb.argmax(1)
    for name, fn in policies:
        loss, probes, hit = [], [], []
        for ii in pass_idx:
            X = Pdb[ii]
            ch, pr = fn(X)
            l = X.max(1) - X[np.arange(len(X)), ch]
            loss.append(l); probes.append(pr); hit.append(ch == X.argmax(1))
        rows.append(_summ(name, np.concatenate(loss), np.concatenate(probes), np.concatenate(hit)))

    if gps is not None and len(pass_idx) > 1:
        for topn in (1, 3):
            pred = gps_knn_predict(gps, best_all, pass_idx, topn=topn)
            valid = pred[:, 0] >= 0
            if valid.sum() == 0:
                continue
            ch = np.array([p[np.argmax(Pdb[t, p])] if topn > 1 else p[0]
                           for t, p in enumerate(pred)])
            ch = np.where(valid, ch, best_all)
            l = Pdb.max(1) - Pdb[np.arange(len(Pdb)), ch]
            probes = np.full(len(Pdb), 0 if topn == 1 else topn)
            rows.append(_summ(f"gps_knn_top{topn}", l[valid], probes[valid], (ch == best_all)[valid]))
    return pd.DataFrame(rows)


def _summ(name, loss, probes, hit):
    return {
        "policy": name,
        "mean_loss_dB": float(np.mean(loss)),
        "p95_loss_dB": float(np.percentile(loss, 95)),
        "frac_loss_gt3dB": float(np.mean(loss > 3)),
        "top1_acc": float(np.mean(hit)),
        "mean_beams_measured_per_step": float(np.mean(probes)),
    }


# ---- blockage / deep-fade events ------------------------------------------ #
def find_events(Pdb, pass_idx, seq, rgb, floor_db, args):
    """
    An event = the power on the *LOS-trend beam* drops well below the local
    typical best power. The LOS-trend beam is a rolling median of the best-beam
    index, so it follows the car's normal drift but ignores short jumps to a
    reflected path. This catches blockages whether or not a backup path exists.
    A drop only counts where the normal level is itself clearly above the noise
    floor; otherwise it is just noise around the floor.
    """
    ev = []
    nb = Pdb.shape[1]
    for p_no, ii in enumerate(pass_idx):
        X = Pdb[ii]
        best_idx = pd.Series(X.argmax(1).astype(float))
        # full window only: a truncated window at the pass edges lags the drift
        # and would create fake events, so the edges fall back to the raw best beam
        # (a blockage in the first/last med_win/2 samples of a pass is not detected)
        track = best_idx.rolling(args.med_win, center=True, min_periods=args.med_win).median()
        track = track.fillna(best_idx)
        track = np.clip(np.round(track.values).astype(int), 0, nb - 1)
        on_track = X[np.arange(len(X)), track]
        bp = pd.Series(X.max(1))
        med = bp.rolling(args.med_win, center=True, min_periods=3).median().values
        low = (on_track < med - args.drop_db) & (med > floor_db + args.min_normal_db)
        t = 0
        while t < len(low):
            if not low[t]:
                t += 1
                continue
            s = t
            while t < len(low) and low[t]:
                t += 1
            e = t  # exclusive
            if e - s < args.min_len:
                continue
            pre = max(s - 1, 0)
            pre_beam = int(track[pre])
            pre_pow = float(X[pre].max())
            seg = X[s:e]
            seg_track = track[s:e]
            # best you get by tracking the LOS beam (+- far_beams around the trend)
            near = np.array([seg[k, max(b - args.far_beams, 0):min(b + args.far_beams, nb - 1) + 1].max()
                             for k, b in enumerate(seg_track)])
            # best you get from a beam clearly off the LOS trend (reflection / NLOS)
            far = np.array([seg[k, np.abs(np.arange(nb) - b) > args.far_beams].max()
                            for k, b in enumerate(seg_track)])
            stay = seg[np.arange(len(seg)), seg_track]
            gain_far = float(np.mean(far - near))
            gain_far_vs_stay = float(np.mean(far - stay))
            gain_any = float(np.mean(seg.max(1) - stay))
            above_floor = float(np.mean(far >= floor_db + args.snr_min_db))
            mid = ii[(s + e) // 2]
            before = ii[max(s - args.before_gap, 0)]
            ev.append({
                "pass": p_no, "seq_index": int(seq[ii[s]]),
                "row_start": int(ii[s]), "row_end": int(ii[e - 1]),
                "len_samples": e - s,
                "depth_dB": float(np.max(med[s:e] - on_track[s:e])),
                "normal_above_floor_dB": float(np.min(med[s:e]) - floor_db),
                "pre_beam": pre_beam, "pre_power_dB": pre_pow,
                "switch_gain_any_dB": gain_any,
                f"switch_gain_far(>{args.far_beams})_vs_near_dB": gain_far,
                f"switch_gain_far(>{args.far_beams})_vs_stay_dB": gain_far_vs_stay,
                f"frac_far_above_floor+{args.snr_min_db:g}dB": above_floor,
                "useful_switch": bool(gain_far_vs_stay >= args.useful_gain_db and above_floor >= 0.5),
                "camera_frame_before": (str(rgb[before]) if rgb is not None else ""),
                "camera_frame_mid": (str(rgb[mid]) if rgb is not None else ""),
            })
    return pd.DataFrame(ev)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_passes(name, Pdb, pass_idx, events, out, max_passes=6):
    n = min(max_passes, len(pass_idx))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    # the power only spans a few dB, so scale to the data's own range
    vmin, vmax = float(Pdb.min()), float(Pdb.max())
    for a, ii in zip(axes[:, 0], pass_idx[:n]):
        X = Pdb[ii]
        im = a.imshow(X.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        fig.colorbar(im, ax=a, pad=0.01, label="dB")
        a.plot(X.argmax(1), color="white", lw=0.8)
        if len(events):
            for _, r in events.iterrows():
                if r["row_start"] >= ii[0] and r["row_end"] <= ii[-1]:
                    s = np.searchsorted(ii, r["row_start"]); e = np.searchsorted(ii, r["row_end"])
                    a.axvspan(s, e, color="red", alpha=0.25)
        a.set_ylabel("beam")
    axes[-1, 0].set_xlabel("sample within pass")
    fig.suptitle(f"{name}: per-beam power (dB), white = best beam, red = candidate event")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_passes.png"), dpi=120)
    plt.close(fig)


def save_event_frames(name, d, events, out):
    """One before/during camera pair per event, for visual confirmation."""
    if not len(events):
        return []
    odir = os.path.join(out, "event_frames", name)
    os.makedirs(odir, exist_ok=True)
    paths = []
    for _, r in events.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for a, col, label in zip(axes, ("camera_frame_before", "camera_frame_mid"), ("before", "during")):
            p = resolve(d, r[col])
            if p is not None:
                a.imshow(plt.imread(p))
            a.set_title(f"{label}: {os.path.basename(str(r[col]))}", fontsize=9)
            a.axis("off")
        fig.suptitle(f"{name} seq {r['seq_index']} rows {r['row_start']}-{r['row_end']} "
                     f"({r['len_samples']} samples, depth {r['depth_dB']:.1f} dB)", fontsize=10)
        fig.tight_layout()
        p_out = os.path.join(odir, f"seq{r['seq_index']}_rows{r['row_start']}-{r['row_end']}.png")
        fig.savefig(p_out, dpi=90)
        plt.close(fig)
        paths.append(os.path.relpath(p_out, out))
    return paths


def plot_policies(name, pol, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(pol["mean_beams_measured_per_step"], pol["mean_loss_dB"])
    for _, r in pol.iterrows():
        ax.annotate(r["policy"], (r["mean_beams_measured_per_step"], r["mean_loss_dB"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("beams measured per step (overhead)")
    ax.set_ylabel("mean power loss vs oracle (dB)")
    ax.set_title(f"{name}: loss vs overhead")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"{name}_policies.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario-dirs", nargs="+", required=True)
    ap.add_argument("--csv-name", default=None, help="force a CSV file name inside each dir")
    ap.add_argument("--out", default="feasibility_results")
    ap.add_argument("--noise-floor-db", type=float, default=None,
                    help="noise floor in the same dB scale as 10log10(power). "
                         "Default: 5th percentile of all beam powers in the scenario.")
    ap.add_argument("--periods", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--local-w", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[3.0, 6.0])
    ap.add_argument("--drop-db", type=float, default=3.0, help="event: best power below rolling median by this")
    ap.add_argument("--min-normal-db", type=float, default=3.0,
                    help="event: only count a drop where the normal (rolling median) level is "
                         "more than this above the noise floor")
    ap.add_argument("--med-win", type=int, default=21)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--far-beams", type=int, default=2, help="'far' alternative beam = more than this many indices away")
    ap.add_argument("--snr-min-db", type=float, default=1.0,
                    help="useful switch: the far beam must be at least this far above the noise floor")
    ap.add_argument("--useful-gain-db", type=float, default=3.0,
                    help="useful switch: the far beam must beat staying on the LOS-trend beam by this")
    ap.add_argument("--before-gap", type=int, default=5,
                    help="'before' camera frame = this many samples before the event start")
    ap.add_argument("--cheap-max-beams", type=float, default=5.0,
                    help="a policy measuring <= this many beams/step counts as 'cheap'")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = []
    for d in args.scenario_dirs:
        name = os.path.basename(os.path.normpath(d))
        print(f"\n=== {name} ===")
        try:
            S = load_scenario(d, args.csv_name)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        Pdb = to_db(S["P"])
        pidx = passes(S["seq"])
        floor_db = args.noise_floor_db if args.noise_floor_db is not None else float(np.percentile(Pdb, 5))
        lens = [len(i) for i in pidx]

        dyn = beam_dynamics(Pdb, pidx)
        st = staleness(Pdb, pidx)
        pol = evaluate_policies(Pdb, pidx, S["gps"], args)
        ev = find_events(Pdb, pidx, S["seq"], S["rgb"], floor_db, args)
        if len(ev):
            ev["frames_png"] = save_event_frames(name, d, ev, args.out)

        pol.to_csv(os.path.join(args.out, f"{name}_policies.csv"), index=False)
        ev.to_csv(os.path.join(args.out, f"{name}_events.csv"), index=False)
        plot_passes(name, Pdb, pidx, ev, args.out)
        plot_policies(name, pol, args.out)

        cheap = pol[pol["mean_beams_measured_per_step"] <= args.cheap_max_beams]
        best_cheap = cheap.sort_values("mean_loss_dB").iloc[0] if len(cheap) else None
        n_useful = int(ev["useful_switch"].sum()) if len(ev) else 0

        row = {
            "scenario": name, "csv": S["info"]["csv"], "n_samples": len(Pdb), "n_beams": S["info"]["n_beams"],
            "n_passes": len(pidx), "median_samples_per_pass": float(np.median(lens)),
            "noise_floor_dB_used": floor_db, "gps_available": S["gps"] is not None,
            "lidar_column": S["info"]["lidar_col"] or "",
            **dyn, **st,
            "best_cheap_policy": best_cheap["policy"] if best_cheap is not None else "",
            "best_cheap_loss_dB": float(best_cheap["mean_loss_dB"]) if best_cheap is not None else np.nan,
            "best_cheap_top1": float(best_cheap["top1_acc"]) if best_cheap is not None else np.nan,
            "best_cheap_beams_per_step": float(best_cheap["mean_beams_measured_per_step"]) if best_cheap is not None else np.nan,
            "n_events": len(ev), "n_events_useful_switch": n_useful,
            "warnings": " | ".join(S["warnings"]),
        }
        summary.append(row)

        print(f"  samples={row['n_samples']} passes={row['n_passes']} "
              f"median/pass={row['median_samples_per_pass']:.0f} beams={row['n_beams']}")
        print(f"  best-beam change rate={dyn['best_beam_change_rate']:.2f}/step, "
              f"jumps within +-2: {dyn['jump_within_2']:.0%}")
        print(f"  hold 5 steps: mean loss {st['hold5_mean_loss_dB']:.2f} dB, "
              f">3 dB in {st['hold5_frac_gt3dB']:.0%} of steps")
        print(pol.round(3).to_string(index=False))
        print(f"  events={len(ev)}, with useful far switch={n_useful}")

    if not summary:
        print("\nNothing processed.")
        sys.exit(1)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # ------------------------------------------------------------------ verdict
    print("\n=================  FEASIBILITY VERDICT (heuristic)  =================")
    tot_passes = sm["n_passes"].sum()
    print(f"[data]      {len(sm)} scenarios, {sm['n_samples'].sum()} samples, {tot_passes} passes.")
    if (sm["warnings"].str.contains("challenge window")).any():
        print("            -> At least one CSV is a challenge split. Re-run on the full scenario CSVs.")
    drift = sm["hold5_mean_loss_dB"].mean()
    print(f"[problem]   Holding a beam 5 steps costs {drift:.2f} dB on average "
          f"({sm['hold5_frac_gt3dB'].mean():.0%} of steps > 3 dB).")
    print("            -> " + ("Beam drift is real: a keep/check decision exists every step."
                               if drift > 1.0 else
                               "Beam barely drifts: the decision problem is weak at this time scale."))
    hc = sm["best_cheap_loss_dB"].mean(); ht = sm["best_cheap_top1"].mean()
    print(f"[headroom]  Best cheap policy (<= {args.cheap_max_beams:g} beams/step): "
          f"{hc:.2f} dB loss, top-1 {ht:.0%}.")
    print("            -> " + ("Small headroom: a smarter policy can only win on overhead or on rare events. "
                               "Frame the gain as overhead saved at equal loss."
                               if hc < 0.5 else
                               "Room for a better policy on loss/overhead trade-off."))
    nu = int(sm["n_events_useful_switch"].sum()); ne = int(sm["n_events"].sum())
    print(f"[blockage]  {ne} candidate events, {nu} where a far beam helps by >= {args.useful_gain_db:g} dB.")
    print("            -> " + ("Enough to evaluate blockage-time switching as a sub-result."
                               if nu >= 10 else
                               "Too few for a blockage-switching claim: treat blockage as the anomaly "
                               "case where the right action is often NOT to sweep."))
    print("Check every event visually with camera_frame_mid in <scenario>_events.csv.")
    print(f"\nOutputs in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
