"""
Plots of the DeepSense 6G passes for scenarios 32 and 33: GPS tracks and best beam.

Run from the Code folder:
    python scripts/plot_passes.py
Data location: <Code>/../Data, or set DEEPSENSE_DATA (same as the data check).
Figures: <Code>/outputs/data_checks/figures/

Per scenario it draws one figure with four panels:
  (a) GPS tracks seen from above, metres around the RSU, coloured by direction
  (b) best beam (unit1_beam) against distance north of the RSU, one line per pass
  (c) lane per pass: distance east of the RSU on the straight part of the road
      (the road turns about 50 m north of the RSU; the turn is left out here)
  (d) all 64 beam powers over time for one northbound and one southbound pass
Duplicated rows (same timestamp and power as the previous row) are left out.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR / "tests"))
import test_pass_properties as dc  # loader and per-pass table from the data check

FIG_DIR = dc.OUT_DIR / "figures"
COL = {"+axis": "tab:blue", "-axis": "tab:orange"}
NAME = {"+axis": "northbound", "-axis": "southbound"}
STRAIGHT_Y = (5.0, 40.0)   # metres north of the RSU counted as the straight part (lane panel)


def power_matrix(s, g):
    """64 x T matrix of beam power in dB for the rows of one pass."""
    base = dc.DATA_DIR / f"scenario-{s}"
    P = np.vstack([dc._read_vec(dc._path(base, v)) for v in g["unit1_pwr_60ghz"]])
    return 10 * np.log10(np.maximum(P, 1e-12)).T


def plot_scenario(s):
    S = dc.load_scenario(s)
    df = S["df"]
    df = df[~df["dup_power"]]
    T, _ = dc.pass_table(s)
    dirs = dict(zip(T["seq_index"], T["direction"]))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_gps = fig.add_subplot(gs[0, 0])
    ax_beam = fig.add_subplot(gs[0, 1:])
    ax_lane = fig.add_subplot(gs[1, 0])
    ax_h = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    # (a) GPS tracks, top view
    for q, g in df.groupby("seq_index"):
        d = dirs[q]
        ax_gps.plot(g["x"], g["y"], color=COL[d], lw=1, alpha=0.7)
        ax_gps.annotate("", xy=(g["x"].iloc[-1], g["y"].iloc[-1]),
                        xytext=(g["x"].iloc[-6], g["y"].iloc[-6]),
                        arrowprops=dict(arrowstyle="->", color=COL[d], lw=1))
    ax_gps.plot(0, 0, marker="^", color="k", ms=10, ls="none", label="RSU")
    for d in COL:
        ax_gps.plot([], [], color=COL[d], label=f"{NAME[d]} ({(T['direction'] == d).sum()} passes)")
    ax_gps.set_aspect("equal", adjustable="datalim")
    ax_gps.set_xlabel("east of RSU (m)")
    ax_gps.set_ylabel("north of RSU (m)")
    ax_gps.set_title("(a) GPS tracks, seen from above")
    ax_gps.legend(fontsize=8, loc="best")
    ax_gps.grid(alpha=0.3)

    # (b) best beam along the road
    for q, g in df.groupby("seq_index"):
        ax_beam.plot(g["y"], g["unit1_beam"], color=COL[dirs[q]], lw=0.8, alpha=0.6)
    for d in COL:
        ax_beam.plot([], [], color=COL[d], label=NAME[d])
    ax_beam.set_xlabel("north of RSU (m), same axis as the y axis of (a)")
    ax_beam.set_ylabel("best beam (unit1_beam, 1-64)")
    ax_beam.set_ylim(0, 65)
    ax_beam.set_title("(b) Best beam vs position: noisy beyond ~40 m (road turn, weak signal)")
    ax_beam.legend(fontsize=8)
    ax_beam.grid(alpha=0.3)

    # (c) lane: lateral position per pass
    Ts = T.sort_values(["direction", "seq_index"])
    for i, r in enumerate(Ts.itertuples()):
        g = df[(df["seq_index"] == r.seq_index) & df["y"].between(*STRAIGHT_Y)]["x"]
        ax_lane.boxplot(g.dropna(), positions=[i], widths=0.6, showfliers=False,
                        patch_artist=True, boxprops=dict(facecolor=COL[r.direction], alpha=0.5),
                        medianprops=dict(color="k"))
    ax_lane.set_xticks(range(len(Ts)))
    ax_lane.set_xticklabels(Ts["seq_index"], fontsize=7)
    ax_lane.set_xlabel("pass (seq_index), grouped by direction")
    ax_lane.set_ylabel(f"east of RSU on straight part, {STRAIGHT_Y[0]:.0f}-{STRAIGHT_Y[1]:.0f} m north (m)")
    ax_lane.set_title("(c) Lane per pass (smaller = closer to RSU)")
    ax_lane.grid(alpha=0.3, axis="y")

    # (d) 64-beam power heatmaps: the longest pass of each direction
    for ax, d in zip(ax_h, ["+axis", "-axis"]):
        q = int(T[T["direction"] == d].sort_values("n_rows").iloc[-1]["seq_index"])
        g = df[df["seq_index"] == q]
        M = power_matrix(s, g)
        t = g["t_s"].values - g["t_s"].values[0]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       extent=[t[0], t[-1], 0.5, M.shape[0] + 0.5])
        ax.plot(t, g["unit1_beam"], color="w", lw=0.8, label="best beam")
        ax.set_xlabel("time in pass (s)")
        ax.set_ylabel("beam index")
        ax.set_title(f"(d) Beam power, {NAME[d]} pass {q}")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, label="power (dB, uncalibrated)")

    fig.suptitle(f"DeepSense 6G scenario {s}: {len(T)} passes, road bearing "
                 f"{S['axis_bearing_deg']:.0f} deg from north", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"scenario{s}_passes_gps_beam.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    for s in dc.SCENARIOS:
        print("saved", plot_scenario(s))


if __name__ == "__main__":
    main()
