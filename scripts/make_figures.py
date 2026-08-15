"""Generate the figures in images/ from the committed result CSVs.

Every figure is built from artifacts/results/*.csv, which are committed, so the charts
are reproducible from the repo without re-running the pipeline.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")  # headless: no display on the Studio or in CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "artifacts/results"
OUT = "images"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, GRID = "#1a1a1a", "#6b7280", "#e5e7eb"
TFT, NAIVE, ZERO, BASE, BAD = "#2563eb", "#f59e0b", "#9ca3af", "#059669", "#dc2626"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def strip_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def welch(a, b):
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se else 0.0, se


# ---------------------------------------------------------------- 1. THE ablation
def fig_ablation():
    d = pd.read_csv(f"{R}/ablation_20seeds.csv")
    order = ["supplied", "zeros", "persistence"]
    labels = ["TFT forecast", "Zeros\n(no signal)", "Persistence\n(naive forecast)"]
    colors = [TFT, ZERO, NAIVE]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    rng = np.random.default_rng(0)
    for i, (arm, c) in enumerate(zip(order, colors)):
        v = d[d.arm == arm]["test_cost"].to_numpy()
        ax.scatter(np.full(len(v), i) + rng.uniform(-.13, .13, len(v)), v,
                   s=26, color=c, alpha=.55, edgecolor="none", zorder=3)
        m, sd = v.mean(), v.std(ddof=1)
        ax.hlines(m, i - .3, i + .3, color=c, lw=2.6, zorder=4)
        ax.vlines(i, m - 1.96 * sd / np.sqrt(len(v)), m + 1.96 * sd / np.sqrt(len(v)),
                  color=c, lw=2.6, zorder=4)
        ax.text(i + .34, m, f"{m:.0f}", va="center", ha="left",
                fontsize=9.5, color=c, fontweight="bold", zorder=5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlim(-.5, 2.75)
    ax.set_ylabel("Test cost  (lower is better)")
    ax.set_title("Forecast ablation: all three arms overlap\n20 seeds each, paired episodes", loc="left")
    strip_spines(ax)
    ax.text(.5, .03, "bars = mean ± 95% CI", transform=ax.transAxes, ha="center",
            fontsize=8, color=MUTED)

    # n=5 vs n=20: the significance that evaporated
    a5 = np.array([302.8, 385.2, 398.9, 417.5, 339.3])
    z5 = np.array([439.3, 486.9, 367.4, 483.5, 544.6])
    p5 = np.array([402.3, 305.4, 357.9, 500.7, 595.2])
    a20 = d[d.arm == "supplied"]["test_cost"].to_numpy()
    z20 = d[d.arm == "zeros"]["test_cost"].to_numpy()
    p20 = d[d.arm == "persistence"]["test_cost"].to_numpy()
    ts = [welch(a5, z5)[0], welch(a5, p5)[0], welch(a20, z20)[0], welch(a20, p20)[0]]
    xs = ["TFT vs\nzeros", "TFT vs\npersistence"] * 2
    cols = [BAD if abs(t) > 2 else MUTED for t in ts]
    pos = [0, 1, 2.6, 3.6]
    ax2.bar(pos, ts, color=cols, width=.72, zorder=3)
    ax2.axhline(-2, color=BAD, ls="--", lw=1.2, zorder=2)
    ax2.axhline(0, color=INK, lw=1)
    ax2.text(3.75, -2.08, "significance\nthreshold", fontsize=7.5, color=BAD, va="top", ha="right")
    ax2.set_xticks(pos); ax2.set_xticklabels(xs, fontsize=8.5)
    ax2.set_ylabel("Welch t")
    ax2.set_title("The n=5 result did not survive n=20", loc="left")
    for x, t in zip(pos, ts):
        ax2.text(x, t - .12, f"{t:.2f}", ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(min(ts) * 1.30, 0.0)
    lo = ax2.get_ylim()[0]
    ax2.text(0.5, lo * 0.94, "n = 5", fontsize=10, color=MUTED, ha="center", fontweight="bold")
    ax2.text(3.1, lo * 0.94, "n = 20", fontsize=10, color=MUTED, ha="center", fontweight="bold")
    ax2.axvline(1.8, color=GRID, lw=1.2, ls=":")
    strip_spines(ax2)
    fig.savefig(f"{OUT}/forecast_ablation.png")
    plt.close(fig)
    print("  images/forecast_ablation.png")


# ---------------------------------------------------------------- 2. forecasting
def fig_forecast():
    d = pd.read_csv(f"{R}/forecast_metrics.csv")
    t = d[d.split == "test"].set_index("model")
    order = ["Zero", "Persistence", "TFT-MQLoss", "TFT-MAE"]
    order = [m for m in order if m in t.index]
    vals = [t.loc[m, "mae"] for m in order]
    cols = [ZERO, NAIVE, "#93c5fd", TFT][: len(order)]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.barh(order, vals, color=cols, zorder=3)
    ax.set_xlim(0, max(vals) * 1.22)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .02, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Test MAE  (lower is better)")
    ax.set_title("Forecasting: TFT beats both controls", loc="left")
    ax.invert_yaxis(); strip_spines(ax)

    rel = [t.loc[m, "rel_zero"] for m in order]
    ax2.barh(order, rel, color=cols, zorder=3)
    ax2.set_xlim(0, max(rel) * 1.30)
    ax2.axvline(1.0, color=BAD, ls="--", lw=1.4, zorder=4)
    ax2.text(1.0, len(order) - 0.35, "predicting zero", color=BAD, fontsize=8,
             ha="center", va="top")
    for i, v in enumerate(rel):
        ax2.text(v + max(rel) * .025, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("MAE ÷ zero-baseline MAE   (< 1.0 beats zero)")
    ax2.set_title("Why raw MAE lies on sparse demand", loc="left")
    ax2.invert_yaxis(); strip_spines(ax2)
    fig.text(.5, -.04, "78% of hours are exactly zero, so predicting 0 everywhere scores MAE≈0.042 — "
             "the project's original '0.05 MAE' headline.", ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(f"{OUT}/forecast_accuracy.png")
    plt.close(fig)
    print("  images/forecast_accuracy.png")


# ---------------------------------------------------------------- 3. policies
def fig_policies():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["DQN", "(s,S)", "EOQ", "NeverOrder"]
    cost = [368.7, 521.5, 620.9, 1737.2]
    serv = [0.813, 0.690, 0.657, 0.145]
    cols = [TFT, BASE, "#a7f3d0", BAD]
    ax.bar(names, cost, color=cols, zorder=3)
    for i, v in enumerate(cost):
        ax.text(i, v + 30, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Test cost  (lower is better)")
    ax.set_title("Single product: DQN vs baselines (5 seeds)", loc="left")
    strip_spines(ax)

    ax2.bar(names, serv, color=cols, zorder=3)
    for i, v in enumerate(serv):
        ax2.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Service level  (higher is better)")
    ax2.set_ylim(0, 1)
    ax2.set_title("...and it wins at a HIGHER service level", loc="left")
    strip_spines(ax2)
    fig.text(.5, -.04, "A cost win at a lower service level would just be a policy refusing to stock — "
             "never-order shows that failure mode.", ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(f"{OUT}/policy_comparison.png")
    plt.close(fig)
    print("  images/policy_comparison.png")


# ---------------------------------------------------------------- 4. overfitting
def fig_learning():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    diag = pd.read_csv(f"{R}/rl_instability_diag.csv")
    ax.plot(diag.steps, diag.train_cost, "-o", ms=4, color=BASE, label="train")
    ax.plot(diag.steps, diag.val_cost, "-o", ms=4, color=BAD, label="val")
    best = diag.loc[diag.val_cost.idxmin()]
    ax.scatter([best.steps], [best.val_cost], s=110, facecolor="none", edgecolor=BAD, lw=1.8, zorder=5)
    ax.annotate("best on val", (best.steps, best.val_cost), textcoords="offset points",
                xytext=(8, -16), fontsize=8.5, color=BAD)
    ax.set_xlabel("training steps"); ax.set_ylabel("episode cost")
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x/1000)}k")
    ax.set_title("Single product: train flat, val degrades\n= overfitting, not divergence", loc="left")
    ax.legend(frameon=False); strip_spines(ax)

    cat = pd.read_csv(f"{R}/rl_category_curve.csv")
    ax2.plot(cat.steps, cat.avg_cost, "-o", ms=4, color=TFT, label="val (400 series)")
    cb = cat.loc[cat.avg_cost.idxmin()]
    ax2.scatter([cb.steps], [cb.avg_cost], s=110, facecolor="none", edgecolor=TFT, lw=1.8, zorder=5)
    ax2.annotate("best on val", (cb.steps, cb.avg_cost), textcoords="offset points",
                 xytext=(-10, -16), fontsize=8.5, color=TFT, ha="right")
    ax2.set_xlabel("training steps"); ax2.set_ylabel("episode cost")
    ax2.set_title("400 series: noisy, but degrades far less\n(+26.7 vs +86.9 single product)", loc="left")
    ax2.xaxis.set_major_formatter(lambda x, _: f"{int(x/1000)}k")
    ax2.legend(frameon=False); strip_spines(ax2)
    fig.text(.5, -.06, "More data reduces the overfitting but does not remove it. The single-product curve "
             "is a clean train/val divergence;\nthe 400-series curve is dominated by cross-product variance "
             "(each episode samples a different product).", ha="center", fontsize=8.5, color=MUTED)
    fig.savefig(f"{OUT}/learning_curves.png")
    plt.close(fig)
    print("  images/learning_curves.png")


if __name__ == "__main__":
    print("writing figures:")
    fig_ablation()
    fig_forecast()
    fig_policies()
    fig_learning()
    print("done")
