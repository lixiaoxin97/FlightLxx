#!/usr/bin/env python3
"""Visualize recovery experiment results — publication-quality figures."""
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

COLORS = {"CTBR": "#0072B2", "CTBR+DR": "#D55E00", "CTBR+ID": "#009E73"}
MODELS = ["CTBR", "CTBR+DR", "CTBR+ID"]
PLATFORMS = ["prototype", "iris", "crazyflie"]
PLAT_LABELS = {
    "prototype": "Prototype\n(0.78 kg)", "iris": "Iris\n(1.5 kg)",
    "crazyflie": "Crazyflie\n(0.027 kg)"}
PLAT_SHORT = {"prototype": "Proto", "iris": "Iris", "crazyflie": "CF"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help="Results CSV")
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'figures'))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-compute stats
    stats = {}
    for p in PLATFORMS:
        for m in MODELS:
            sub = df[(df.platform == p) & (df.model == m)]
            succ = sub[sub.success]
            stats[(p, m)] = {
                "rate": sub.success.mean() * 100,
                "rate_ste": np.sqrt(sub.success.mean() * (1 - sub.success.mean()) / len(sub)) * 100,
                "rec": succ.recovery_time_s.mean() if len(succ) > 0 else None,
                "rec_ste": succ.recovery_time_s.std() / np.sqrt(len(succ)) if len(succ) > 0 else None,
                "err": sub.max_pos_error_m.mean(),
                "err_ste": sub.max_pos_error_m.std() / np.sqrt(len(sub)),
                "att": sub.max_att_error_deg.mean(),
                "jerk": succ.jerk_ms3.mean() if len(succ) > 0 else None,
                "spd": sub.max_speed_ms.mean(),
                "n": len(sub), "n_succ": len(succ),
            }

    # ============================================================
    # Figure 1: Combined 2×2 panel — success, error, recovery, attitude
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    x = np.arange(len(PLATFORMS))
    w = 0.25

    _bar(axes[0, 0], x, w, "rate", "Success Rate (%)",
         [(0, 100)], stats)
    _bar(axes[0, 1], x, w, "err", "Max Position Error (m)",
         [(0, 7)], stats)
    _bar(axes[1, 0], x, w, "rec", "Recovery Time (s)",
         [(0, 11)], stats, skip_missing=True, show_labels=False)
    _bar(axes[1, 1], x, w, "att", "Max Attitude Error (°)",
         [(0, 300)], stats)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], edgecolor="white", lw=0.5) for m in MODELS]
    fig.legend(handles, MODELS, loc="upper center", ncol=3, framealpha=0.85,
               edgecolor="#ccc", fontsize=8, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(pad=1.2, rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.output_dir, "combined_results.pdf"))

    # ============================================================
    # Figure 2: Heatmap table
    # ============================================================
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")

    cell_text = []
    for m in MODELS:
        row = []
        for p in PLATFORMS:
            s = stats[(p, m)]
            cell = f"{s['rate']:.0f}%\n{s['err']:.1f}m"
            row.append(cell)
        cell_text.append(row)

    col_labels = [PLAT_SHORT[p] for p in PLATFORMS]
    table = ax.table(cellText=cell_text, rowLabels=MODELS, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        if c == -1:
            cell.set_facecolor("#f0f0f0")
            cell.get_text().set_fontweight("bold")
        if r > 0 and c >= 0:
            rate = df[(df.platform == PLATFORMS[c]) & (df.model == MODELS[r - 1])].success.mean()
            if rate > 0.8:
                cell.set_facecolor("#d5f5e3")
            elif rate > 0.4:
                cell.set_facecolor("#fef9e7")

    ax.set_title("Success Rate (%) / Max Position Error (m)", fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout(pad=1)
    fig.savefig(os.path.join(args.output_dir, "summary_table.pdf"))
    plt.close("all")

    # ============================================================
    # Text summary
    # ============================================================
    print(f"Figures saved to: {args.output_dir}")
    print(f"\n{'='*60}")
    print("RECOVERY EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Platform':12s} {'Model':8s} {'Success':>7s} {'Rec(s)':>8s} "
          f"{'MaxErr(m)':>9s} {'MaxAtt':>6s} | trials")
    print(f"  {'-'*60}")
    for p in PLATFORMS:
        for m in MODELS:
            s = stats[(p, m)]
            rec_str = f"{s['rec']:.2f}s" if s['rec'] is not None else "   N/A "
            print(f"  {p:12s} {m:8s} {s['rate']:6.1f}% {rec_str:>8s} "
                  f"{s['err']:8.2f}m {s['att']:5.0f}° | {s['n_succ']}/{s['n']}")

    # Cross-platform ranking
    print(f"\n{'='*60}")
    print("CROSS-PLATFORM SUMMARY")
    print(f"{'='*60}")
    for m in MODELS:
        parts = []
        for p in PLATFORMS:
            s = stats[(p, m)]
            parts.append(f"{PLAT_SHORT[p]}={s['rate']:.0f}%")
        print(f"  {m:8s}: {' → '.join(parts)}")


def _bar(ax, x, w, key, ylabel, ylims, stats, skip_missing=False, show_labels=True):
    """Draw a grouped bar chart with error bars."""
    for i, model in enumerate(MODELS):
        vals, errs = [], []
        for p in PLATFORMS:
            s = stats[(p, model)]
            v = s[key]
            if v is None and skip_missing:
                v = 0
            vals.append(v if v is not None else 0)
            errs.append(s.get(f"{key}_ste", 0) or 0)
        ax.bar(x + i * w, vals, w, color=COLORS[model], edgecolor="white", lw=0.5)
        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0 and np.isfinite(v) and np.isfinite(e):
                ax.errorbar(x[j] + i * w, v, yerr=e, fmt="none", ecolor="#333",
                            capsize=3, lw=0.8)
            if show_labels and v > 0 and np.isfinite(v):
                offset = (e if np.isfinite(e) else 0) + ylims[0][1] * 0.02
                fmt = f"{v:.0f}" if key != "err" else f"{v:.1f}"
                ax.text(x[j] + i * w, v + offset, fmt, ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

    ax.set_xticks(x + w)
    ax.set_xticklabels([PLAT_LABELS[p] for p in PLATFORMS], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylims)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    main()
