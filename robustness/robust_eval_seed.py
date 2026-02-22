import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACCENT = "#1f77b4"   # single accent color
BLACK  = "#000000"
GRAY   = "#7a7a7a"
LGRAY  = "#c0c0c0"

df = pd.read_csv("results/eval_seed_sweep.csv")

pat = re.compile(r"^(cleanup|harvest)_(baseline|iml)_agents(\d+)_seed(\d+)$")
m = df["run_name"].apply(lambda x: pat.match(x))
if m.isna().any():
    bad = df.loc[m.isna(), "run_name"].unique().tolist()
    raise SystemExit(f"Could not parse run_name for: {bad[:10]}")

df["env"] = df["run_name"].str.extract(r"^(cleanup|harvest)_")[0]
df["cond"] = df["run_name"].str.extract(r"^(?:cleanup|harvest)_(baseline|iml)_")[0]
df["agents"] = df["run_name"].str.extract(r"_agents(\d+)_")[0].astype(int)
df["train_seed"] = df["run_name"].str.extract(r"_seed(\d+)$")[0].astype(int)

# per-run (train_seed) stats across eval seeds
run = (df.groupby(["env","cond","agents","train_seed","run_name"], as_index=False)
         .agg(
             return_mean_mean=("return_mean","mean"),
             return_mean_std=("return_mean","std"),
             gini_mean=("gini","mean"),
             gini_std=("gini","std"),
             net_sanctions_mean=("net_sanctions","mean"),
             net_sanctions_std=("net_sanctions","std"),
         ))

# paired deltas (IML - baseline) per env×train_seed
pivot_ret = run.pivot_table(index=["env","agents","train_seed"], columns="cond", values="return_mean_mean")
pivot_gin = run.pivot_table(index=["env","agents","train_seed"], columns="cond", values="gini_mean")
paired = (pivot_ret.reset_index()
            .rename_axis(None, axis=1)
            .merge(pivot_gin.reset_index().rename_axis(None, axis=1),
                   on=["env","agents","train_seed"], suffixes=("_ret","_gini")))
paired["delta_return_mean"] = paired["iml_ret"] - paired["baseline_ret"]
paired["delta_gini"] = paired["iml_gini"] - paired["baseline_gini"]
paired.to_csv("robustness/eval_seed_paired_deltas.csv", index=False)

# env×cond summary across training seeds
summary = (run.groupby(["env","cond"], as_index=False)
             .agg(
                 n_seeds=("train_seed","nunique"),
                 return_mean=("return_mean_mean","mean"),
                 return_sd_across_seeds=("return_mean_mean","std"),
                 avg_eval_seed_sd=("return_mean_std","mean"),
                 gini=("gini_mean","mean"),
                 gini_sd_across_seeds=("gini_mean","std"),
                 avg_eval_seed_sd_gini=("gini_std","mean"),
                 net_sanctions=("net_sanctions_mean","mean"),
                 net_sanctions_sd=("net_sanctions_mean","std"),
             ))
summary["return_ci95"] = 1.96 * summary["return_sd_across_seeds"] / np.sqrt(summary["n_seeds"])
summary["gini_ci95"]   = 1.96 * summary["gini_sd_across_seeds"] / np.sqrt(summary["n_seeds"])
summary.to_csv("robustness/eval_seed_robustness_summary.csv", index=False)

# ---- Figure: 2×2 panels (Return top row, Gini bottom row; Cleanup left, Harvest right)
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), constrained_layout=True)

def paired_panel(ax, env, ycol_mean, ycol_std, ylabel, title):
    r = run[run["env"] == env].copy()
    seeds = sorted(r["train_seed"].unique().tolist())

    for s in seeds:
        rb = r[(r["train_seed"] == s) & (r["cond"] == "baseline")]
        ri = r[(r["train_seed"] == s) & (r["cond"] == "iml")]
        if rb.empty or ri.empty:
            continue

        yb = float(rb[ycol_mean].iloc[0])
        yi = float(ri[ycol_mean].iloc[0])
        sb = float(rb[ycol_std].iloc[0]) if not np.isnan(rb[ycol_std].iloc[0]) else 0.0
        si = float(ri[ycol_std].iloc[0]) if not np.isnan(ri[ycol_std].iloc[0]) else 0.0

        ax.plot([0, 1], [yb, yi], color=LGRAY, linewidth=1.0, zorder=1)
        ax.errorbar([0], [yb], yerr=[sb], fmt="o", color=BLACK, ecolor=BLACK,
                    elinewidth=0.9, capsize=2, markersize=4, zorder=2)
        ax.errorbar([1], [yi], yerr=[si], fmt="o", color=ACCENT, ecolor=ACCENT,
                    elinewidth=0.9, capsize=2, markersize=4, zorder=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "IML"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(False)

paired_panel(axes[0,0], "cleanup", "return_mean_mean", "return_mean_std", "Eval return (mean per agent)", "Cleanup: return robustness")
paired_panel(axes[0,1], "harvest", "return_mean_mean", "return_mean_std", "Eval return (mean per agent)", "Harvest: return robustness")
paired_panel(axes[1,0], "cleanup", "gini_mean", "gini_std", "Eval inequality (Gini)", "Cleanup: inequality robustness")
paired_panel(axes[1,1], "harvest", "gini_mean", "gini_std", "Eval inequality (Gini)", "Harvest: inequality robustness")

fig.suptitle("Robustness to evaluation RNG seeds (error bars: std over eval seeds)", fontsize=10)
fig.savefig("robustness/fig_eval_seed_robustness.pdf")
print("Wrote robustness/eval_seed_robustness_summary.csv")
print("Wrote robustness/eval_seed_paired_deltas.csv")
print("Wrote robustness/fig_eval_seed_robustness.pdf")
