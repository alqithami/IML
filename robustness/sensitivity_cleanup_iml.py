import os, re, glob, shutil, subprocess, sys
import pandas as pd
import yaml

RUNS = sorted(glob.glob("runs/cleanup_iml_agents5_seed*"))
if not RUNS:
    raise SystemExit("No runs matched runs/cleanup_iml_agents5_seed*")

P_FALSE = [1e-4, 1e-3, 1e-2]
P_REVIEW = [0.0, 0.2, 0.5, 0.8]
EPISODES = 20
EVAL_SEED = 0

OUT_CSV = "robustness/sensitivity_cleanup_iml_grid.csv"
OUT_FIG = "robustness/fig_sensitivity_cleanup_iml.pdf"
TMP_ROOT = "robustness/tmp_sens"

os.makedirs("robustness", exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)

rows = []

for run_dir in RUNS:
    run_name = os.path.basename(run_dir)

    cfg_yaml = os.path.join(run_dir, "config.yaml")
    cfg_json = os.path.join(run_dir, "config.json")
    model_pt = os.path.join(run_dir, "model.pt")

    if not (os.path.isfile(cfg_yaml) and os.path.isfile(model_pt)):
        print("[skip missing]", run_name)
        continue

    with open(cfg_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("ppo", {})
    cfg["ppo"]["device"] = "mps"  # use Apple GPU if evaluator supports it

    cfg.setdefault("iml", {})
    cfg["iml"]["enabled"] = True

    for pf in P_FALSE:
        for pr in P_REVIEW:
            tmp_dir = os.path.join(TMP_ROOT, f"{run_name}_pf{pf:g}_pr{pr:g}")
            os.makedirs(tmp_dir, exist_ok=True)

            shutil.copyfile(model_pt, os.path.join(tmp_dir, "model.pt"))
            shutil.copyfile(cfg_yaml, os.path.join(tmp_dir, "config.yaml"))
            if os.path.isfile(cfg_json):
                shutil.copyfile(cfg_json, os.path.join(tmp_dir, "config.json"))

            cfg2 = dict(cfg)
            cfg2["ppo"] = dict(cfg.get("ppo", {}))
            cfg2["iml"] = dict(cfg.get("iml", {}))
            cfg2["iml"]["p_detect_false"] = float(pf)
            cfg2["iml"]["p_review"] = float(pr)

            with open(os.path.join(tmp_dir, "config.yaml"), "w") as f:
                yaml.safe_dump(cfg2, f, sort_keys=False)

            cmd = [sys.executable, "-m", "iml_ssd.experiments.evaluate",
                   "--run_dir", tmp_dir, "--episodes", str(EPISODES), "--seed", str(EVAL_SEED)]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if p.returncode != 0:
                print("[FAIL]", run_name, "pf", pf, "pr", pr)
                print(p.stdout[-1500:])
                continue

            ev_path = os.path.join(tmp_dir, "eval.csv")
            if not (os.path.isfile(ev_path) and os.path.getsize(ev_path) > 0):
                print("[FAIL no eval.csv]", run_name, pf, pr)
                continue

            ev = pd.read_csv(ev_path)
            sanctions = float(ev["iml_sanctions"].mean()) if "iml_sanctions" in ev.columns else 0.0
            overturned = float(ev["iml_overturned"].mean()) if "iml_overturned" in ev.columns else 0.0

            rows.append({
                "run_name": run_name,
                "train_seed": int(re.search(r"_seed(\d+)$", run_name).group(1)),
                "p_detect_false": pf,
                "p_review": pr,
                "episodes": EPISODES,
                "eval_seed": EVAL_SEED,
                "return_mean": float(ev["return_mean"].mean()),
                "gini": float(ev["return_gini"].mean()),
                "false_pos": float(ev["iml_false_pos"].mean()) if "iml_false_pos" in ev.columns else 0.0,
                "overturned": overturned,
                "net_sanctions": sanctions - overturned,
            })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV, "rows=", len(df))

import numpy as np
import matplotlib.pyplot as plt

ACCENT = "#1f77b4"
BLACK  = "#000000"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
})

agg = (df.groupby(["p_detect_false","p_review"], as_index=False)
         .agg(return_mean=("return_mean","mean"),
              net_sanctions=("net_sanctions","mean")))

linestyles = {1e-4: ":", 1e-3: "--", 1e-2: "-"}

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)

for pf in sorted(agg["p_detect_false"].unique()):
    sub = agg[agg["p_detect_false"] == pf].sort_values("p_review")
    x = sub["p_review"].to_numpy()
    axes[0].plot(x, sub["return_mean"].to_numpy(), linestyle=linestyles.get(pf,"-"), color=BLACK, linewidth=1.1, label=f"p_false={pf:g}")
    axes[1].plot(x, sub["net_sanctions"].to_numpy(), linestyle=linestyles.get(pf,"-"), color=BLACK, linewidth=1.1, label=f"p_false={pf:g}")

nom = agg[(agg["p_detect_false"] == 1e-2) & (agg["p_review"] == 0.2)]
if len(nom) == 1:
    axes[0].plot([0.2], [float(nom["return_mean"].iloc[0])], marker="o", markersize=5, color=ACCENT)
    axes[1].plot([0.2], [float(nom["net_sanctions"].iloc[0])], marker="o", markersize=5, color=ACCENT)

axes[0].set_xlabel("p_review")
axes[0].set_ylabel("Eval return (mean per agent)")
axes[0].set_title("Cleanup IML: welfare sensitivity")
axes[1].set_xlabel("p_review")
axes[1].set_ylabel("Net sanctions / episode")
axes[1].set_title("Cleanup IML: sanction overhead")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

fig.savefig(OUT_FIG)
print("Wrote", OUT_FIG)
