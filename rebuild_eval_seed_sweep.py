import os, glob, pathlib
import pandas as pd

files = sorted(glob.glob("runs/*/eval_seed*.csv"))

# Ignore macOS resource-fork files if they exist
files = [f for f in files if "/._" not in f and not os.path.basename(f).startswith("._")]

if not files:
    raise SystemExit("No files matched runs/*/eval_seed*.csv")

rows = []
for fp in files:
    if os.path.getsize(fp) == 0:
        continue

    run_name = pathlib.Path(fp).parts[-2]          # runs/<run_name>/eval_seedX.csv
    eval_seed = int(pathlib.Path(fp).stem.replace("eval_seed", ""))

    ev = pd.read_csv(fp)

    def mean(col, default=0.0):
        return float(ev[col].mean()) if col in ev.columns else float(default)

    sanctions = mean("iml_sanctions", 0.0)
    overturned = mean("iml_overturned", 0.0)

    rows.append({
        "run_name": run_name,
        "eval_seed": eval_seed,
        "return_mean": mean("return_mean"),
        "return_sum": mean("return_sum"),
        "gini": mean("return_gini"),
        "false_pos": mean("iml_false_pos", 0.0),
        "overturned": overturned,
        "net_sanctions": sanctions - overturned,
    })

df = pd.DataFrame(rows).sort_values(["run_name", "eval_seed"])

os.makedirs("results", exist_ok=True)
df.to_csv("results/eval_seed_sweep.csv", index=False)

agg = (df.groupby("run_name", as_index=False)
         .agg(
             return_mean_mean=("return_mean","mean"),
             return_mean_std=("return_mean","std"),
             return_sum_mean=("return_sum","mean"),
             return_sum_std=("return_sum","std"),
             gini_mean=("gini","mean"),
             gini_std=("gini","std"),
             net_sanctions_mean=("net_sanctions","mean"),
             net_sanctions_std=("net_sanctions","std"),
             false_pos_mean=("false_pos","mean"),
             false_pos_std=("false_pos","std"),
             overturned_mean=("overturned","mean"),
             overturned_std=("overturned","std"),
         ))
agg.to_csv("results/eval_seed_sweep_agg.csv", index=False)

print("found_files", len(files))
print("wrote_rows_eval_seed_sweep", len(df))
print("unique_eval_seeds", sorted(df.eval_seed.unique().tolist()))
print("wrote_rows_eval_seed_sweep_agg", len(agg))
