from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bin_curves(df: pd.DataFrame, step_col: str, value_col: str, bin_size: int) -> pd.DataFrame:
    # Bin env steps
    max_step = int(df[step_col].max())
    bins = np.arange(0, max_step + bin_size, bin_size)
    if len(bins) < 2:
        bins = np.array([0, max_step + 1])

    df = df.copy()
    df["step_bin"] = pd.cut(df[step_col], bins=bins, labels=bins[:-1], include_lowest=True).astype(int)
    # Average within each run/seed/bin
    gb = df.groupby(["env", "num_agents", "iml_enabled", "seed", "step_bin"], as_index=False)[value_col].mean()
    return gb


def _mean_ci_across_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # Aggregate across seeds for each condition and bin
    def agg(g):
        vals = g[value_col].to_numpy(dtype=float)
        m = float(np.mean(vals))
        if len(vals) <= 1:
            return pd.Series({"mean": m, "lo": m, "hi": m})
        se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        lo = m - 1.96 * se
        hi = m + 1.96 * se
        return pd.Series({"mean": m, "lo": lo, "hi": hi})

    out = df.groupby(["env", "num_agents", "iml_enabled", "step_bin"]).apply(agg).reset_index()
    return out


def plot_learning_curves(curves: pd.DataFrame, out_dir: Path, bin_size: int = 10_000) -> None:
    _ensure_dir(out_dir)
    for env_name in sorted(curves["env"].unique()):
        sub = curves[curves["env"] == env_name]
        # Return sum
        gb = _bin_curves(sub, "env_step", "return_sum", bin_size=bin_size)
        ci = _mean_ci_across_seeds(gb, "return_sum")

        plt.figure(figsize=(6, 4), dpi=200)
        for enabled in [False, True]:
            s = ci[ci["iml_enabled"] == enabled].sort_values("step_bin")
            if s.empty:
                continue
            label = "IML" if enabled else "Baseline"
            x = s["step_bin"].to_numpy()
            y = s["mean"].to_numpy()
            lo = s["lo"].to_numpy()
            hi = s["hi"].to_numpy()
            plt.plot(x, y, label=label)
            plt.fill_between(x, lo, hi, alpha=0.2)
        plt.xlabel("Environment steps (binned)")
        plt.ylabel("Collective episodic return (sum over agents)")
        plt.title(f"{env_name}: Learning curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"learning_curves_{env_name}.pdf")
        plt.close()

        # Fairness: Gini
        gb = _bin_curves(sub, "env_step", "return_gini", bin_size=bin_size)
        ci = _mean_ci_across_seeds(gb, "return_gini")
        plt.figure(figsize=(6, 4), dpi=200)
        for enabled in [False, True]:
            s = ci[ci["iml_enabled"] == enabled].sort_values("step_bin")
            if s.empty:
                continue
            label = "IML" if enabled else "Baseline"
            x = s["step_bin"].to_numpy()
            y = s["mean"].to_numpy()
            lo = s["lo"].to_numpy()
            hi = s["hi"].to_numpy()
            plt.plot(x, y, label=label)
            plt.fill_between(x, lo, hi, alpha=0.2)
        plt.xlabel("Environment steps (binned)")
        plt.ylabel("Return Gini (across agents)")
        plt.title(f"{env_name}: Fairness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"fairness_gini_{env_name}.pdf")
        plt.close()

        # Sanctions (only meaningful for IML)
        if "iml_sanctions" in curves.columns:
            gb = _bin_curves(sub, "env_step", "iml_sanctions", bin_size=bin_size)
            ci = _mean_ci_across_seeds(gb, "iml_sanctions")
            plt.figure(figsize=(6, 4), dpi=200)
            for enabled in [False, True]:
                s = ci[ci["iml_enabled"] == enabled].sort_values("step_bin")
                if s.empty:
                    continue
                label = "IML" if enabled else "Baseline"
                x = s["step_bin"].to_numpy()
                y = s["mean"].to_numpy()
                lo = s["lo"].to_numpy()
                hi = s["hi"].to_numpy()
                plt.plot(x, y, label=label)
                plt.fill_between(x, lo, hi, alpha=0.2)
            plt.xlabel("Environment steps (binned)")
            plt.ylabel("Sanctions per episode")
            plt.title(f"{env_name}: Institutional sanctions")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"sanctions_{env_name}.pdf")
            plt.close()


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    if summary.empty:
        return
    # Bar plot of eval_return_sum by env/condition
    if "eval_return_sum" in summary.columns:
        for env_name in sorted(summary["env"].unique()):
            sub = summary[summary["env"] == env_name]
            # group
            grp = sub.groupby(["iml_enabled"], as_index=False)["eval_return_sum"].mean()
            plt.figure(figsize=(4, 3), dpi=200)
            plt.bar(["Baseline", "IML"], [grp.loc[grp["iml_enabled"] == False, "eval_return_sum"].mean(),
                                         grp.loc[grp["iml_enabled"] == True, "eval_return_sum"].mean()])
            plt.ylabel("Eval collective return (mean)")
            plt.title(f"{env_name}: Eval welfare")
            plt.tight_layout()
            plt.savefig(out_dir / f"eval_welfare_{env_name}.pdf")
            plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--out_dir", type=str, default="figures")
    p.add_argument("--bin_size", type=int, default=10_000)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    curves_path = results_dir / "learning_curves.csv"
    summary_path = results_dir / "summary.csv"

    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        plot_learning_curves(curves, out_dir, bin_size=args.bin_size)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        plot_summary(summary, out_dir)

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
