import argparse
import json
import math
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import select

import config
from database_components import Experiment, Generation, Individual, Population
from revolve2.experimentation.database import OpenMethod, open_database_sqlite

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 35,
    "axes.labelsize": 35,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30,
    "figure.titlesize": 35,
    "figure.figsize": (15, 12),
})

# ---------- helpers ----------

def _parse_history(hist):
    if hist is None or (isinstance(hist, float) and math.isnan(hist)):
        return (np.nan, np.nan, np.nan)
    try:
        if isinstance(hist, (str, bytes)):
            hist = json.loads(hist)
        if isinstance(hist, list) and hist:
            triple = hist[0]
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                return (float(triple[0]), float(triple[1]), float(triple[2]))
    except Exception:
        pass
    return (np.nan, np.nan, np.nan)

def _add_delta_columns(df):
    parsed = df["fitness_history"].apply(_parse_history)
    df[["unlearned", "learned", "delta"]] = pd.DataFrame(parsed.tolist(), index=df.index)
    return df

def _load_single_db(db_path: str) -> pd.DataFrame:
    """Open one SQLite and return a parsed dataframe with delta, plus db metadata."""
    engine = open_database_sqlite(db_path, open_method=OpenMethod.OPEN_IF_EXISTS)
    query = (
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.id.label("individual_id"),
            Individual.fitness_history,
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id)
    )
    df = pd.read_sql(query, engine)
    if df.empty:
        return df

    df = _add_delta_columns(df)
    df["db_name"] = os.path.basename(db_path)
    return df

def _gather_databases(names, folder, pattern):
    """Build a list of sqlite paths."""
    paths = []
    # -name can be provided multiple times without extension
    for n in names or []:
        p = os.path.join("Databases", f"{n}.sqlite") if folder is None else os.path.join(folder, f"{n}.sqlite")
        paths.append(p)
    # --folder + --glob pattern (default *.sqlite)
    if folder is not None:
        pat = pattern or "*.sqlite"
        paths.extend(sorted(glob.glob(os.path.join(folder, pat))))
    # Fallback: single -name legacy usage
    if not paths:
        # keep old behavior: Databases/<name>.sqlite
        pass
    # De-dup and keep only existing files
    uniq = []
    seen = set()
    for p in paths:
        if p not in seen and os.path.isfile(p):
            uniq.append(p)
            seen.add(p)
    return uniq

# ---------- plotting (unchanged from your fixed version) ----------

def plotFitness(df, name, figName, fMax, fMin):
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
          .agg({name: ["max", "mean"]})
          .reset_index()
    )
    agg_per_experiment_per_generation.columns = [
        "experiment_id", "generation_index", "max_fitness", "mean_fitness"
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
          .agg({"max_fitness": ["mean", "std"], "mean_fitness": ["mean", "std"]})
          .reset_index()
    )
    agg_per_generation.columns = [
        "generation_index",
        "max_fitness_mean", "max_fitness_std",
        "mean_fitness_mean", "mean_fitness_std",
    ]

    plt.figure()

    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["max_fitness_mean"],
        label=f"Max {name}",
        lw=4,
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["max_fitness_mean"] - agg_per_generation["max_fitness_std"],
        agg_per_generation["max_fitness_mean"] + agg_per_generation["max_fitness_std"],
        alpha=0.2,
    )

    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_fitness_mean"],
        label=f"Mean {name}",
        lw=4,
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation["mean_fitness_mean"] - agg_per_generation["mean_fitness_std"],
        agg_per_generation["mean_fitness_mean"] + agg_per_generation["mean_fitness_std"],
        alpha=0.2,
    )

    if name == "delta":
        plt.axhline(y=0, linestyle="--", alpha=0.5, linewidth=2)

    title = f"Mean and max {name} across repetitions with std as shade"
    if figName:
        title += ("\n" + figName)
    plt.xlabel("Generation index")
    ylabel = "Learning Delta (fitness improvement)" if name == "delta" else "Fitness (no. of targets reached)"
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.set_xlim(0, 10)  # Temporary
    ax.set_ylim(fMin, fMax)
    plt.grid(which="major", axis="both")
    plt.title(title)
    plt.legend()

def plotLearningDelta_per_experiment_per_id(df, figName, fMax, fMin):
    for exp_id, df_exp in df.groupby("experiment_id"):
        plt.figure()
        for ind_id, df_ind in df_exp.groupby("individual_id"):
            df_ind = df_ind.sort_values("generation_index")
            plt.plot(
                df_ind["generation_index"],
                df_ind["delta"],
                linewidth=1.0,
                alpha=0.35,
            )

        per_gen = (
            df_exp.groupby("generation_index")["delta"]
                  .agg(["mean", "std"])
                  .reset_index()
                  .sort_values("generation_index")
        )
        plt.plot(
            per_gen["generation_index"],
            per_gen["mean"],
            label="Mean Δ (this experiment)",
            linewidth=4,
        )
        plt.fill_between(
            per_gen["generation_index"],
            per_gen["mean"] - per_gen["std"],
            per_gen["mean"] + per_gen["std"],
            alpha=0.2,
            label="±1 std",
        )

        plt.axhline(y=0, linestyle="--", alpha=0.5, linewidth=2)

        title = f"Per-ID learning delta per generation — Experiment {exp_id}"
        if figName:
            title += f"\n{figName}"
        plt.xlabel("Generation index")
        plt.ylabel("Learning Delta (learned − unlearned)")
        ax = plt.gca()
        ax.set_xlim(0, 10)  # Temporary
        ax.set_ylim(fMin, fMax)
        plt.grid(which="major", axis="both")
        plt.title(title)
        plt.legend()

# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning delta across one or many databases.")
    # You can pass -name multiple times: -name exp01 -name exp02 ...
    parser.add_argument("-name", action="append",
                        help="Database base name (without extension). Can be repeated.")
    parser.add_argument("--folder", type=str, default="Databases",
                        help="Folder containing sqlite files (default: Databases).")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob pattern inside folder (e.g., 'exp_*.sqlite'). Defaults to *.sqlite when --folder is set.")
    parser.add_argument("-figName", type=str, help="Custom figure title.")
    parser.add_argument("-fMax", type=float, default=3.0, help="Max y-value.")
    parser.add_argument("-fMin", type=float, default=-1.0, help="Min y-value.")
    parser.add_argument("-plotDelta", action="store_true",
                        help="Plot aggregated delta across experiments (mean/max with std shade).")
    parser.add_argument("-perID", action="store_true",
                        help="Plot per-ID delta per experiment (one figure per experiment).")
    args = parser.parse_args()

    # Build the list of DB paths
    db_paths = _gather_databases(args.name, args.folder, args.glob)

    if not db_paths:
        # Preserve old single-db behavior if user passed exactly one -name but file missing
        if args.name and len(args.name) == 1:
            single = os.path.join("Databases", f"{args.name[0]}.sqlite")
            print(f"Could not find any DBs. Checked: {single} and {args.folder}/{args.glob or '*.sqlite'}")
        else:
            print("No databases found. Use -name repeatedly, or --folder with --glob.")
        return

    # Load and stack
    dfs = []
    for p in db_paths:
        try:
            dfp = _load_single_db(p)
            if not dfp.empty:
                dfs.append(dfp)
            else:
                print(f"Warning: no rows in {p}")
        except Exception as e:
            print(f"Error reading {p}: {e}")

    if not dfs:
        print("No usable data across the provided databases.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Sanity: ensure we have deltas
    if df["delta"].isnull().all():
        print("No valid 'delta' values parsed from fitness_history in any DB.")
        return

    figName = args.figName or f"{len(db_paths)} databases"

    # Per-ID per experiment (figures per experiment across ALL DBs)
    if args.perID:
        plotLearningDelta_per_experiment_per_id(df, figName, args.fMax, args.fMin)

    # Aggregated across experiments and databases
    if args.plotDelta:
        # Reuse generic aggregator on 'delta'
        plotFitness(df, "delta", figName, args.fMax, args.fMin)

    # Default if no flags: per-ID view
    if not args.perID and not args.plotDelta:
        plotLearningDelta_per_experiment_per_id(df, figName, args.fMax, args.fMin)

    plt.show()

if __name__ == "__main__":
    main()
