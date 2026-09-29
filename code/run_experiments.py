"""Run the two experiments and save the results as JSON.

This is the script that produced `results/experiment_results.json`, which the
notebook loads. You do not need to run it to read the notebook -- it is here so
that every number in the notebook is reproducible.

    python code/run_experiments.py            # full run (~2.5 hours)
    python code/run_experiments.py --quick    # small version (~3 minutes)

The two runs write to different files, so a quick run will not overwrite the
full results.

Experiment 1  Synthetic instance, eq. (7.1) of the paper.
              Four difficulty levels (eps), three algorithms.

Experiment 2  UCI Car Evaluation, 1728 real cars.
              Three algorithms, several horizons.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Make the `algorithms` package importable no matter where this is run from.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from algorithms import (                        # noqa: E402
    algorithm_1, algorithm_3, saure_zeevi,
    synthetic_instance, car_instance, best_assortment,
)

RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")

ALGORITHMS = {
    "Algorithm 1": algorithm_1,
    "Algorithm 3": algorithm_3,
    "Saure-Zeevi": saure_zeevi,
}


def mean_and_error(values):
    """Mean and standard error of a list of numbers."""
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std_error": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
        "values": [float(x) for x in a],
    }


def experiment_1(horizons, n_seeds, eps_values):
    """Synthetic instance: how does each algorithm do as the problem gets easier?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1  Synthetic instance (paper eq. 7.1)")
    print("=" * 70)

    out = {}
    for eps in eps_values:
        v, r, K = synthetic_instance(eps)
        best_value, best_set = best_assortment(v, r, K)
        readable = tuple(int(i) + 1 for i in best_set)      # 1-based, like the paper
        print(f"\n  eps = {eps}   best assortment = {readable}"
              f"   its revenue = {best_value:.4f}")

        per_algorithm = {}
        for name, algo in ALGORITHMS.items():
            rows = []
            for T in horizons:
                regrets = [algo(v, r, K, T, seed=1000 + s)["regret"]
                           for s in range(n_seeds)]
                rows.append({"T": T, **mean_and_error(regrets)})
                print(f"    {name:12s} T = {T:>7,}   "
                      f"regret = {rows[-1]['mean']:8.1f}")
            per_algorithm[name] = rows

        out[str(eps)] = {
            "epsilon": eps,
            "best_revenue": float(best_value),
            "best_assortment": [int(i) for i in best_set],
            "results": per_algorithm,
        }
    return out


def experiment_2(horizons, n_seeds, K=100):
    """UCI cars: the same three algorithms on a real dataset."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2  UCI Car Evaluation (1728 real cars)")
    print("=" * 70)

    data = car_instance()
    v, r = data["v"], data["r"]
    best_value, best_set = best_assortment(v, r, K)
    print(f"\n  {data['n_cars']} cars, {data['n_features']} features")
    print(f"  showing at most K = {K} cars at a time")
    print(f"  best possible purchase probability = {best_value:.6f}")

    per_algorithm = {}
    for name, algo in ALGORITHMS.items():
        rows = []
        for T in horizons:
            regrets = [algo(v, r, K, T, seed=7000 + s)["regret"]
                       for s in range(n_seeds)]
            rows.append({"T": T, **mean_and_error(regrets)})
            print(f"    {name:12s} T = {T:>7,}   "
                  f"regret = {rows[-1]['mean']:8.2f}")
        per_algorithm[name] = rows

    return {
        "K": K,
        "best_revenue": float(best_value),
        "results": per_algorithm,
        "dataset": {
            "n_cars": data["n_cars"],
            "n_features": data["n_features"],
            "accuracy": data["accuracy"],
            "buy_rate": data["buy_rate"],
            "largest_v": float(v.max()),
            "median_v": float(np.median(v)),
            "n_cars_with_v_above_1": int((v > 1).sum()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small, fast version for checking the code runs")
    args = ap.parse_args()

    if args.quick:
        eps_values = [0.05, 0.10, 0.15, 0.25]
        horizons_1 = [2000, 5000, 10000]
        seeds_1 = 3
        horizons_2 = [5000, 10000]
        seeds_2 = 2
        outfile = os.path.join(RESULTS_DIR, "experiment_results_quick.json")
    else:
        eps_values = [0.05, 0.10, 0.15, 0.25]
        horizons_1 = [2000, 5000, 10000, 25000, 50000, 100000]
        seeds_1 = 10
        horizons_2 = [10000, 25000, 50000, 100000]
        seeds_2 = 6
        outfile = os.path.join(RESULTS_DIR, "experiment_results.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    start = time.time()

    payload = {
        "settings": {
            "quick": args.quick,
            "epsilon_values": eps_values,
            "horizons_experiment_1": horizons_1,
            "seeds_experiment_1": seeds_1,
            "horizons_experiment_2": horizons_2,
            "seeds_experiment_2": seeds_2,
        },
        "experiment_1": experiment_1(horizons_1, seeds_1, eps_values),
        "experiment_2": experiment_2(horizons_2, seeds_2),
    }
    payload["runtime_minutes"] = round((time.time() - start) / 60, 2)

    with open(outfile, "w") as fh:
        json.dump(payload, fh, indent=1)

    print(f"\nFinished in {payload['runtime_minutes']} minutes")
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
