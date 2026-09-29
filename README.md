# MNL-Bandit

Implementation and partial replication of Agrawal, Avadhanula, Goyal & Zeevi
(2019), *MNL-Bandit: A Dynamic Learning Approach to Assortment Selection*.

## What is here

| path | contents |
|---|---|
| [`notebooks/mnl_bandit_experiments.ipynb`](notebooks/mnl_bandit_experiments.ipynb) | **the results** — both experiments, plots, and commentary |
| [`code/algorithms.py`](code/algorithms.py) | MNL model, assortment oracle, Algorithm 1, Algorithm 3, Sauré–Zeevi |
| [`code/run_experiments.py`](code/run_experiments.py) | runs both experiments, writes the JSON |
| [`results/experiment_results.json`](results/) | precomputed output (41 min) |
| [`data/car.data`](data/) | UCI Car Evaluation, unmodified |

## Algorithms implemented

| | paper reference |
|---|---|
| Algorithm 1 | §3.2, confidence bound eq. (3.4) |
| Algorithm 3 | §6.2 |
| Sauré–Zeevi (2013) | their Algorithm 1, exploration constant $20\log T$ as used in paper §7.2 |

## Experiments

1. **Synthetic** — instance eq. (7.1), $N=10$, $K=4$, at $\varepsilon \in
   \{0.05, 0.10, 0.15, 0.25\}$. Horizons up to $10^5$, 10 runs each.
2. **UCI Car Evaluation** — $N=1728$, $K=100$. Horizons up to $10^5$, 6 runs each.
   The dataset has no purchase data; §4 of the notebook shows the construction
   used to obtain $v$.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/mnl_bandit_experiments.ipynb
```

The notebook loads the precomputed JSON, so plots appear immediately. Section 6
runs the algorithms live (~2 min).

To regenerate the results:

```bash
python code/run_experiments.py --quick   # ~3 min
python code/run_experiments.py           # ~41 min
```

## Implementation checks

- All three algorithms reproduce an independent earlier implementation in this
  project to the cent on identical seeds.
- The assortment oracle agrees with exhaustive search over all subsets.
- An oracle policy given the true parameters scores exactly zero regret.
