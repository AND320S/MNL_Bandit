"""The MNL-Bandit problem and three algorithms that solve it.

Everything the notebook needs lives in this one file, deliberately: the point is
that a reader can follow the whole method top to bottom without jumping between
modules.

Reading order:

    1. THE MODEL            what an MNL customer is, and what regret means
    2. THE OFFLINE PROBLEM  which assortment is best when you already know v
    3. THE EPOCH TRICK      how to estimate v from censored purchase data
    4. ALGORITHM 1          Agrawal et al. (2019), Section 3
    5. ALGORITHM 3          the same idea without the v_i <= 1 assumption
    6. SAURE-ZEEVI          the earlier "explore, then commit" approach
    7. PROBLEM INSTANCES    the synthetic instance and the UCI car data

Reference: Agrawal, Avadhanula, Goyal & Zeevi (2019), "MNL-Bandit: A Dynamic
Learning Approach to Assortment Selection", Operations Research.
Equation numbers in the comments refer to that paper.
"""

import os

import numpy as np

# ===========================================================================
# 1. THE MODEL
# ===========================================================================
#
# A shop can display a set of products (an "assortment"). A customer looks at
# what is on offer and either buys ONE item or walks away buying nothing.
#
# Each product i has an unknown "attractiveness" v_i > 0. The no-purchase
# option has attractiveness v_0, which we fix at 1.0 -- this is just a choice
# of units, like measuring everything relative to "walking away".
#
# Given an assortment S, the customer picks product i with probability
#
#                            v_i
#     P(buys i) = ---------------------------            ... eq. (2.1)
#                  1 + sum of v_j over j in S
#
# and buys nothing with probability 1 / (1 + sum of v_j).
#
# The denominator is the same for every product, so a product's chance of being
# bought goes DOWN when you add other attractive products next to it. That is
# the whole difficulty: you cannot judge a product in isolation.

NO_PURCHASE_ATTRACTION = 1.0


def choice_probabilities(assortment, v):
    """Probability of buying each product in `assortment`, and of buying nothing.

    Returns (probability_per_product, probability_of_no_purchase).
    """
    attractions = np.asarray([v[i] for i in assortment], dtype=float)
    denominator = NO_PURCHASE_ATTRACTION + attractions.sum()
    return attractions / denominator, NO_PURCHASE_ATTRACTION / denominator


def expected_revenue(assortment, v, r):
    """Average revenue from one customer shown `assortment`.       ... eq. (2.2)

    If every r_i = 1 this is just the probability that the customer buys
    something.
    """
    if len(assortment) == 0:
        return 0.0
    per_product, _ = choice_probabilities(assortment, v)
    revenues = np.asarray([r[i] for i in assortment], dtype=float)
    return float(np.dot(per_product, revenues))


def simulate_one_customer(assortment, v, rng):
    """Draw one customer's decision. Returns a product index, or None."""
    per_product, no_buy = choice_probabilities(assortment, v)
    probabilities = np.append(per_product, no_buy)
    pick = rng.choice(len(probabilities), p=probabilities)
    return None if pick == len(assortment) else assortment[pick]


# ===========================================================================
# 2. THE OFFLINE PROBLEM
# ===========================================================================
#
# If we already knew v, which assortment of size <= K should we offer?
#
# Checking all subsets is impossible (there are astronomically many). But the
# MNL model has a convenient structure: for any target revenue L, the best
# assortment is found by ranking products by (r_i - L) * v_i and taking the
# positive ones. So we only have to search over candidate values of L, and the
# best L is always one of the achievable revenues.


def best_assortment(v, r, K):
    """Best assortment of size at most K, and the revenue it earns.

    Returns (revenue, assortment) with the assortment as a sorted tuple.
    """
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)

    def best_for_target(L):
        """Best set assuming we are aiming for revenue level L."""
        scores = (r - L) * v
        order = np.argsort(-scores)              # most promising first
        chosen = [i for i in order[:K] if scores[i] > 0]
        return tuple(sorted(chosen))

    # Try each product's revenue as a candidate level, plus the levels those
    # candidates actually produce. One of them is optimal.
    candidates = {0.0}
    candidates.update(float(x) for x in r)
    best_value, best_set = 0.0, ()
    for _ in range(2):                            # two passes is enough
        new_levels = set()
        for L in sorted(candidates):
            S = best_for_target(L)
            if not S:
                continue
            value = expected_revenue(S, v, r)
            new_levels.add(value)
            if value > best_value:
                best_value, best_set = value, S
        candidates |= new_levels
    return best_value, best_set


# ===========================================================================
# 3. THE EPOCH TRICK
# ===========================================================================
#
# Here is the problem the paper solves. If you show assortment S and the
# customer buys product 3, what have you learned about v_3? Not much on its
# own: the purchase probability depends on every product in S through the
# shared denominator.
#
# The trick: keep offering the SAME assortment S to arriving customers until
# somebody buys nothing. Call that stretch an "epoch". Then count how many
# times each product was bought during the epoch.
#
# Why this works. Compare "buys product i" with "buys nothing":
#
#     P(buys i)          v_i / (1 + sum v_j)
#     ------------  =  ------------------------  =  v_i
#     P(buys 0)           1  / (1 + sum v_j)
#
# The denominator cancels. From product i's point of view the epoch is just
# repeated coin-flipping with odds v_i : 1 against the stopping event, so the
# number of times i is bought before the epoch ends is a geometric random
# variable with mean EXACTLY v_i -- no matter what else was on offer.
#
# That is the estimator: purchases of i per epoch, averaged over epochs.


def run_one_epoch(assortment, v, rng):
    """Offer `assortment` until a customer buys nothing.

    Returns (purchase_counts_by_product, n_customers_served).
    The final no-purchase customer is included in the count.
    """
    counts = {}
    customers = 0
    while True:
        customers += 1
        bought = simulate_one_customer(assortment, v, rng)
        if bought is None:
            return counts, customers
        counts[bought] = counts.get(bought, 0) + 1


# ===========================================================================
# 4. ALGORITHM 1  (Agrawal et al. 2019, Section 3.2)
# ===========================================================================
#
# "Optimism in the face of uncertainty."
#
# We never know v exactly, so we keep an OPTIMISTIC guess for each product:
#
#     v_upper = (average purchases per epoch) + (a margin for uncertainty)
#
# The margin is large when we have few observations of a product and shrinks as
# we collect more. Then we simply offer the assortment that would be best if
# the optimistic guess were true.
#
# This is self-correcting. A product we have barely tried gets a big margin, so
# it looks attractive and gets offered -- and then we learn about it. A product
# we have tried a lot has a tiny margin, so the guess is close to the truth.
#
# Assumption: the paper's Theorem 1 assumes every v_i <= v_0 = 1, i.e. "buying
# nothing" is the single most likely outcome. See Algorithm 3 for what happens
# when that fails.


def confidence_margin(average, n_epochs, N, epoch_index):
    """The uncertainty margin added to a product's estimate.    ... eq. (3.4)

    Shrinks like 1/sqrt(n_epochs): more observations, smaller margin.
    """
    log_term = 48.0 * np.log(np.sqrt(N * epoch_index) + 1)
    return np.sqrt(average * log_term / n_epochs) + log_term / n_epochs


def algorithm_1(v, r, K, T, seed=0):
    """Algorithm 1 of the paper. Returns a dict with the total regret."""
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)
    N = len(v)
    rng = np.random.default_rng(seed)

    best_value, _ = best_assortment(v, r, K)

    total_purchases = np.zeros(N)     # total times each product was bought
    epochs_offered = np.zeros(N)      # how many epochs contained each product
    v_upper = np.ones(N)              # optimistic guess; starts at v_0 = 1

    t = 0                             # customers served so far
    epoch_index = 0
    regret = 0.0

    while t < T:
        epoch_index += 1

        # Choose the assortment that looks best under the optimistic guess.
        _, S = best_assortment(v_upper, r, K)
        if not S:
            S = tuple(range(min(K, N)))

        # Offer it until somebody buys nothing.
        counts, customers = run_one_epoch(S, v, rng)

        # Pay regret for every customer served, but do not count past T.
        served = min(customers, T - t)
        regret += served * (best_value - expected_revenue(S, v, r))
        t += customers

        # Update the estimates for the products we just offered.
        for i in S:
            total_purchases[i] += counts.get(i, 0)
            epochs_offered[i] += 1

        seen = epochs_offered > 0
        averages = np.zeros(N)
        averages[seen] = total_purchases[seen] / epochs_offered[seen]
        v_upper = np.ones(N)
        v_upper[seen] = averages[seen] + confidence_margin(
            averages[seen], epochs_offered[seen], N, epoch_index)

    return {"regret": regret, "epochs": epoch_index,
            "best_revenue": best_value}


# ===========================================================================
# 5. ALGORITHM 3  (Agrawal et al. 2019, Section 6.2)
# ===========================================================================
#
# Algorithm 1 assumes v_i <= 1 for every product. On real data that can fail
# badly -- in the car dataset below, the most attractive car has v_i = 258.
#
# Two things break, and Algorithm 3 fixes one each:
#
#   (a) The starting guess v_upper = 1 is supposed to be an OVER-estimate. If
#       the truth is 258, we start pessimistic, never offer the product, and
#       so never find out. Fix: force every product to be tried a minimum
#       number of times before trusting the optimiser.
#
#   (b) The margin uses sqrt(average), which is the right size when the average
#       is below 1 but far too small when it is large. Fix: use
#       max(sqrt(average), average), which agrees with Algorithm 1 exactly at
#       average = 1 and is wider above it.


def wide_confidence_margin(average, n_epochs, N, epoch_index):
    """Margin for Algorithm 3: correct for any size of v."""
    log_term = 48.0 * np.log(np.sqrt(N * epoch_index) + 1)
    spread = np.maximum(np.sqrt(average), average)
    return spread * np.sqrt(log_term / n_epochs) + log_term / n_epochs


def algorithm_3(v, r, K, T, seed=0):
    """Algorithm 3: like Algorithm 1 but valid without the v_i <= 1 assumption."""
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)
    N = len(v)
    rng = np.random.default_rng(seed)

    best_value, _ = best_assortment(v, r, K)

    total_purchases = np.zeros(N)
    epochs_offered = np.zeros(N)
    v_upper = np.ones(N)

    t = 0
    epoch_index = 0
    regret = 0.0
    forced_epochs = 0

    while t < T:
        epoch_index += 1
        required = 48.0 * np.log(np.sqrt(N * epoch_index) + 1)

        # First, what would the optimistic choice be? (paper, line 5)
        _, S = best_assortment(v_upper, r, K)
        if not S:
            S = tuple(range(min(K, N)))

        # (a) If any product IN THAT SET is still under-explored, replace the
        #     set with under-explored products instead (paper, lines 6-9).
        #     Among those we prefer the highest-revenue ones, so the forced
        #     exploration costs as little as possible.
        under_explored = [i for i in range(N) if epochs_offered[i] < required]
        if under_explored and any(epochs_offered[i] < required for i in S):
            forced_epochs += 1
            by_revenue = sorted(under_explored, key=lambda i: -r[i])
            S = tuple(sorted(by_revenue[:min(K, len(by_revenue))]))

        counts, customers = run_one_epoch(S, v, rng)
        served = min(customers, T - t)
        regret += served * (best_value - expected_revenue(S, v, r))
        t += customers

        for i in S:
            total_purchases[i] += counts.get(i, 0)
            epochs_offered[i] += 1

        seen = epochs_offered > 0
        averages = np.zeros(N)
        averages[seen] = total_purchases[seen] / epochs_offered[seen]
        v_upper = np.ones(N)
        # (b) the wider margin
        v_upper[seen] = averages[seen] + wide_confidence_margin(
            averages[seen], epochs_offered[seen], N, epoch_index)

    return {"regret": regret, "epochs": epoch_index,
            "forced_epochs": forced_epochs, "best_revenue": best_value}


# ===========================================================================
# 6. SAURE-ZEEVI  (Sauré & Zeevi 2013, Algorithm 1)
# ===========================================================================
#
# The earlier approach, included because the 2019 paper compares against it.
#
#   Phase 1 (explore):  split the products into groups, show each group to a
#                       fixed number of customers, and estimate v.
#   Phase 2 (exploit):  pick the best assortment under those estimates and
#                       show it to everyone else, forever.
#
# The estimate is v_i = (times i was bought) / (times nobody bought), which is
# the sample version of the ratio in section 3 above.
#
# The weakness: how long should phase 1 last? Too short and you commit to the
# wrong assortment and pay for it for the rest of time, with no way to notice.
# The right length depends on how close the best and second-best assortments
# are -- which is exactly what you do not know. The 2019 paper uses 20*log(T),
# so we do too.


def saure_zeevi(v, r, K, T, seed=0, exploration_constant=20.0):
    """Explore-then-commit. Returns a dict with the total regret."""
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)
    N = len(v)
    rng = np.random.default_rng(seed)

    best_value, best_set = best_assortment(v, r, K)

    # Split products into groups of size K: [0..K-1], [K..2K-1], ...
    groups = [tuple(range(i, min(i + K, N))) for i in range(0, N, K)]
    customers_per_group = int(np.ceil(exploration_constant * np.log(max(T, 3))))

    purchases = np.zeros(N)
    no_purchases = np.zeros(len(groups))

    t = 0
    regret = 0.0

    # ---- Phase 1: explore ----
    for g, S in enumerate(groups):
        gap = best_value - expected_revenue(S, v, r)
        for _ in range(customers_per_group):
            if t >= T:
                break
            bought = simulate_one_customer(S, v, rng)
            if bought is None:
                no_purchases[g] += 1
            else:
                purchases[bought] += 1
            regret += gap
            t += 1

    # ---- Estimate v from what we saw ----
    estimates = np.zeros(N)
    degenerate_groups = 0
    for g, S in enumerate(groups):
        members = list(S)
        if no_purchases[g] > 0:
            estimates[members] = purchases[members] / no_purchases[g]
        else:
            # Nobody ever declined this group, so the ratio is undefined. That
            # means the group is extremely attractive; mark it as such while
            # keeping the order of products within the group.
            degenerate_groups += 1
            top = purchases[members].max()
            if top > 0:
                estimates[members] = 1e6 * purchases[members] / top

    # ---- Phase 2: commit ----
    _, committed = best_assortment(estimates, r, K)
    if not committed:
        committed = tuple(range(min(K, N)))
    if t < T:
        regret += (T - t) * (best_value - expected_revenue(committed, v, r))

    return {"regret": regret,
            "committed_assortment": committed,
            "found_best": set(committed) == set(best_set),
            "customers_per_group": customers_per_group,
            "degenerate_groups": degenerate_groups,
            "best_revenue": best_value}


# ===========================================================================
# 7. PROBLEM INSTANCES
# ===========================================================================


def synthetic_instance(epsilon, N=10, K=4):
    """The paper's test problem, eq. (7.1).

    Ten products. Four of them (1, 2, 9, 10 in the paper's numbering) are
    slightly more attractive than the rest:

        v_i = 0.25 + epsilon   for those four
        v_i = 0.25             for the other six

    Every product earns the same revenue, r_i = 1, so the task is purely to
    find the four good products. `epsilon` controls the difficulty: small
    epsilon means the good and bad products are nearly identical and the
    problem is hard.

    Returns (v, r, K).
    """
    v = np.full(N, 0.25)
    v[[0, 1, N - 2, N - 1]] = 0.25 + epsilon
    r = np.ones(N)
    return v, r, K


# --- UCI Car Evaluation ---------------------------------------------------
#
# See `notebooks/mnl_bandit_experiments.ipynb`, "Where the car data comes
# from", for the full explanation of the steps below.

CAR_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CODE_DIR)
_DEFAULT_CAR_PATH = os.path.join(_PROJECT_DIR, "data", "car.data")


def _read_car_file(path):
    rows, ratings = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 7:
                rows.append(parts[:6])
                ratings.append(parts[6])
    return rows, ratings


def _one_hot(rows):
    """Turn six text columns into 0/1 columns, plus a constant column."""
    levels = [sorted({row[j] for row in rows}) for j in range(6)]
    names = ["(constant)"]
    for j, options in enumerate(levels):
        names += [f"{CAR_COLUMNS[j]}={o}" for o in options]

    M = np.zeros((len(rows), len(names)))
    M[:, 0] = 1.0
    for i, row in enumerate(rows):
        col = 1
        for j, options in enumerate(levels):
            for option in options:
                if row[j] == option:
                    M[i, col] = 1.0
                col += 1
    return M, names


def _fit_logistic(M, y, penalty=1.0, iterations=300):
    """Logistic regression, fitted by Newton's method.

    Maximises  sum_i log P(buy_i)  -  penalty * ||theta||^2 .
    """
    theta = np.zeros(M.shape[1])
    for _ in range(iterations):
        z = np.clip(M @ theta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = M.T @ (y - p) - 2.0 * penalty * theta
        weights = p * (1.0 - p) + 1e-12
        hessian = -(M.T * weights) @ M - 2.0 * penalty * np.eye(M.shape[1])
        step = np.linalg.solve(hessian, gradient)
        theta_next = theta - step
        if np.max(np.abs(theta_next - theta)) < 1e-10:
            return theta_next
        theta = theta_next
    return theta


def car_instance(path=None, penalty=1.0):
    """Build an MNL problem from the UCI Car Evaluation dataset.

    The dataset has 1728 cars, each described by six categorical attributes and
    labelled unacceptable / acceptable / good / very good. It contains no
    purchase data, so we manufacture a plausible ground truth:

      1. one-hot encode the attributes           -> 22 numbers per car
      2. call "acceptable or better" = would buy -> a 0/1 label
      3. fit a logistic regression               -> weights theta
      4. set v_i = exp(theta . features_i)       -> attractiveness per car

    Step 4 is what makes it an MNL problem. From then on the fitted v is
    TREATED AS THE TRUTH and used to simulate customers; the algorithms never
    see theta.

    Returns a dict with v, r and some descriptive numbers.
    """
    path = path or _DEFAULT_CAR_PATH
    rows, ratings = _read_car_file(path)
    M, names = _one_hot(rows)
    would_buy = np.array([0.0 if t == "unacc" else 1.0 for t in ratings])

    theta = _fit_logistic(M, would_buy, penalty=penalty)
    v = np.exp(M @ theta)

    predicted = (1.0 / (1.0 + np.exp(-np.clip(M @ theta, -30, 30)))) >= 0.5
    accuracy = float(np.mean(predicted == (would_buy >= 0.5)))

    return {
        "v": v,
        "r": np.ones(len(v)),
        "theta": theta,
        "features": M,
        "feature_names": names,
        "would_buy": would_buy,
        "ratings": ratings,
        "n_cars": len(v),
        "n_features": M.shape[1],
        "accuracy": accuracy,
        "buy_rate": float(would_buy.mean()),
    }
