# Problem Statement

We have $N$ substitutable objects, out of which we must select the optimal $K$ or $\le K$.

The above problem was solved using a model which is called the Multinomial Logit. We essentially assign an attractiveness score to each value, normalise it to $1$ for no purchase, and then write the probability of choosing $i$ out of an assortment $S$ as

$$p_i(S) = \frac{v_i}{1 + \sum_{j \in S} v_j}$$

The issue is that in real life scenarios we don't quite know $v_i$, and we must use some algorithm to first estimate these values $v_i$ using given data and then act on that data to show the optimal assortment.

This approach of learning the most while also profiting the most demands an exploration-exploitation algorithm.

The paper describes an algorithm for this usage. It uses an upper confidence bound, so we optimistically estimate a certain value that we know is better than the optimal reward we can get, to then estimate regret and also impose a bound on the regret.

Establishing a bound on the regret is extremely important as we must prove that the algorithm does decently with a certain high probability, and that it also converges with a certain high probability.

What is interesting here is how we can have better algorithms that work under different constraints and perform better.

# A bit more formal description of the problem

$$p_i(S) = \frac{v_i}{1 + \sum_{j \in S} v_j}$$

$$R(S, v) = \frac{\sum_{i \in S} r_i v_i}{1 + \sum_{j \in S} v_j}$$

The above is the expected revenue, and $r_i$ is the revenue due to the $i$-th product.

We use the form $Ax \le b$, where $A$ is a matrix and $b$ is a vector, to denote constraints.

The admissible policies can only act based on the information they have seen in the past.

The goal is to maximize the cumulative expected revenue across all iterations,

$$\mathbb{E}^\pi\left( \sum_{t=1}^{T} R(S_t, v) \right)$$

We measure the performance of a policy using regret.

Regret = Reward due to optimal assortment shown all of the times $-$ cumulative expected revenue,

$$\mathrm{Reg}^\pi(T, v) = \sum_{t=1}^{T} \Big( R(S^\star, v) - \mathbb{E}^\pi\big[R(S_t, v)\big] \Big), \qquad S^\star = \arg\max_{S \in \mathcal{S}} R(S, v)$$

# Algorithm 1

This is an extension of the UCB Bandit algorithm proposed in 2002.

We run an epoch till we get a no purchase; when we get a no purchase the epoch ends. Writing $E_\ell$ for the set of time steps in epoch $\ell$, this means $E_\ell$ contains every time step after the end of epoch $\ell - 1$ up to and including the step at which the no purchase happens.

$\hat{v}_{i,\ell}$ is defined as the number of times a product $i$ is purchased in the epoch $\ell$:

$$\hat{v}_{i,\ell} := \sum_{t \in E_\ell} \mathbf{1}(c_t = i)$$

We denote the number of epochs that offer $i$ till the epoch $\ell$ by $T_i(\ell)$. More precisely, we track both the set of such epochs and its size:

$$\mathcal{T}_i(\ell) = \{ \tau \le \ell \mid i \in S_\tau \}, \qquad T_i(\ell) = |\mathcal{T}_i(\ell)|$$

We define $\bar{v}_{i,\ell}$ as the number of times product $i$ was purchased per epoch:

$$\bar{v}_{i,\ell} = \frac{1}{T_i(\ell)} \sum_{\tau \in \mathcal{T}_i(\ell)} \hat{v}_{i,\tau}$$

Both $\hat{v}_{i,\ell}$ and $\bar{v}_{i,\ell}$ are unbiased estimators of $v_i$.

And we define $v^{\mathrm{UCB}}$ as

$$v^{\mathrm{UCB}}_{i,\ell} := \bar{v}_{i,\ell} + \sqrt{\bar{v}_{i,\ell} \, \frac{48 \log\big(\sqrt{N\ell} + 1\big)}{T_i(\ell)}} + \frac{48 \log\big(\sqrt{N\ell} + 1\big)}{T_i(\ell)}$$

This is an upper confidence bound on the true parameter, i.e. $v^{\mathrm{UCB}}_{i,\ell} \ge v_i$ for all $i$ and $\ell$ with high probability.

Then we take the optimistic assortment: the assortment that is best over every parameter vector consistent with our upper bounds,

$$S_{\ell+1} := \arg\max_{S \in \mathcal{S}} \; \max_{\hat{v}_i \le v^{UCB}_{i,\ell}} R(S, \hat{v})$$

This turns out to be equivalent to a much simpler problem — we just evaluate the revenue at the upper bounds themselves:

$$S_{\ell+1} := \arg\max_{S \in \mathcal{S}} \tilde{R}_{\ell+1}(S), \qquad \tilde{R}_{\ell+1}(S) := \frac{\sum_{i \in S} r_i v^{\mathrm{UCB}}_{i,\ell}}{1 + \sum_{j \in S} v^{\mathrm{UCB}}_{j,\ell}}$$

And finally we write the algorithm 1:

**Algorithm 1** (Exploration-Exploitation algorithm for MNL-Bandit)

```
 1: Initialization: v_UCB[i,0] = 1 for all i = 1, ..., N
 2: t = 1, l = 1            (time steps and number of epochs)
 3: while t < T do
 4:     compute S_l = argmax over S of R~_l(S)
 5:     offer assortment S_l, observe the purchase decision c_t
 6:     if c_t = 0 then
 7:         compute v^[i,l], the number of consumers who preferred i
                  in epoch l, for all i in S_l
 8:         update T_i(l), the number of epochs until l that offered i
 9:         update v-bar[i,l], the sample mean of the estimates
10:         update v_UCB[i,l];  l = l + 1
11:     else
12:         E_l = E_l union {t}
13:     end if
14:     t = t + 1
15: end while
```


# Theorem 1

For any instance $v = (v_0, \dots, v_N)$ of the MNL-Bandit problem with $N$ products, $r_i \in [0,1]$, and under Assumption 4.1, the regret of Algorithm 1 at any time $T$ is bounded as

$$\mathrm{Reg}^\pi(T, v) \;\le\; C_1 \sqrt{NT \log NT} \;+\; C_2 N \log^2 NT$$

where $C_1$ and $C_2$ are absolute constants, independent of the problem parameters.

If we prove this, we can say with high probability that our algorithm will converge and will have this upper bound.

# Proof sketch

We first establish that $v_i$ is bounded by $v^{UCB}$ with high probability. We do this by assuming $\hat{v}$ to be an unbiased estimator of $v_i$ and a geometric variable. Using the Chernoff-Hoeffding bounds for geometric variables we then get the $v^{UCB}$ formulation. Essentially the upper confidence bounds converge to the true quantities $v_i$.

First we establish that $\hat{v}$ is an unbiased estimate. Then we prove using the moment generating function that $\hat{v}$ is a geometric random variable with mean $v_i$, so expected value of $\hat{v}$ is $v_i$.

$$\mathbb{E}^{\pi} \left[ e^{\theta \hat{v}_{i,\ell}} \mid S_{\ell} \right] = \frac{1}{1 - v_i (e^{\theta} - 1)}$$

$$\mathbb{E} \left[ \hat{v}_{i,\ell} \right] = v_i$$

We use this and multiplicative Chernoff-Hoeffding bounds to get the upper confidence bound conditions.

$$v^{UCB}_{i,\ell} \ge v_i \qquad \text{with probability at least } 1 - \frac{6}{N \ell}$$

$$v^{UCB}_{i,\ell} - v_i \le C_1 \sqrt{\frac{v_i \log (\sqrt{N \ell} + 1)}{T_i(\ell)}} + C_2 \frac{\log (\sqrt{N \ell} + 1)}{T_i(\ell)}$$

We show that our optimistic assortment of $\tilde{R}$ always bounds the optimal $R^{*}$, so estimated revenue converges to the optimal expected revenue. We do this by showing that value of expected revenue corresponding to the optimal assortment increases with increase in MNL parameters, proving

$$\tilde{R}_{\ell}(S^{*}) \ge R(S^{*}, v)$$

Then we know that since we choose $S_{\ell}$ such that it is optimal for $\tilde{R}_{\ell}$, it will always be greater than $\tilde{R}_{\ell}(S^{*})$.

$$\tilde{R}_{\ell}(S_{\ell}) \ge \tilde{R}_{\ell}(S^{*}) \ge R(S^{*}, v)$$

Finally, since $\tilde{R}$ is known and we can subtract the revenues of epochs to get a measurable quantity, and we can subtract the revenues of epochs from $R^{*}$ to get the regret, we can bound the regret.

$$R(S^{*}, v) - R(S_{\ell}, v) \le \tilde{R}_{\ell}(S_{\ell}) - R(S_{\ell}, v)$$

And we can solve further to get the proposed regret bounds. Lemma 4.1, 4.2, 4.3 are used to prove this step by step. The regret bound is

$$O \left( \sqrt{N T \log N T} \right)$$

This is better than the regret bound we have seen before by Sauré and Zeevi. We see this in results too.

# Experiments

**Experiment 1.** We assume 4 products to have value $v + \epsilon$ and other products to have value $v$.

$$v_i = \begin{cases} 0.25 + \epsilon, & i \in \{1, 2, 9, 10\} \\ 0.25, & \text{else} \end{cases}$$

As we vary $\epsilon$ we see that some algorithms work only when the products have well separated attractiveness, which shows how robust the MNL-Bandit algorithm is.

**Experiment 2.** We look at a dataset of cars and their attributes and estimate the attractiveness values using these attributes, such as maintenance costs, luggage capacity, safety perception. Cars have 4 ratings: unacceptable, acceptable, good, very good. Unacceptable is a sign for no buy, the other ratings are a sign for a purchase.

$$v_i = e^{\theta \cdot m_i}$$

The paper shows that for Experiment 2 initial regret is smaller but later regret is higher for SZ, but for Algorithm 1 the regret decreases sublinearly, while being high initially, it decreases fast.

When I reproduced the algorithm, the results were a bit different, where SZ seemed to be the better algorithm for Experiment 2, but we see SZ perform subpar when separability is low.
