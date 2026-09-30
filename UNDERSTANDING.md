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

We use the form $Ax \le b$, where $A$ is a matrix and $b$ is a vector, to denote constraints. Formally, writing $x(S) \in \{0,1\}^N$ for the incidence vector of an assortment $S$ (so $x_i(S) = 1$ if $i \in S$ and $0$ otherwise), the feasible family is

$$\mathcal{S} = \{ S \subseteq \{1, \dots, N\} \;:\; A\,x(S) \le b, \; 0 \le x \le 1 \}$$

where $A$ is totally unimodular and $b$ is integral.

The admissible policies can only act based on the information they have seen in the past. Writing $\mathcal{H}_t$ for the history available before time $t$, the assortment offered at time $t$ is

$$S_t = \pi_t(\mathcal{H}_t), \qquad \mathcal{H}_t = \sigma\big(U, c_1, \dots, c_{t-1}, S_1, \dots, S_{t-1}\big)$$

where $c_t$ is the customer's choice at time $t$ (with $c_t = 0$ denoting no purchase) and $U$ carries any extra randomisation the policy uses.

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

$$S_{\ell+1} := \arg\max_{S \in \mathcal{S}} \; \max \left\{ R(S, \hat{v}) : \hat{v}_i \le v^{\mathrm{UCB}}_{i,\ell} \right\}$$

This turns out to be equivalent to a much simpler problem — we just evaluate the revenue at the upper bounds themselves:

$$S_{\ell+1} := \arg\max_{S \in \mathcal{S}} \tilde{R}_{\ell+1}(S), \qquad \tilde{R}_{\ell+1}(S) := \frac{\sum_{i \in S} r_i v^{\mathrm{UCB}}_{i,\ell}}{1 + \sum_{j \in S} v^{\mathrm{UCB}}_{j,\ell}}$$

And finally we write the algorithm 1:

> **Algorithm 1** — Exploration-Exploitation algorithm for MNL-Bandit
>
> 1. **Initialization:** $v^{\mathrm{UCB}}_{i,0} = 1$ for all $i = 1, \dots, N$
> 2. $t = 1$, $\ell = 1$ keep track of the time steps and total number of epochs respectively
> 3. **while** $t < T$ **do**
> 4. &nbsp;&nbsp;&nbsp;&nbsp; Compute $S_\ell = \arg\max_{S \in \mathcal{S}} \tilde{R}_\ell(S)$
> 5. &nbsp;&nbsp;&nbsp;&nbsp; Offer assortment $S_\ell$, observe the purchasing decision $c_t$ of the consumer
> 6. &nbsp;&nbsp;&nbsp;&nbsp; **if** $c_t = 0$ **then**
> 7. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; compute $\hat{v}_{i,\ell}$, the number of consumers who preferred $i$ in epoch $\ell$, for all $i \in S_\ell$
> 8. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; update $\mathcal{T}_i(\ell)$ and $T_i(\ell)$
> 9. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; update $\bar{v}_{i,\ell}$, the sample mean of the estimates
> 10. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; update $v^{\mathrm{UCB}}_{i,\ell}$; $\ell = \ell + 1$
> 11. &nbsp;&nbsp;&nbsp;&nbsp; **else**
> 12. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $E_\ell = E_\ell \cup t$
> 13. &nbsp;&nbsp;&nbsp;&nbsp; **end if**
> 14. &nbsp;&nbsp;&nbsp;&nbsp; $t = t + 1$
> 15. **end while**


# Theorem 1

For any instance $v = (v_0, \dots, v_N)$ of the MNL-Bandit problem with $N$ products, $r_i \in [0,1]$, and under Assumption 4.1, the regret of Algorithm 1 at any time $T$ is bounded as

$$\mathrm{Reg}^\pi(T, v) \;\le\; C_1 \sqrt{NT \log NT} \;+\; C_2 N \log^2 NT$$

where $C_1$ and $C_2$ are absolute constants, independent of the problem parameters.

Assumption 4.1 has two parts: $v_i \le v_0 = 1$ for every product (no purchase is the most likely outcome), and the feasible family is closed under taking subsets ($S \in \mathcal{S}$ and $Q \subseteq S$ imply $Q \in \mathcal{S}$).

If we prove this, we can say with high probability that our algorithm will converge and will have this upper bound.

# Proof sketch

We first establish that $v_i$ is bounded by $v^{\mathrm{UCB}}_{i,\ell}$ with high probability. We do this by assuming $\hat{v}_{i,\ell}$ to be an unbiased estimator of $v_i$ and a geometric variable. Using the Chernoff-Hoeffding bounds for geometric variables, we then get the $v^{\mathrm{UCB}}$ formulation — essentially the upper confidence bounds converge to the true quantities $v_i$.

First we establish that $\hat{v}_{i,\ell}$ is an unbiased estimate. Then we prove, using the moment generating function, that $\hat{v}_{i,\ell}$ is a geometric random variable with mean $v_i$, so the expected value of $\hat{v}_{i,\ell}$ is $v_i$. The moment generating function conditioned on the offered assortment is

$$\mathbb{E}^\pi\left[ e^{\theta \hat{v}_{i,\ell}} \;\Big|\; S_\ell \right] = \frac{1}{1 - v_i(e^\theta - 1)}, \qquad \theta \le \log \frac{1 + v_i}{v_i}$$

The important point is that the right hand side does not depend on $S_\ell$ at all, only on $v_i$. This identifies $\hat{v}_{i,\ell}$ as geometric with parameter $\tfrac{1}{1 + v_i}$, so

$$\mathbb{P}\big( \hat{v}_{i,\ell} = m \big) = \left( \frac{v_i}{1 + v_i} \right)^{m} \left( \frac{1}{1 + v_i} \right), \qquad \mathbb{E}\big[ \hat{v}_{i,\ell} \big] = v_i$$

We use this and multiplicative Chernoff-Hoeffding bounds to get the upper confidence bound conditions (Lemma 4.1):

$$v^{\mathrm{UCB}}_{i,\ell} \ge v_i \quad \text{w.p. at least } 1 - \frac{6}{N\ell}$$

$$v^{\mathrm{UCB}}_{i,\ell} - v_i \;\le\; C_1 \sqrt{\frac{v_i \log\big(\sqrt{N\ell} + 1\big)}{T_i(\ell)}} + C_2 \frac{\log\big(\sqrt{N\ell} + 1\big)}{T_i(\ell)} \quad \text{w.p. at least } 1 - \frac{7}{N\ell}$$

The first says we never underestimate; the second says the overestimate shrinks as $T_i(\ell)$ grows.

We show that our optimistic assortment value $\tilde{R}_\ell$ always bounds the optimal $R(S^\star, v)$, so the estimated revenue converges to the optimal expected revenue. We do this by showing that the value of the expected revenue corresponding to the optimal assortment increases with an increase in the MNL parameters, proving $\tilde{R}_\ell(S^\star) \ge R(S^\star, v)$. Then we know that since we choose $S_\ell$ such that it is optimal for $\tilde{R}_\ell$, it will always be greater than $\tilde{R}_\ell(S^\star)$. Putting the two together (Lemma 4.2),

$$\tilde{R}_\ell(S_\ell) \;\ge\; \tilde{R}_\ell(S^\star) \;\ge\; R(S^\star, v) \quad \text{w.p. at least } 1 - \frac{6}{\ell}$$

Here the monotonicity claim is only about the optimal assortment: the expected revenue is not a monotone function of $v$ in general, but the value achieved by the assortment that is optimal for its own parameters is.

Finally, since $\tilde{R}_\ell$ is known and we can subtract the revenues of epochs to get a measurable quantity, and we can subtract the revenues of epochs from $R(S^\star, v)$ to get the regret, we can bound the regret. Combining the two inequalities above, the per-epoch regret is bounded by something we can actually measure:

$$\underbrace{R(S^\star, v) - R(S_\ell, v)}_{\text{what we want, involves the unknown } S^\star} \;\le\; \underbrace{\tilde{R}_\ell(S_\ell) - R(S_\ell, v)}_{\text{estimation error on the assortment we offered}}$$

The right hand side is then controlled by a Lipschitz-type bound (Lemma 4.3), which converts the revenue error into a sum of the individual parameter errors:

$$\Big( 1 + \sum_{j \in S_\ell} v_j \Big) \big( \tilde{R}_\ell(S_\ell) - R(S_\ell, v) \big) \;\le\; \sum_{i \in S_\ell} \left( C_1 \sqrt{\frac{v_i \log\big(\sqrt{N\ell}+1\big)}{T_i(\ell)}} + C_2 \frac{\log\big(\sqrt{N\ell}+1\big)}{T_i(\ell)} \right)$$

And we can solve further to get the proposed regret bounds. Lemmas 4.1, 4.2 and 4.3 are used to prove this step by step. Summing over epochs and using $\sum_{k=1}^{T_i} \tfrac{1}{\sqrt{k}} = O(\sqrt{T_i})$, together with the fact that the horizon limits how often products can be offered,

$$\sum_{i=1}^{N} v_i \, \mathbb{E}^\pi\big(T_i\big) \;\le\; T$$

a Cauchy-Schwarz step gives $\sum_i \sqrt{v_i \mathbb{E}(T_i)} \le \sqrt{NT}$, and the regret bound is

$$O\!\left( \sqrt{NT \log NT} \right)$$

This is better than the regret bound we have seen before by Sauré and Zeevi. We see this in the results too.

# Experiments

**Experiment 1.** We assume 4 products to have value $v + \epsilon$ and the other products to have value $v$. Concretely the paper uses $N = 10$, $K = 4$, $r_i = 1$, and

$$v_i = \begin{cases} 0.25 + \epsilon, & i \in \{1, 2, 9, 10\} \\ 0.25, & \text{otherwise} \end{cases} \qquad 0 < \epsilon < 0.25$$

with $\epsilon \in \{0.05, 0.1, 0.15, 0.25\}$. As we vary $\epsilon$ we see that some algorithms work only when the products have well separated attractiveness, which shows how robust the MNL-Bandit algorithm is.

**Experiment 2.** We look at a dataset of cars and their attributes and estimate the attractiveness values using these attributes, such as maintenance costs, luggage capacity and safety perception. Each car is described by six categorical attributes, which are one-hot encoded into a vector $m_i \in \{0,1\}^{22}$, and the attractiveness is taken to be

$$v_i = e^{\theta \cdot m_i}$$

where $\theta \in \mathbb{R}^{22}$ is fitted from the data. Cars have 4 ratings: unacceptable, acceptable, good, very good. Unacceptable is a sign for no buy, and the other ratings are a sign for a purchase, so the fit is a logistic regression on that binary label:

$$\theta_{\mathrm{MLE}} = \arg\max_{\theta} \sum_{i=1}^{N} \log p_{\mathrm{buy}}(\theta, m_i) - \lVert \theta \rVert^2, \qquad p_{\mathrm{buy}}(\theta, m) = \frac{e^{\theta \cdot m}}{1 + e^{\theta \cdot m}}$$

The fitted $\theta_{\mathrm{MLE}}$ is then treated as the ground truth and used to simulate customer choices. Here $N = 1728$ and $K = 100$.

The paper shows that for Experiment 2 the initial regret is smaller but the later regret is higher for SZ, but for Algorithm 1 the regret decreases sublinearly — while being high initially, it decreases fast.

When I reproduced the algorithm, the results were a bit different, where SZ seemed to be the better algorithm for Experiment 2, but we see SZ perform subpar when separability is low.