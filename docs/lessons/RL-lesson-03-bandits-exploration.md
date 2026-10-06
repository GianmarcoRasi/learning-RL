# Lesson 3 — Multi-Armed Bandits: Exploration and Exploitation — Version 2

> **Version 2 — Expanded explanations and worked practical examples.**  
> Read each concept in words, then follow its numbers through the equation. The progressive theoretical exercises are retained here; the command-line labs and practical exercises are in the companion hands-on file.  
> **Study pace:** Allow 75–100 minutes for explanations and worked examples. The progressive theory sequence can be a separate 60–90 minute session.

> **Course:** Reinforcement Learning — Theory + Practice  
> **Prerequisites:** Lessons 1–2.  
> **Goal:** Learn why good decision making requires gathering information as well as using it.  

**Hands-on companion:** [Python labs and practical exercises](RL-lesson-03-bandits-exploration-hands-on.md)

### How to study this lesson

On a first pass, read the explanations and redo each worked example on paper. Then try the short in-text questions. Use the companion hands-on file for the command-line lab. Work through the progressive theory sequence in order; its final levels deliberately go beyond the first-pass material. An example shows the reasoning; an exercise asks you to reproduce or extend it.

## 1. What you will learn

Imagine choosing a lunch stall each day. You want enjoyable meals, but you do not yet know which stall you like best. Returning to your current favorite gives a reasonably promising lunch. Trying another might disappoint today, but could reveal a better choice for the rest of the month.

This lesson studies that tradeoff using a **multi-armed bandit**: a repeated choice among options with unknown reward distributions. The name comes from gambling machines with pull handles; an “arm” simply means one available option.

We temporarily remove state transitions to focus on **learning which action is best while still earning reward**. You will learn to estimate average rewards, update those estimates one observation at a time, choose using epsilon-greedy, and distinguish reward from regret. Repeated experiments matter because one lucky run is weak evidence.

## 2. A bandit is a repeated choice

Suppose three lunch stalls receive your personal satisfaction scores. For this simplified model, choosing a stall does not change its future quality or your future situation. Each visit gives an independent score drawn from that stall's fixed distribution.

| Stall (arm) | Example of varying scores | True mean score, unknown to the learner |
| --- | --- | ---: |
| A | Sometimes good, sometimes disappointing | $6$ |
| B | Variable, but better on average | $8$ |
| C | Usually less enjoyable | $5$ |

The means in the table describe the environment. The learner would have to discover them from visits; it does not get to read this column.

The true action value is

$$
q_*(a)=\mathbb E[R_t\mid A_t=a].
$$

Read this as: **the average reward from selecting arm $a$**. $A_t$ is the action on pull $t$, $R_t$ is the reward from that pull, and $\mathbb E$ averages its possible outcomes. The star marks the true mean. For B, $q_*(B)=8$, even if one visit earns only $3$.

We use $R_t$ locally for the reward of bandit pull $t$, rather than the transition notation $R_{t+1}$ of Lessons 1–2. Bandits have no next-state transition to track.

**Stationary** means each arm's reward distribution stays fixed over time. **Independent draws conditional on the chosen arm** means a previous score does not change the distribution of its next score. Real lunch choices can violate these assumptions: your tastes or the chef may change, and eating may alter tomorrow's appetite. The example is a simplified model to make the decision principle clear.

The environment does not move to new states, but the learner's information changes. In a full MDP, charging a battery changes later possibilities; in this bandit, trying B changes what we know about B.

## 3. Exploration versus exploitation

The learner keeps an estimate $Q_t(a)$ for each arm before pull $t$. It is a guess from available observations, rather than the hidden true mean $q_*(a)$.

- **Exploitation:** select an arm with the largest current estimate.
- **Exploration:** try options to gather information that may improve later decisions.

A greedy rule uses

$$
A_t\in\arg\max_a Q_t(a).
$$

$\arg\max$ means “which action gives the largest estimate?” If estimates are $[7,4,5]$, the greedy choice is A. The largest estimate is $7$, but the action is A: these are different outputs.

### Worked example: a disappointing first meal

Suppose the first observed scores are A: $7$, B: $3$, C: $5$. Pure greedy selection keeps choosing A. If B's true mean is $8$, its unlucky first score hides the best stall. Without visiting B again, the learner receives no evidence to correct that estimate.

A single observation does not establish an arm's average quality. This is why more time alone does not fix a greedy learner: it may spend that time collecting data only about A.

### Why pay for information?

If today's unknown stall could reveal a better option for the next 100 lunches, exploration may be worthwhile. If this is your final lunch, learning about tomorrow has no benefit. Noise, differences between means and the number of choices remaining all affect how useful information can be.

Exploration is not automatically beneficial on every visit. It accepts some immediate expected cost for a chance of better future choices. The final theoretical exercise calculates this tradeoff in a two-choice horizon.

## 4. Learning an action value: update an average

After observing rewards $r_1,\ldots,r_n$ from **one particular arm**, estimate its mean using

$$
Q_n=\frac1n\sum_{i=1}^n r_i.
$$

Here $n$ counts observations of that arm. The subscript in this section counts arm-specific updates, while $t$ elsewhere counts all pulls. Each arm needs its own count and estimate.

### Worked example: three visits to stall B

The scores are $3,9,6$:

| Visit to B | Observed score | Average after this visit |
| --- | ---: | --- |
| $1$ | $3$ | $3$ |
| $2$ | $9$ | $(3+9)/2=6$ |
| $3$ | $6$ | $(3+9+6)/3=6$ |

You do not need to store the full list. After $n-1$ visits, the total of earlier scores is $(n-1)Q_{n-1}$. Add the new score $r_n$ and divide by $n$:

$$
Q_n=\frac{(n-1)Q_{n-1}+r_n}{n}
=Q_{n-1}+\frac1n(r_n-Q_{n-1}).
$$

This is **old estimate plus a fraction of the prediction error**. On the second visit, the old estimate is $3$, the new score is $9$, and the error is $6$. Take half that error: $3+\tfrac12(9-3)=6$.

The equivalent third update is $6+\tfrac13(6-6)=6$. The latest score agrees with the old estimate, so nothing changes.

If ten visits to A happened between these visits to B, B's second-update denominator would still be two. Data about A is not data about B.

### Constant step sizes: remembering recent experience more strongly

The general update is

$$
Q\leftarrow Q+\alpha(R-Q).
$$

The arrow means replace the old estimate by the expression on the right. $R-Q$ is observed reward minus prediction, and $\alpha$ is the fraction of that difference we use.

With $Q=6$, a new score $10$, and $\alpha=0.25$:

$$
Q_{\text{new}}=6+0.25(10-6)=7.
$$

The estimate moves one quarter of the way from $6$ toward $10$. Using $\alpha=1/n$ reproduces the sample average. Using a fixed $\alpha$ weights recent observations more strongly: each later update reduces an older observation's influence by a factor $1-\alpha$.

If a stall changes chefs, a constant step size can track the new mean faster than an average diluted by hundreds of old meals. It still needs visits to that stall. Neither update rule learns anything new about an unselected arm.

### Theory exercise 3A — Incremental estimates

An arm produces rewards $2,-1,5$. Calculate its estimate after each reward, starting at zero and using $\alpha_n=1/n$. Then repeat using constant $\alpha=0.5$.

**Solution**

Sample-average estimates: $2$, $0.5$, $2$. Constant-step estimates: $1$, $0$, $2.5$. The sample average equals $(2-1+5)/3=2$; the constant-step estimate weights the observations differently and still depends on initialization.



## 5. Epsilon-greedy: mix a favorite with random trials

**Epsilon-greedy** uses a probability $\epsilon$ to decide how to choose:

1. With probability $\epsilon$, explore by selecting uniformly from **all** $k$ arms.
2. With probability $1-\epsilon$, exploit by selecting uniformly among arms tied for the largest estimate.

“Uniformly” means equally likely. $\epsilon=0$ gives pure greedy; $\epsilon=1$ gives uniform random selection regardless of estimates.

### Worked example: three stalls and $\epsilon=0.3$

Suppose B is the single current favorite. About 70% of decisions use the exploitation branch and select B. About 30% use the exploration branch; one third of that branch also selects B.

| Arm | Probability from exploitation | Probability from exploration | Total |
| --- | ---: | ---: | ---: |
| A | $0$ | $0.3/3=0.1$ | $0.1$ |
| B | $0.7$ | $0.1$ | $0.8$ |
| C | $0$ | $0.1$ | $0.1$ |

This explains the formula for one greedy arm:

$$
P(A_t=a_{\mathrm{greedy}})=1-\epsilon+\frac\epsilon k.
$$

Every other arm gets probability $\epsilon/k$. Exploration does not mean “definitely choose something different”: it can choose the favorite too. Here the exploration probability is $0.3$, but the chance of a non-greedy arm is only $0.2$.

If A and B tie, split the $0.7$ exploitation probability between them. Each gets $0.35+0.1=0.45$, while C gets $0.1$. Random tie-breaking avoids an arbitrary preference for whichever arm is listed first.

### What happens over a long horizon?

A fixed positive epsilon keeps collecting data from every arm. In a finite stationary bandit with finite reward means and the independent-draw assumptions here, sample averages can converge to those means. But the rule keeps sometimes choosing worse arms even after identifying the best.

Reducing epsilon over time can reduce that ongoing cost. Reducing it too quickly can leave an arm with a misleading early estimate and too few later observations. More exploration is not universally better: it trades information against reward over the available horizon.

### Theory exercise 3B — Action probabilities

There are four arms and one greedy arm. Let $\epsilon=0.2$. What is the probability of selecting the greedy arm? What if two arms tie for the maximum?

**Solution**

With one greedy arm: $0.8+0.2/4=0.85$. With two tied greedy arms, each gets $0.8/2+0.2/4=0.45$. Each other arm gets $0.05$.



## 6. Measuring performance: reward and missed opportunity

Reward tells us what was earned. **Regret** compares performance with a benchmark, usually always choosing an arm with the largest true mean. In the simulator, we know the hidden means and can measure **cumulative pseudo-regret**:

$$
\overline{\mathcal R}_T=\sum_{t=1}^T[q_*(a_*)-q_*(A_t)],
\qquad a_*\in\arg\max_a q_*(a).
$$

$T$ is the number of pulls. $a_*$ is a true best arm, and $A_t$ is what was actually chosen. For each pull, subtract the chosen arm's true mean from the best true mean, then add those gaps.

### Worked example: counting the mean gaps

Use lunch means A: $6$, B: $8$, C: $5$. A learner chooses A, B, C, B:

| Pull | Chosen arm | Chosen mean | Best mean | Gap |
| --- | --- | ---: | ---: | ---: |
| $1$ | A | $6$ | $8$ | $2$ |
| $2$ | B | $8$ | $8$ | $0$ |
| $3$ | C | $5$ | $8$ | $3$ |
| $4$ | B | $8$ | $8$ | $0$ |

Pseudo-regret is $2+0+3+0=5$. We do not need the actual meal scores. Even if C gives a lucky score of $10$, selecting C still contributes gap $8-5=3$: its expected quality is below B's.

Each gap is nonnegative, so this pseudo-regret cannot be negative. The realized quantity $Tq_*(a_*)-\sum_tR_t$ can be negative after unusually good rewards. The two quantities agree in expectation under the stationary model here, but not necessarily in one run.

The learner normally cannot calculate pseudo-regret because it lacks the true means. The simulator uses them only for evaluation, never for selecting actions.

### Why repeat experiments?

One learner may discover B early; another may see misleading first rewards. Compare strategies across independent runs and report their average performance.

| Metric | What it answers |
| --- | --- |
| Mean reward | How much did the strategy earn per pull? |
| Cumulative pseudo-regret | How much mean reward did its choices miss? |
| Fraction of optimal actions | How often did it select a true best arm? |
| Standard error of mean reward | How uncertain is the reported average across runs? |

If independent run-average rewards are $5,7,6$, their overall mean is $6$. The sample standard deviation is $1$, and its standard error is $1/\sqrt3\approx0.577$. Standard deviation describes variation among run averages; standard error describes uncertainty in their estimated mean. Neither is the spread of individual meal scores. Under comparable conditions, more independent runs usually reduce standard error.

Reusing seed schedules across strategies can correlate their results, so separate standard errors alone are not a test of the difference between strategies. Assess strategy differences with that paired structure in mind.

### Theory exercise 3C — Compute pseudo-regret

Arm means are $[0,0.2,1]$. The chosen arms, indexed from zero, are $[0,2,1,2]$. Calculate cumulative pseudo-regret.

**Solution**

The gaps are $1,0,0.8,0$, so cumulative pseudo-regret is $1.8$. No sampled rewards are needed for this calculation.



## Progressive theoretical exercises — Bandits

Work through these exercises **in order**, after reading the lesson. The earlier in-text exercises are warm-ups; this sequence gradually removes the scaffolding. No programming is needed. Write your reasoning as well as your final answer. Allow an additional 60–90 minutes, or split the sequence across study sessions. The final two exercises are stretch challenges: use their hints when needed. Try each question before reading its solution.

### Tools for the stretch exercises

Variance measures spread using squared distances from the mean; standard deviation is its square root, in the original units. If variance is $4$, standard deviation is $2$. For independent fixed-count samples, averaging $n$ observations reduces sample-mean variance to $\sigma^2/n$. Exercise 3T6 uses this to connect reward noise and the difficulty of separating close means.

A prior is a probability assignment before seeing the new observation. In 3T8, it describes uncertainty about an arm's fixed type. Updating that information after a pull can change which final action is best.

### Level 1 — Guided application

### 3T1 — Keep separate arm counts

Two arms begin with estimate zero. The observed sequence is $(A,4),(B,-2),(A,0),(A,2),(B,6)$.

After every pull, record both arm counts and sample-average estimates. Explain the error in using the total pull number as the denominator for each update.

**Hint**

Only the selected arm receives a new observation.



**Solution**

| Pull | $N(A)$ | $Q(A)$ | $N(B)$ | $Q(B)$ |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 4 | 0 | 0 |
| 2 | 1 | 4 | 1 | -2 |
| 3 | 2 | 2 | 1 | -2 |
| 4 | 3 | 2 | 1 | -2 |
| 5 | 3 | 2 | 2 | 2 |

Each estimate averages that arm's rewards only. A total-pull denominator counts observations belonging to other arms and no longer implements the sample average.



### 3T2 — Ties and exploration overlap

Five arms have estimates $[3,3,1,0,-1]$. Use $\epsilon=0.3$ with uniform selection among all arms during exploration and among tied maxima during exploitation.

Find each arm's selection probability, check their sum, and compute the probability of selecting an arm outside the greedy set.

**Hint**

An arm in the greedy set can be selected through either branch.



**Solution**

Each of the two maxima gets $(1-0.3)/2+0.3/5=0.41$. Each other arm gets $0.06$. The probabilities sum to $2(0.41)+3(0.06)=1$. Selection outside the greedy set has probability $3(0.06)=0.18$, not $0.3$. Exploration sometimes selects a greedy arm.



### Level 2 — Multi-step reasoning

### 3T3 — Reward noise does not define pseudo-regret

Arm means are $[1,0.5,-0.5]$. Four selected arms are $[0,2,1,0]$ and observed rewards are $[0,3,2,-1]$.

1. Compute cumulative pseudo-regret.
2. Compute $4q_*(a_*)-\sum_{t=1}^4R_t$.
3. Explain why the second quantity can differ from pseudo-regret or be negative, while pseudo-regret cannot be negative here.

**Hint**

Use means for pseudo-regret and observations only for the second calculation.



**Solution**

The chosen mean gaps are $0,1.5,0.5,0$, giving pseudo-regret $2$. The mean-based benchmark minus realized rewards is $4-(0+3+2-1)=0$. This second quantity contains reward noise and is not the defined pseudo-regret. If the observed reward sum exceeds $4$, it is negative. Pseudo-regret sums nonnegative mean gaps. Under the stated stationary model, taking expectations makes the two quantities agree, although individual runs need not agree.



### 3T4 — The permanent price of fixed epsilon

Assume estimates now identify a unique true best arm and keep doing so. The arm means are $[0,0.5,1.5]$ and $\epsilon=0.2$.

1. Compute expected reward and pseudo-regret per pull.
2. Compute expected cumulative pseudo-regret over $T$ further pulls.
3. Generalize using gaps $\Delta_a=q_*(a_*)-q_*(a)$ for $k$ arms.
4. Does the resulting expression include regret from mistaken estimates during early learning?

**Hint**

Exploitation has zero gap under the stated assumption; uniform exploration averages the arm gaps.



**Solution**

Expected reward is $0.8(1.5)+0.2(0+0.5+1.5)/3=4/3$. Expected gap is $1.5-4/3=1/6$, so $T$ further pulls incur expected pseudo-regret $T/6$. Generally, the expected per-pull gap is
$$\frac{\epsilon}{k}\sum_a\Delta_a,$$
and the cumulative expectation is $T$ times this quantity. This assumes correct exploitation throughout those pulls. Errors in the greedy estimate introduce an additional exploitation cost, so this is not a full finite-learning analysis.



### Level 3 — Derivation and critique

### 3T5 — Unroll a constant-step estimate

For $Q_n=(1-\alpha)Q_{n-1}+\alpha r_n$, with constant $0<\alpha<1$:

1. Expand $Q_1,Q_2,Q_3$ in terms of $Q_0$ and the rewards.
2. Derive the general reward weights in $Q_n$.
3. Show that the reward weights plus the initialization weight sum to one.
4. For $\alpha=0.2$, compare the weights of the newest reward and one observed three updates earlier.

**Hint**

Each subsequent update multiplies every earlier contribution by $1-\alpha$.



**Solution**

$Q_1=(1-\alpha)Q_0+\alpha r_1$. $Q_2=(1-\alpha)^2Q_0+\alpha(1-\alpha)r_1+\alpha r_2$. $Q_3=(1-\alpha)^3Q_0+\alpha(1-\alpha)^2r_1+\alpha(1-\alpha)r_2+\alpha r_3$. By repeated substitution,
$$Q_n=(1-\alpha)^nQ_0+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}r_i.$$
The geometric sum of reward weights is $1-(1-\alpha)^n$, leaving initialization weight $(1-\alpha)^n$; the total is one. The newest reward has weight $0.2$; a reward three updates older has weight $0.2(0.8)^3=0.1024$. Recent observations matter more, explaining responsiveness to changing means.



### 3T6 — Small gaps require more evidence

Two arms yield independent rewards with means $\mu_1>\mu_2$ and common variance $\sigma^2$. Collect exactly $n$ independent observations from each arm, independently across arms. Let $D=\overline R_1-\overline R_2$ and $\Delta=\mu_1-\mu_2$.

Use the supplied facts $\operatorname{Var}(\overline R)=\sigma^2/n$ and variance additivity for independent variables.

1. Find $\mathbb{E}[D]$ and $\operatorname{Var}(D)$.
2. The standard deviation is the square root of variance. Require it to be at most $\Delta/2$ and solve for $n$.
3. What happens to the required $n$ if the gap is halved? Why is this criterion not a guarantee of correct ranking?

**Hint**

The variance of a difference of independent variables is the sum of their variances.



**Solution**

$\mathbb{E}[D]=\Delta$ and $\operatorname{Var}(D)=2\sigma^2/n$. The condition $\sqrt{2\sigma^2/n}\leq\Delta/2$ gives $n\geq8\sigma^2/\Delta^2$ (round up for an integer sample count). Halving the gap requires four times as many samples by this criterion. A standard-deviation threshold describes scale of uncertainty, not certainty: noisy estimates can still reverse the ranking. This fixed-sample derivation also does not automatically describe adaptively selected sample counts.



### Level 4 — Synthesis challenges

### 3T7 — Can exploration disappear too quickly?

At pull $t\geq1$, independently choose the exploration branch with probability $\epsilon_t$, then choose uniformly among $k$ arms. Consider schedules A: $\epsilon_t=1/t$, and B: $\epsilon_t=1/t^2$.

1. Express the expected number of exploration selections of one fixed arm through $T$.
2. Use the facts that $\sum_{t=1}^{\infty}1/t$ diverges and $\sum_{t=1}^{\infty}1/t^2$ converges to compare the schedules.
3. With independent exploration draws, schedule A selects each arm through exploration infinitely often with probability one. Explain why schedule B lacks this guarantee.
4. Does infinite exploration imply zero cumulative regret?

**Hint**

For part 3, a nonnegative count with finite expectation cannot be infinite with positive probability. For A, the supplied independence condition is important.



**Solution**

The expectation for a fixed arm is $\frac1k\sum_{t=1}^T\epsilon_t$. For A this grows without bound; for B it remains bounded as $T\to\infty$. Under the supplied independent-draw fact, A ensures infinitely many exploration samples per arm. For B, total exploration has finite expectation, hence is finite with probability one; it cannot itself guarantee infinitely many samples per arm. Some arms might still be selected through exploitation, but poor early estimates can leave an arm neglected. Infinite exploration is not zero cumulative regret: every exploration selection of a suboptimal arm contributes a positive mean gap. Vanishing exploration frequency can reduce per-pull cost without eliminating accumulated cost. These observations alone are not a finite-time regret theorem for the complete learner.



### 3T8 — A finite-horizon information decision

There are exactly two pulls remaining and no discounting. Arm A always pays $0.6$. Arm B is deterministic but unknown: with prior probability $0.5$ it always pays $1$, otherwise it always pays $0$. One pull of B reveals its type perfectly.

1. Calculate expected total reward if you choose A first and then choose optimally on the last pull without any observation of B.
2. Calculate expected total reward if you choose B first and then use the revealed type optimally.
3. Which first action wins? What if only one pull remains?
4. Explain the role of information despite B's lower immediate expected reward.

**Hint**

After observing B, use it again if it is good; otherwise use A.



**Solution**

Choosing A first reveals nothing about B. On the final pull, A's mean $0.6$ exceeds B's prior mean $0.5$, so expected total is $1.2$. Choosing B first yields total $2$ in the good case and $0+0.6=0.6$ in the bad case. The expectation is $0.5(2)+0.5(0.6)=1.3$. Thus B wins with two pulls remaining; A wins with only one. The first pull of B buys information that improves the last choice enough to repay its immediate expected disadvantage. This is a reasoning exercise with an explicit prior, not the epsilon-greedy rule from the lab.




## 9. Common mistakes

Using total pulls in every arm's denominator produces incorrect estimates. Deterministically choosing the first tied arm introduces arbitrary preference. Evaluating only one seed can confuse luck with a reliable strategy. Using true means to choose actions gives the agent information it should not have. Treating $\epsilon=0.1$ as “choose a non-greedy arm 10% of the time” implements a different probability rule.

## 10. Mastery checkpoint

Explain without notes:

1. Why can pure greedy selection fail even after many pulls?
2. Why does epsilon-greedy's best-action probability exceed $1-\epsilon$?
3. What information does pseudo-regret require that a real learner may lack?
4. Why can constant step sizes help in a changing environment?
5. What does a bandit leave out compared with a full MDP?

**Next:** Lesson 4 returns to sequential decisions. Given a known model, we will improve an entire policy using Bellman backups. Later lessons will replace the known model with experience.
