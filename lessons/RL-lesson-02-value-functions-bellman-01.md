# Lesson 2 — Value Functions and the Bellman Equation — Version 2

> **Vocabulary:** Use the [project glossary](../docs/glossary.md) for the terms and notation shared by all lessons and code.

> **Version 2 — Expanded explanations and worked practical examples.**  
> Read each concept in words, then follow its numbers through the equation. The progressive theoretical exercises are retained here; the command-line labs and practical exercises are in the companion hands-on file.  
> **Study pace:** Allow 75–100 minutes for explanations and worked examples. The progressive theory sequence can be a separate 60–90 minute session.

> **Course:** Reinforcement Learning — Theory + Practice  
> **Lesson goal:** Learn how an agent can assign long-term value to states and actions.

**Hands-on companion:** [Python labs and practical exercises](RL-lesson-02-value-functions-bellman-02-hands-on.md)

### How to study this lesson

On a first pass, read the explanations and redo each worked example on paper. Then try the short in-text questions. Use the companion hands-on file for the command-line lab. Work through the progressive theory sequence in order; its final levels deliberately go beyond the first-pass material. An example shows the reasoning; an exercise asks you to reproduce or extend it.

## Learning objectives

By the end of this lesson, you should be able to:

-   Define the state-value function $V^\pi(s)$.
-   Define the action-value function $Q^\pi(s,a)$.
-   Explain why values depend on a policy.
-   Derive and interpret the Bellman expectation equation.
-   Estimate values from sampled episodes.
-   Explain iterative policy evaluation for a tiny MDP.

------------------------------------------------------------------------

## 1. From return to value: judging a future before it happens

A return describes the score of one future trajectory:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

Before choosing, we usually do not know which trajectory will happen. A courier may meet traffic; a robot may slip; a stochastic policy may choose different actions. A **value** is the expected return across those possible futures.

For example, a delivery plan has return $8$ on a clear day and $2$ on a delayed day. If each is equally likely, its value is

$$
0.5(8)+0.5(2)=5.
$$

One trip earns $8$ or $2$, while the plan's expected score is $5$. Keep this distinction in mind: **return is a trajectory's score; value is an average prediction before the future is known**.

In this lesson we evaluate a specified policy. We are not yet asking for the best possible policy.

## 2. State value: how good is starting here under this policy?

The state-value function is

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s].
$$

Read it as: **start in state $s$, follow policy $\pi$, and average the resulting returns**.

| Notation | Meaning |
| --- | --- |
| $V$ | A value for a state |
| Superscript $\pi$ | The policy we will follow; this is a label, not a power |
| $\mathbb E$ | Expected value: probability-weighted average |
| $\mid S_t=s$ | Given that we start from state $s$ |

### Worked example: two delivery habits

A courier at junction $s$ can take a direct route giving rewards $-1,10$, or a detour giving $-1,-1,10$. After choosing a route, there are no more decisions. Set $\gamma=0.9$.

| Policy at the junction | Calculation | $V^\pi(s)$ |
| --- | --- | ---: |
| Always direct | $-1+0.9(10)$ | $8$ |
| Always detour | $-1+0.9(-1)+0.9^2(10)$ | $6.2$ |
| Choose each route with probability $0.5$ | $0.5(8)+0.5(6.2)$ | $7.1$ |

The location is identical in all three rows. Its value changes because the future behavior changes. Value also depends on the environment, reward rule and discount. It is not a permanent score attached to a physical location.

For a terminal state, $V^\pi(\text{terminal})=0$: there are no rewards left to receive. This does not mean delivery was worthless. Its reward was received on the transition into that terminal state.

## 3. Action value: what if I choose this action first?

The action-value function is

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].
$$

It asks: **start at $s$, take action $a$ once, then follow policy $\pi$ for later decisions**. Fixing the first action lets us compare alternatives even if the current policy rarely uses them. Their consequences still depend on how we behave afterward.

In the delivery example, $Q^\pi(s,\text{direct})=8$ and $Q^\pi(s,\text{detour})=6.2$. These happen not to depend on the later policy because there are no later choices on those routes. If the courier later reaches another junction, the continuation policy matters again.

The state value averages the action values according to the policy:

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

Suppose the policy chooses direct with probability $0.75$:

$$
V^\pi(s)=0.75(8)+0.25(6.2)=7.55.
$$

$Q$ evaluates a specified first choice. $V$ evaluates the choice rule. Replacing the average by $\max_a Q^\pi(s,a)$ asks a different question: which first action looks best?

A greedy improvement chooses

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

$\max$ returns the largest number; $\arg\max$ returns the action or actions attaining it. Here the maximum is $8$ and the maximizing action is direct. Use $\in$ because ties may give more than one maximizing action. Lesson 4 explains improvement; Lesson 3 explains why uncertain estimates complicate choosing greedily.

## 4. The Bellman idea: split now from later

A long return can be divided into **the next reward** and **everything after it**:

$$
G_t=R_{t+1}+\gamma G_{t+1}.
$$

Why? Write out the future:

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
=R_{t+1}+\gamma(R_{t+2}+\gamma R_{t+3}+\cdots).
$$

The parentheses are exactly $G_{t+1}$. Multiplying them by $\gamma$ shifts all their rewards one step further away from decision $t$.

### Worked example: backward scoring

For a trip earning $-1,-1,10$ with $\gamma=0.9$, start at the end:

| Decision | Reward next | Return afterward | Return here |
| --- | ---: | ---: | --- |
| $t=2$ | $10$ | $G_3=0$ | $G_2=10+0.9(0)=10$ |
| $t=1$ | $-1$ | $G_2=10$ | $G_1=-1+0.9(10)=8$ |
| $t=0$ | $-1$ | $G_1=8$ | $G_0=-1+0.9(8)=6.2$ |

This agrees with adding the discounted rewards directly. The recursive form lets us reuse the future's score instead of listing every later reward each time.

For a sampled trajectory, this identity uses its actual future return $G_{t+1}$. In a value equation, we average over futures and use the next state's expected value instead.

## 5. Bellman expectation: average the next reward plus continuation

For a Markov state and a stationary policy, future expected return after the next transition is represented by the next state's value. Thus

$$
V^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s].
$$

The **Bellman expectation equation** says: “my value here equals the average of reward now plus discounted value where I go next.”

For a discrete MDP, writing out both sources of randomness gives

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')].
$$

There are two averages. First, for a particular action, average its possible environmental outcomes using $p(s',r\mid s,a)$. Then average those action scores using the policy's action probabilities $\pi(a\mid s)$.

### Worked example: the courier chooses a route, then traffic decides an outcome

Set $\gamma=0.9$. At a junction, the policy chooses the main road with probability $0.75$ and a side road with probability $0.25$. Suppose the values under the same policy are $V^\pi(x)=10$ and $V^\pi(y)=4$.

| Action | Outcome probability given this action | Reward $r$ | Next state | $r+0.9V^\pi(s')$ |
| --- | ---: | ---: | --- | ---: |
| Main road | $0.8$ | $-1$ | $x$ | $-1+0.9(10)=8$ |
| Main road | $0.2$ | $-3$ | $y$ | $-3+0.9(4)=0.6$ |
| Side road | $1$ | $-2$ | $y$ | $-2+0.9(4)=1.6$ |

First average outcomes for each action:

$$
Q^\pi(s,\text{main})=0.8(8)+0.2(0.6)=6.52,
\qquad Q^\pi(s,\text{side})=1.6.
$$

Then average actions:

$$
V^\pi(s)=0.75(6.52)+0.25(1.6)=5.29.
$$

Traffic probabilities answer “what happens if I choose main?” Policy probabilities answer “how often do I choose main?” They play different roles, and each relevant probability group sums to one.

We can also write the action backup directly:

$$
Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')].
$$

It has no average over the first action because that action is already fixed. Every terminal outcome contributes its reward with zero continuation.

If $x$ could lead back to $s$, their values would depend on each other. Bellman equations then form a set of simultaneous consistency equations. Solving them or repeatedly updating estimates is how we obtain the values; they are not additional rewards handed to the agent.

## 6. Theory exercise --- Solve values by hand

Consider the deterministic MDP:

``` text
A --right--> B --right--> TERMINAL
```

Rewards:

``` text
A -> B        : +1
B -> terminal : +4
```

The policy always chooses `right`.

Let

$$
\gamma=0.5.
$$

You can picture A as a pickup point and B as a delivery point. The pickup earns $1$; completing delivery earns $4$. Starting at B has only the delivery reward left. Starting at A has both rewards left, but delivery is one step further away. This is why we work backward from the terminal value zero.

### Exercise 2A

Calculate:

$$
V^\pi(B)
$$

and

$$
V^\pi(A).
$$

**Solution**

From B:

$$
V^\pi(B)=4.
$$

From A:

$$
V^\pi(A)
=
1+\gamma V^\pi(B).
$$

Therefore,

$$
V^\pi(A)
=
1+0.5(4)
=
3.
$$


### Exercise 2B

Repeat for:

1.  $\gamma=0$
2.  $\gamma=0.9$
3.  $\gamma=1$

Explain why $V^\pi(B)$ does not change in this example while $V^\pi(A)$ does.

------------------------------------------------------------------------

## 7. A stochastic example

Suppose state $S$ has only one action.

After taking it:

-   with probability $0.8$, you receive `+2` and terminate;
-   with probability $0.2$, you receive `-1` and terminate.

Because there is no future reward after termination,

$$
V(S)
=
0.8(2)+0.2(-1).
$$

Thus,

$$
V(S)=1.4.
$$

For a practical interpretation, imagine a one-shot inspection: a satisfactory parcel earns $2$ and a damaged parcel earns $-1$. The expected score $1.4$ is an average over inspections, not a possible individual score. When an outcome instead leads to another job at X, its future earning opportunity must be included too.

### Exercise 2C

Now modify the first outcome:

-   probability $0.8$: reward `+2`, transition to state $X$;
-   probability $0.2$: reward `-1`, terminate.

Suppose

$$
V(X)=5,\qquad \gamma=0.9.
$$

Calculate $V(S)$.

**Solution**

$$
V(S)
=
0.8[2+0.9(5)]
+
0.2[-1].
$$

Therefore,

$$
V(S)=0.8(6.5)-0.2=5.
$$



------------------------------------------------------------------------

## 8. Estimate values from experience

We can calculate values from a known model, or estimate them by observing what happens. Think of estimating the value of a delivery strategy using either a complete route-and-traffic model or a log of completed trips.

| Approach | What we have | What we do |
| --- | --- | --- |
| Model-based evaluation | Rewards and transition probabilities | Compute probability-weighted Bellman backups |
| Experience-based evaluation | Trajectories from following the policy | Calculate and average observed returns |

### Monte Carlo intuition: average completed-trip scores

**Monte Carlo (MC)** methods use sampled outcomes. If independent completed trips from a state give returns $8,2,5,9$, their average is

$$
\widehat V(s)=\frac{8+2+5+9}{4}=6.
$$

In general,

$$
\widehat V(s)=\frac1N\sum_{i=1}^N G^{(i)}.
$$

$N$ is the number of sampled returns, and $G^{(i)}$ is the return on sample $i$, not a power of $G$. The hat on $\widehat V$ means estimate. Four trips do not establish that the true value is exactly $6$; different samples can give a different mean.

Returns must come from the policy whose value we want. Mixing trips from different route-choice rules does not generally estimate one fixed policy's value.

If we stop every trip before delivery, collecting more such records will not restore the missing rewards. For the earlier deterministic trip $-1,-1,10$, a cutoff after two rewards records $-1+0.9(-1)=-1.9$, while the full return is $6.2$. Repeating the same cutoff infinitely often still records $-1.9$. This is missing continuation, not bad luck.

**Every-visit** MC works as follows: if an episode visits a state more than once, each visit provides a suffix return starting there. Those returns share parts of the same trip and are correlated. The advanced exercises explore both this dependence and truncation bias.

**Time-limit caveat:** using zero continuation at a truncated trajectory's end omits later rewards and can bias estimates of the full-episode value. Simply discarding all truncated episodes can also bias estimates by selecting shorter trajectories. Later, TD learning will give us a principled way to bootstrap across external limits. If time remaining is part of the task itself, it belongs in the state and you must evaluate that finite-horizon task instead.

## 9. Iterative policy evaluation

Rather than wait for trips, a known model lets us repeatedly revise a table of values. A **Bellman backup** calculates reward plus discounted next-state estimates and averages according to the fixed policy. A **sweep** performs one such update for every state.

For example, a delivery route has A to B with reward $1$, and B to terminal with reward $4$. Set $\gamma=0.5$ and start with estimates $V_0(A)=V_0(B)=0$.

| Sweep | Estimate for A | Estimate for B |
| --- | --- | --- |
| Initial | $0$ | $0$ |
| First | $1+0.5(0)=1$ | $4+0.5(0)=4$ |
| Second | $1+0.5(4)=3$ | $4$ |
| Third | $3$ | $4$ |

The delivery benefit reaches A on the second sweep. These are revisions to predictions, not trips through the environment. The policy stays fixed throughout evaluation.

With **synchronous** sweeps, every new value reads from the previous table. Using B's newly updated value immediately when updating A would be an in-place method with different intermediate tables.

A common stopping rule stops when the largest change in a sweep is below $\theta$:

$$
\max_s|V_{k+1}(s)-V_k(s)|<\theta.
$$

Here $k$ counts sweeps, $|\cdot|$ means absolute difference, and $\max_s$ finds the largest difference over states. For changes $0.02,0.005,0$, the largest is $0.02$; a tolerance of $0.01$ would not stop yet.

In a finite MDP with bounded rewards and $\gamma<1$, discounted policy backups converge, including for policies that loop forever. A policy earning $-1$ forever has value $-1/(1-\gamma)$, even though it never reaches a goal. This discounted DP setting assumes $\gamma<1$; allowing $\gamma=1$ needs further assumptions.

A small sweep change is a stopping signal rather than an exact value-error measurement. If a value table's **Bellman residual**, its largest disagreement with its own backup, is $b$, then

$$
\|V-V^\pi\|_\infty\leq\frac{b}{1-\gamma}.
$$

The maximum norm $\|\cdot\|_\infty$ means the largest absolute entry. A residual $0.01$ gives an error bound $0.1$ at $\gamma=0.9$, but $1$ at $\gamma=0.99$. These are worst-case bounds, not exact errors. The stretch exercises derive them.

### Theory exercises

1. Derive $V(3)$ under the random policy. Check that $V(3)=0.5[-1+\gamma V(2)]+0.5(10)$.
2. Calculate all values for the always-right policy by hand.
3. Consider a policy that always moves left. Why can discounted policy evaluation compute values although the goal is never reached? Why does finite truncated MC estimate a different quantity?

------------------------------------------------------------------------

## 10. Theory exercise --- $V$ versus $Q$

Suppose a state has two available actions:

$$
Q^\pi(s,a_1)=8
$$

and

$$
Q^\pi(s,a_2)=2.
$$

The current policy selects

$$
\pi(a_1\mid s)=0.25
$$

and

$$
\pi(a_2\mid s)=0.75.
$$

Calculate $V^\pi(s)$.

Hint:

$$
V^\pi(s)
=
\sum_a
\pi(a\mid s)Q^\pi(s,a).
$$

**Solution**

$$
V^\pi(s)
=
0.25(8)+0.75(2)
=
3.5.
$$

Notice that the best available action has value 8, but the state value under the **current policy** is only 3.5 because that policy usually chooses the poorer action.


This distinction prepares us for **policy improvement**.

------------------------------------------------------------------------

## 11. Challenge --- Improve the policy

Return to the five-state LineWorld.

You now have estimated $Q(s,a)$ values---or can reason about them.

Construct a new policy:

$$
\pi'(s)\in\arg\max_a Q(s,a).
$$

Then reason about how the new policy compares with the random policy.

### Think carefully

If the estimates of $Q$ are imperfect, is it always safe to stop exploring and choose the largest estimate forever?

This question leads directly toward bandits, exploration strategies, Monte Carlo control, temporal-difference learning, and eventually Q-learning.

------------------------------------------------------------------------

## Progressive theoretical exercises — Values and Bellman equations

Work through these exercises **in order**, after reading the lesson. The earlier in-text exercises are warm-ups; this sequence gradually removes the scaffolding. No programming is needed. Write your reasoning as well as your final answer. Allow an additional 60–90 minutes, or split the sequence across study sessions. The final two exercises are stretch challenges: use their hints when needed. Try each question before reading its solution.

### Tools for the stretch exercises

A self-loop means returning to the same state. If waiting earns $1$ and returns here, with $\gamma=0.5$, Bellman gives $V=1+0.5V$. Subtract $0.5V$ from both sides to obtain $V=2$. A variable appearing on both sides is a consistency equation, not a mistake.

An operator $T_\pi$ is a function taking an entire value table and returning its policy-backup table. “Fixed point” means $V=T_\pi V$: the table already agrees with its backup. A residual measures disagreement. A contraction shrinks differences between tables. Exercises 2T7–2T8 use these ideas to justify uniqueness and error bounds.

### Level 1 — Guided application

### 2T1 — Backward returns

An episode has rewards $-2,3,4$ and then terminates. Set $\gamma=0.5$.

Calculate $G_2,G_1,G_0$ backward using the recursive return identity. Explain why this single trajectory does not necessarily give $V^\pi(S_0)$.

**Hint**

Begin with terminal continuation $G_3=0$.



**Solution**

$G_2=4$, $G_1=3+0.5(4)=5$, and $G_0=-2+0.5(5)=0.5$. This is one realized return. $V^\pi(S_0)$ is an expectation over the future trajectories generated by the policy and environment. They coincide only when the return is deterministic or by coincidence.



### 2T2 — Separate action and outcome uncertainty

At state $s$, policy $\pi$ chooses A with probability $0.4$ and B with probability $0.6$. A earns $1$ and reaches $x$ with probability $0.5$, or earns $-1$ and terminates with probability $0.5$. B earns $2$ and reaches $x$ surely. Suppose $V^\pi(x)=4$ and $\gamma=0.5$.

Compute both action values and then $V^\pi(s)$.

**Hint**

First average outcomes within each action, then average actions under the policy.



**Solution**

$Q^\pi(s,A)=0.5[1+0.5(4)]+0.5(-1)=1$. $Q^\pi(s,B)=2+0.5(4)=4$. $V^\pi(s)=0.4(1)+0.6(4)=2.8$. Maximizing over random outcomes would be incorrect; the policy cannot choose which outcome occurs.



### Level 2 — Multi-step reasoning

### 2T3 — A self-loop with possible termination

There is one nonterminal state $s$ and one action. With probability $p$, the action earns $5$ and terminates. Otherwise it earns $-1$ and returns to $s$.

1. Derive $V(s)$ for $0\leq p\leq1$ and $0\leq\gamma<1$.
2. Evaluate at $p=0.2$, $\gamma=0.9$.
3. Evaluate at $p=0$ with $\gamma<1$, and explain what fails if additionally $\gamma=1$.

**Hint**

The value on the self-loop appears on both sides of the Bellman equation.



**Solution**

$V=5p+(1-p)[-1+\gamma V]$, so
$$V=\frac{6p-1}{1-\gamma(1-p)}.$$
At $p=0.2$, $\gamma=0.9$, this is $0.2/0.28=5/7$. At $p=0$, it is $-1/(1-\gamma)$. Discounting gives a finite value even without termination. At $p=0$, $\gamma=1$, every reward is $-1$ forever and return diverges to $-\infty$; there is no finite solution.



### 2T4 — Solve coupled values

A fixed policy produces deterministic transitions $A\to B$ with reward $1$ and $B\to A$ with reward $2$. There is no terminal state. Let $\gamma=0.5$.

1. Write and solve the two Bellman equations.
2. Verify $V(A)$ directly from the infinite reward sequence.
3. Why does nontermination not prevent finite discounted values?

**Hint**

Substitute $V(B)=2+0.5V(A)$ into the equation for $V(A)$.



**Solution**

$V(A)=1+0.5V(B)$ and $V(B)=2+0.5V(A)$ yield $V(A)=8/3$ and $V(B)=10/3$. Starting at A, rewards alternate $1,2,1,2,\ldots$:
$$G_0=(1+0.5\cdot2)\sum_{j=0}^{\infty}(0.5^2)^j=\frac{2}{1-0.25}=\frac83.$$
Bounded rewards with discount below one have an absolutely convergent return, even on an infinite trajectory.



### Level 3 — Derivation and critique

### 2T5 — Derive the action-value Bellman equation

Starting with $Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$ and the recursive return identity, derive an equation using $p(s',r\mid s,a)$ and $\pi(a'\mid s')$. Explain where the Markov assumption is used and how to handle terminal outcomes.

**Hint**

After the first transition, average the next action under $\pi$.



**Solution**

Condition on the first transition and use the Markov state to represent continuation by the next state's value:
$$Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V^\pi(s')].$$
Since $V^\pi(s')=\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')$,
$$Q^\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].$$
For terminal $s'$, define the continuation term as zero; no next action is taken. The Markov property ensures that the conditional future distribution depends on the next state and the future policy, rather than on omitted earlier history.



### 2T6 — Sampling error versus truncation bias

A state loops deterministically forever, earning $1$ on each transition. Let $\gamma=0.9$. Each sampled run starts at that state and is interrupted after exactly $H=3$ transitions.

1. Calculate the true infinite-horizon value and the observed truncated return from the starting visit.
2. Can averaging a million such starting-visit returns remove the error?
3. What happens if every-visit MC averages all suffix returns within these three-transition runs, as the lesson's lab does?

**Hint**

The three recorded visits have different numbers of observed future rewards.



**Solution**

The true value is $1/(1-0.9)=10$. Each starting-visit truncated return is $1+0.9+0.9^2=2.71$, an underestimate of $7.29$. All runs are identical, so more runs cannot remove missing continuation. This is bias from the observation limit, not sampling noise. Every-visit MC averages suffix returns $2.71,1.9,1$, giving $5.61/3=1.87$. Later visits are closer to the observation limit, so their recorded suffixes are even shorter. For a task with a genuine deadline, time remaining would instead be part of the state and these visits would represent different states.



### Level 4 — Synthesis challenges

### 2T7 — Uniqueness of a discounted policy value

Let $T_\pi$ denote the Bellman expectation backup for a fixed policy. The maximum norm is $\|U\|_\infty=\max_s|U(s)|$.

For a finite MDP, prove
$$\|T_\pi V-T_\pi W\|_\infty\leq\gamma\|V-W\|_\infty.$$
Then show that two finite solutions of the same policy Bellman equation must coincide when $\gamma<1$.

**Hint**

Reward terms cancel. A probability-weighted average of absolute differences cannot exceed their maximum.



**Solution**

For each nonterminal state,
$$|(T_\pi V)(s)-(T_\pi W)(s)|\leq\gamma\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)|V(s')-W(s')|\leq\gamma\|V-W\|_\infty.$$
Terminal continuations are fixed at zero. Taking the maximum proves the bound. If $V=T_\pi V$ and $W=T_\pi W$, write $d=\|V-W\|_\infty$. Then $d\leq\gamma d$, implying $(1-\gamma)d\leq0$. Since $d\geq0$ and $1-\gamma>0$, $d=0$, so $V=W$. This proves uniqueness; existence follows from convergence of repeated discounted backups.



### 2T8 — Certify an approximate value

Suppose an approximate vector $\widehat V$ has residual $b=\|\widehat V-T_\pi\widehat V\|_\infty$.

1. Using 2T7, derive $\|\widehat V-V^\pi\|_\infty\leq b/(1-\gamma)$.
2. If $b=0.02$, compare the guarantees for $\gamma=0.5$ and $0.99$.
3. Why does a small backup residual need interpretation in light of the discount?

**Hint**

Insert $T_\pi\widehat V$ between $\widehat V$ and $V^\pi=T_\pi V^\pi$, then use the triangle inequality.



**Solution**

Let $e=\|\widehat V-V^\pi\|_\infty$. Then
$$e\leq\|\widehat V-T_\pi\widehat V\|_\infty+\|T_\pi\widehat V-T_\pi V^\pi\|_\infty\leq b+\gamma e.$$
Rearrangement gives $e\leq b/(1-\gamma)$. The bounds are $0.04$ at $\gamma=0.5$ and $2$ at $0.99$. These are upper bounds, not exact errors. A discount near one permits small local inconsistencies to accumulate over a long effective horizon.




## 12. Key ideas

The state-value function is

$$
\boxed{
V^\pi(s)
=
\mathbb{E}_\pi[G_t\mid S_t=s]
}
$$

The action-value function is

$$
\boxed{
Q^\pi(s,a)
=
\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
}
$$

Return obeys

$$
\boxed{
G_t=R_{t+1}+\gamma G_{t+1}
}
$$

which leads to the Bellman relationship

$$
\boxed{
V^\pi(s)
=
\mathbb{E}_\pi
[
R_{t+1}+\gamma V^\pi(S_{t+1})
\mid S_t=s
]
}
$$

The crucial conceptual leap is that **long-term value can be decomposed recursively**.

That idea appears throughout reinforcement learning.

------------------------------------------------------------------------

## 13. Mastery checkpoint

Before moving on, you should be able to answer:

1.  Why does $V^\pi$ include the policy in its notation?
2.  What is the difference between $V^\pi(s)$ and $Q^\pi(s,a)$?
3.  How does the Bellman equation exploit the recursive structure of return?
4.  What is the difference between estimating value from sampled experience and computing it from a known model?
5.  Why might a greedy policy based on inaccurate $Q$ estimates fail?

### Suggested next lesson

**Lesson 3 --- Multi-Armed Bandits and Exploration vs. Exploitation**

This isolates one of RL's central dilemmas before we return to full sequential environments.
