# Lesson 4 — Improving Policies with Dynamic Programming — Version 2

> **Version 2 — Expanded explanations and worked practical examples.**  
> Read each concept in words, then follow its numbers through the equation. The progressive theoretical exercises are retained here; the command-line labs and practical exercises are in the companion hands-on file.  
> **Study pace:** Allow 75–100 minutes for explanations and worked examples. The progressive theory sequence can be a separate 60–90 minute session.

> **Course:** Reinforcement Learning — Theory + Practice  
> **Prerequisites:** Bellman expectation equations from Lesson 2 and exploration from Lesson 3.  
> **Goal:** Turn value estimates into improved policies using a known MDP model.  

**Hands-on companion:** [Python labs and practical exercises](RL-lesson-04-dynamic-programming-control-hands-on.md)

### How to study this lesson

On a first pass, read the explanations and redo each worked example on paper. Then try the short in-text questions. Use the companion hands-on file for the command-line lab. Work through the progressive theory sequence in order; its final levels deliberately go beyond the first-pass material. An example shows the reasoning; an exercise asks you to reproduce or extend it.

## 1. From prediction to control: choose a better plan

Lesson 2 asked, “what return should I expect if I follow this policy?” That is **prediction**, or policy evaluation. Here we ask, “which policy should I follow?” That is **control**.

Think of a courier with a complete map. The map tells us the outcome and cost of every route choice. We can compare plans before driving them. Using a known model to plan this way is **dynamic programming (DP)**: we solve a long decision problem by repeatedly combining immediate rewards with values of later situations.

DP reuses a table of state values, so different routes can share the same computed continuation. It does not need to sample trips for these updates. Later, experience-based methods will address unknown models.

By the end, you should be able to use one-step lookahead, improve a policy, distinguish policy iteration from value iteration, and interpret convergence checks. The companion hands-on lab implements both algorithms through a CLI.

## 2. The environment and assumptions

We return to the five-stop courier route, called LineWorld, with the same rules as Lessons 1–2:

| Item | Definition |
| --- | --- |
| States | Positions $0,1,2,3,4$ |
| Start | Position $0$ |
| Actions | Left or right at nonterminal positions |
| Left boundary | Left at $0$ stays at $0$ |
| Goal | Position $4$; terminal |
| Reward | $-1$ when the transition does not reach the goal; $+10$ on entry to the goal |
| Discount | $\gamma=0.9$ by default |
| Terminal value | $0$ |

For example, right at $2$ gives $-1$ and moves to $3$. Right at $3$ gives $10$ and ends the episode. Waiting at the left boundary still costs $-1$. The final reward is received once on entry, with no future decisions at $4$.

The general theory here assumes **finitely many states and actions, bounded rewards, and $0\leq\gamma<1$**. Discounting gives convergence guarantees even for policies that can loop forever. Undiscounted episodic tasks need additional assumptions.

A known model supplies $p(s',r\mid s,a)$: the probability of each next-state-and-reward outcome after an action. The courier lab is deterministic, with one outcome of probability one per state-action pair. With random delays or slips, we would average the possible outcomes. Knowing a model means we can calculate consequences of an action even if the agent has never physically tried it.

## 3. One-step lookahead: score a choice using its destination

Suppose we have a table $V$ of continuation estimates. To score action $a$ at $s$, combine its immediate reward with the discounted value of where it leads:

$$
q_V(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')].
$$

Read this as: **for each possible outcome, calculate reward plus discounted continuation, then average the results**. The subscript $V$ says which value table is being used.

For a deterministic transition, there is only one outcome:

$$
q_V(s,a)=r(s,a)+\gamma V(s').
$$

### Worked example: two destinations

At a junction, left costs $-1$ and reaches a place currently valued at $2$. Right also costs $-1$ but reaches a place valued at $8$. With $\gamma=0.9$:

| Action | Reward | Destination value | Lookahead score |
| --- | ---: | ---: | --- |
| Left | $-1$ | $2$ | $-1+0.9(2)=0.8$ |
| Right | $-1$ | $8$ | $-1+0.9(8)=6.2$ |

Right looks better even though the immediate rewards match. One-step lookahead can include a long future because $V(s')$ summarizes that future.

If $V=V^\pi$, then $q_V=Q^\pi$: take this action once and follow $\pi$ afterward. If $V$ is an arbitrary estimate, $q_V$ is only a score based on that table; it need not equal any exact policy's action value.

### Worked example: a risky shortcut

A shortcut gives reward $-1$. With probability $0.8$ it reaches a destination valued at $8$; with probability $0.2$ it reaches one valued at $2$:

$$
q_V(s,\text{shortcut})=0.8[-1+0.9(8)]+0.2[-1+0.9(2)]
=0.8(6.2)+0.2(0.8)=5.12.
$$

A safe route costing $-1$ and reaching a state valued at $6$ scores $-1+0.9(6)=4.4$. The shortcut wins this expected-return comparison. We average its outcomes before comparing actions: the courier controls the route, not whether the shortcut succeeds.

For a terminal destination, continuation is zero. A final delivery reward of $10$ gives $10+0.9(0)=10$, never $10+0.9(10)$.

### Theory exercise 4A — Look ahead

Suppose current values satisfy $V(1)=2$, $V(3)=8$ and $\gamma=0.9$. At position $2$, both actions earn $-1$. Compute their backed-up values and choose a greedy action.

**Solution**

Left: $-1+0.9(2)=0.8$. Right: $-1+0.9(8)=6.2$. Choose right. These are lookahead values based on the supplied $V$; they are true $Q^\pi$ values only if the supplied vector is $V^\pi$.



## 4. Policy improvement: replace a weaker decision rule

First evaluate a policy exactly, obtaining $V^\pi$. Then score each action using that continuation and choose a maximizing action:

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

### Worked example: changing the courier's habit

Suppose two action values at a junction are $8$ and $2$. The current policy chooses the first with probability $0.25$ and the second with probability $0.75$. Its state value is

$$
V^\pi(s)=0.25(8)+0.75(2)=3.5.
$$

Choosing the first action gives one-step lookahead $8$, rather than the old average $3.5$. This motivates greedy improvement: an average cannot exceed its largest entry.

Formally,

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)\leq\max_aQ^\pi(s,a).
$$

The policy improvement theorem extends this first-decision comparison to using the improved rule at every later decision:

$$
V^{\pi'}(s)\geq V^\pi(s)\quad\text{for every state }s.
$$

Here is the reasoning in words. First, the new rule's next action is at least as good when we score continuation using the old policy. Second, averaging larger continuation values cannot make a backup smaller. Repeating backups for the new policy therefore preserves the improvement; discounted convergence reaches its actual value $V^{\pi'}$. Exercise 4T5 writes out the proof.

$Q^\pi(s,a)=8$ still means follow the **old** policy after the first action. The new policy can change later decisions too, so its complete value need not be exactly $8$. It can be higher. Improvement guarantees nondecrease, not strict increase at every state.

### Why accurate evaluation matters

If true action values are $8$ and $7.9$, estimates $7.8$ and $8.1$ reverse their ordering. A greedy decision based on those estimates can select the worse action. The exact theorem assumes exact evaluation; numerical and noisy estimates require care near ties.

### Theory exercise 4B — Evaluation is not improvement

At a state, the old policy chooses actions with values $8$ and $2$ with probabilities $0.25$ and $0.75$. Compute its state value. What one-step value does a greedy improvement choose? Is that automatically the complete value under the new policy?

**Solution**

The old state value is $0.25(8)+0.75(2)=3.5$. Greedy lookahead chooses $8$. That is $Q^\pi(s,a)$: it assumes the old policy after the first action. The new policy can change later actions too, so its complete state value need not equal $8$. Under exact greedy improvement, it is at least as large.



## 5. Policy iteration: alternate evaluation and improvement

**Policy iteration** repeats two jobs:

1. Evaluate the current policy to obtain $V^\pi$.
2. Improve the policy by choosing greedy actions using those values.

Repeat until the policy stops changing. Evaluation asks how good the present rule is; improvement changes the rule.

### Worked example: improvements can spread across rounds

A courier at start $s$ can cash out for $3$, ending the task, or continue to $x$ with reward $-1$. At $x$, it can quit for $0$ or finish for $10$, both terminal. Set $\gamma=0.9$.

| Round | Policy at $s$ | Policy at $x$ | Evaluated $V(s)$ | Evaluated $V(x)$ |
| --- | --- | --- | ---: | ---: |
| Initial | Cash out | Quit | $3$ | $0$ |
| After first improvement | Cash out | Finish | $3$ | $10$ |
| After second improvement | Continue | Finish | $8$ | $10$ |

On the first improvement, continuing at $s$ scores $-1+0.9(0)=-1$, because lookahead uses the old quit policy at $x$. Cash out remains best at $s$. At $x$, finishing wins immediately.

After evaluating that improved policy, $V(x)=10$. The next improvement sees continue's score $-1+0.9(10)=8>3$. Now it changes $s$ too. Evaluate again; the policy is stable.

The model can evaluate actions at $x$ even though the initial policy cashes out before visiting it. This is why planning does not need exploration to discover model-supplied outcomes.

In a finite discounted MDP, exact policy iteration reaches an optimal policy when ties are handled consistently. Switching between equally good actions can change the policy without changing its value.

## 6. Bellman optimality and value iteration

The **optimal value** is the best expected return achievable from a state:

$$
V_*(s)=\max_\pi V^\pi(s).
$$

The star means optimal. It differs from $V^\pi$, which scores a specified policy. The optimal action value $Q_*(s,a)$ takes action $a$ first and then follows an optimal continuation; $Q^\pi$ instead follows the specified $\pi$.

If we already had the optimal continuation table, the best action here would give

$$
V_*(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V_*(s')].
$$

This is the **Bellman optimality equation**. Compare it with policy evaluation: average environmental outcomes in both cases, but here choose the largest action score instead of averaging actions under a fixed policy. The agent can choose its action, not its random outcome.

### Value iteration: improve the table one sweep at a time

Start with any finite table, often zeros, and repeatedly apply

$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V_k(s')].
$$

$k$ counts computational sweeps, not steps in an episode. Each synchronous sweep reads the previous table, considers every action and stores the highest score. After convergence, extract a greedy policy from the final table. There is no separate complete evaluation of every intermediate policy.

### Worked example: the delivery reward travels backward

For default LineWorld with $\gamma=0.9$:

| Sweep | $V(0)$ | $V(1)$ | $V(2)$ | $V(3)$ | $V(4)$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | $0$ | $0$ | $0$ | $0$ | $0$ |
| First | $-1$ | $-1$ | $-1$ | $10$ | $0$ |
| Second | $-1.9$ | $-1.9$ | $8$ | $10$ | $0$ |
| Third | $-2.71$ | $6.2$ | $8$ | $10$ | $0$ |
| Fourth | $4.58$ | $6.2$ | $8$ | $10$ | $0$ |

On the first sweep, only state $3$ can see the goal reward directly. On the second, state $2$ compares left's $-1+0.9(-1)=-1.9$ with right's $-1+0.9(10)=8$. On the third, state $1$ uses the previous value $8$ to get $6.2$. On the fourth, state $0$ uses $6.2$ to get $4.58$.

The benefit spreads one transition backward per synchronous sweep. This example reaches an exact fixed point after four sweeps; general MDPs can require many more and converge gradually.

| Method | What one update does | How policy is handled |
| --- | --- | --- |
| Policy evaluation | Average action scores under a fixed policy | Keep the policy fixed |
| Policy iteration | Evaluate, then greedily improve | Keep an explicit policy between rounds |
| Value iteration | Keep the highest action score each sweep | Extract a greedy policy from final values |

### Why does repeating backups converge?

Write $T_*V$ for the table produced by one optimality sweep. In a finite discounted MDP,

$$
\|T_*V-T_*W\|_\infty\leq\gamma\|V-W\|_\infty.
$$

The maximum norm $\|V-W\|_\infty$ is the largest absolute difference between the two tables. The inequality says the backup cannot increase that largest difference; it shrinks it by at least a factor $\gamma$ when $\gamma<1$. This property is called a **contraction**.

If two tables differ by at most $2$ and $\gamma=0.9$, their backed-up tables differ by at most $1.8$. Compared with the fixed optimal table, repeated sweeps shrink the bound to $2(0.9)^k$ after $k$ sweeps. These are upper bounds, not a promise that every entry changes by that exact amount.

Discounts closer to one shrink the bound more slowly, which can make convergence slower. The stretch exercises derive this property and connect numerical residuals to value and policy accuracy.

### Theory exercise 4C — First value-iteration sweep

Start with $V_0(s)=0$ everywhere. Compute $V_1$ for the default five-state LineWorld. Then calculate $V_2(2)$.

**Solution**

$V_1=[-1,-1,-1,10,0]$. At state $2$, the next sweep compares left, $-1+0.9(-1)=-1.9$, with right, $-1+0.9(10)=8$, so $V_2(2)=8$. The good terminal reward begins propagating backward.



## 7. LineWorld values and numerical accuracy

For the default five-state LineWorld, the optimal values and actions are:

| State | Optimal value | Action |
| --- | ---: | --- |
| 0 | 4.58 | Right |
| 1 | 6.20 | Right |
| 2 | 8.00 | Right |
| 3 | 10.00 | Right |
| 4 | 0.00 | Terminal |

For a state $d\geq1$ moves from the goal, following right gives

$$
V(s)=-\sum_{j=0}^{d-2}\gamma^j+10\gamma^{d-1}.
$$

For $d=1$, the sum is empty and equals zero. At the default discount this gives the table above. Right is uniquely optimal here. At $\gamma=0$, actions away from the goal can tie because only their identical immediate rewards matter; the tie rule can then select left. A discounted-optimal policy at $\gamma=0$ need not reach the goal.

### Reading the distance formula as a delivery bill

If the courier is three moves from the goal, it pays two ordinary movement costs and then earns the goal reward. The formula above becomes

$$
V(s)=-(1+\gamma)+10\gamma^2.
$$

At $\gamma=0.9$, this is $-1-0.9+8.1=6.2$. There are $d-1$ ordinary-cost rewards and one delivery reward. The delivery reward is the $d$-th reward, so its discount exponent is $d-1$. For $d=1$, there is no ordinary movement cost; value is simply $10$.

### Bellman optimality residual

The **Bellman optimality residual** measures disagreement with the optimality backup:

$$
b=\|V-T_*V\|_\infty.
$$

For the discounted setting,

$$
\|V-V_*\|_\infty\leq\frac{b}{1-\gamma}.
$$

A small residual supports accuracy of the values; it is not, by itself, a claim of exact floating-point equality. For example, $b=10^{-6}$ and $\gamma=0.99$ give a maximum value-error bound of $10^{-4}$. This bound concerns values, not an exact guarantee that every selected action matches a particular optimal policy when actions nearly tie.

## 9. A control-design challenge

Design a tiny MDP in which an immediate attractive reward leads to a worse long-term result. Keep at most four states and two actions per nonterminal state.

Specify all transition probabilities, rewards, terminal conditions and discount. Calculate $Q^\pi$ for an initial policy, improve it once by hand, and calculate the new value. Your submission should include a state/action table and your calculations.

One possible starting point: from a start state, `cash_out` terminates with $+3$ while `continue` earns $-1$ and reaches a state where `finish` terminates with $+10$. At $\gamma=0.9$, continuing returns $-1+0.9(10)=8$, exceeding $3$.

## Progressive theoretical exercises — Dynamic programming control

Work through these exercises **in order**, after reading the lesson. The earlier in-text exercises are warm-ups; this sequence gradually removes the scaffolding. No programming is needed. Write your reasoning as well as your final answer. Allow an additional 60–90 minutes, or split the sequence across study sessions. The final two exercises are stretch challenges: use their hints when needed. Try each question before reading its solution.

### Tools for the stretch exercises

“Componentwise” means at every state separately. “Monotone” means that increasing continuation values cannot decrease a probability-weighted Bellman backup. A “margin” is the winner's score minus a competitor's score.

To certify a ranking, allow for error on both sides. If each action estimate can be wrong by $0.1$, an observed gap of $0.3$ guarantees the winner remains ahead by at least $0.1$. A gap of $0.1$ does not settle the true ordering. Exercises 4T7–4T8 extend this idea to continuation error and whole-policy quality.

### Level 1 — Guided application

### 4T1 — Average outcomes before choosing an action

At $s$, A earns $0$ and reaches $x$ surely. B earns $1$ and reaches $x$ with probability $0.5$, otherwise earning $1$ and terminating. Let $V(x)=4$ and $\gamma=0.9$.

Compute $q_V(s,A)$ and $q_V(s,B)$, then select a greedy action. Explain why picking B's best random outcome instead would give an invalid comparison.

**Hint**

A terminal outcome has no discounted continuation.



**Solution**

$q_V(s,A)=0+0.9(4)=3.6$. $q_V(s,B)=0.5[1+0.9(4)]+0.5(1)=2.8$. A is greedy. Replacing B's expectation by its favorable outcome would assign it $4.6$ and choose B incorrectly. The agent controls the action, not the random outcome.



### 4T2 — Trace synchronous value iteration

Use default five-state LineWorld, $\gamma=0.9$, and $V_0=[0,0,0,0,0]$.

Compute the entire vectors $V_1,V_2,V_3,V_4$. In each sweep, read only the previous vector. When does the positive goal reward first influence state zero?

**Hint**

At state zero, left stays at zero and right moves to one. Keep terminal value zero.



**Solution**

$V_1=[-1,-1,-1,10,0]$. $V_2=[-1.9,-1.9,8,10,0]$. $V_3=[-2.71,6.2,8,10,0]$. $V_4=[4.58,6.2,8,10,0]$. The goal reward first influences state zero on sweep four, one transition backward per synchronous sweep. The fourth vector is already a fixed point here, although a stopping rule based on successive-vector differences may need another sweep to recognize that.



### Level 2 — Multi-step reasoning

### 4T3 — One complete policy-iteration improvement

A two-state MDP has nonterminal states $s,x$ and a terminal state. At $s$, `cash` terminates with $3$, while `continue` earns $-1$ and reaches $x$. At $x$, `quit` terminates with $0$ and `finish` terminates with $10$. Set $\gamma=0.9$.

Start with policy cash at $s$, quit at $x$.

1. Evaluate this policy at both states.
2. Improve both states simultaneously using its exact values.
3. Evaluate the improved policy and improve again.
4. Explain why the initially improved policy need not make every state strictly better.

**Hint**

On the first improvement, continuing at $s$ assumes the old quit action at $x$.



**Solution**

Initially $V^\pi(s)=3$, $V^\pi(x)=0$. At $s$, cash has lookahead $3$ and continue has $-1+0.9(0)=-1$, so cash stays. At $x$, finish beats quit ($10$ versus $0$). The first improved policy has values $(3,10)$. Its next improvement chooses continue at $s$ because $-1+0.9(10)=8>3$. The resulting policy has values $(8,10)$ and is stable. The first round improves $x$ strictly but leaves $s$ unchanged. Policy improvement guarantees nondecrease, not strict improvement at every state in every round.



### 4T4 — A stochastic backup near the goal

In five-state LineWorld, an intended direction executes oppositely with probability $p$. At state $3$, a right move reaches the goal with reward $10$, and a left move reaches $2$ with reward $-1$. Let $V(2)=6$, terminal continuation be zero, and $\gamma=0.9$.

1. Derive both intended-action lookahead values as functions of $p$.
2. Determine the greedy intended action for every $p\in[0,1]$.
3. Explain why a high slip probability can reverse the preferred intended direction.

**Hint**

The value of an actual left move is $-1+0.9(6)=4.4$.



**Solution**

Intended right gives $(1-p)10+p(4.4)=10-5.6p$. Intended left gives $(1-p)4.4+p(10)=4.4+5.6p$. Their difference is $5.6(1-2p)$, so intended right wins for $p<0.5$, they tie at $0.5$, and intended left wins for $p>0.5$. Above $0.5$, the opposite direction is more likely than the intended one. Intending left then makes actual right more likely. This comparison uses the supplied value vector; it is not a calculation of the entire slipped MDP's optimal values.



### Level 3 — Derivation and critique

### 4T5 — Prove exact policy improvement

Let $\pi'$ be greedy with respect to the exact $V^\pi$ in a finite discounted MDP.

1. Show $T_{\pi'}V^\pi\geq V^\pi$ componentwise.
2. Explain why a Bellman policy backup is monotone: if $U\geq W$, then $T_{\pi'}U\geq T_{\pi'}W$.
3. Combine monotonicity with convergence of repeated discounted backups to prove $V^{\pi'}\geq V^\pi$.
4. Identify the step that can fail for an approximate value vector.

**Hint**

Compare the maximum over old action values with their average under the old policy.



**Solution**

Greediness gives
$$(T_{\pi'}V^\pi)(s)=\max_aQ^\pi(s,a)\geq\sum_a\pi(a\mid s)Q^\pi(s,a)=V^\pi(s).$$
Backups are monotone because probabilities and $\gamma$ are nonnegative; replacing continuation values by larger ones cannot lower the weighted sum. Applying monotonicity repeatedly yields
$$V^\pi\leq T_{\pi'}V^\pi\leq (T_{\pi'})^2V^\pi\leq\cdots.$$
Discounted policy backups converge to $V^{\pi'}$, so taking limits proves the claim. With approximate values, greediness may order the true $Q^\pi$ values incorrectly. The first inequality relative to the exact $V^\pi$ is then not assured.



### 4T6 — Derive the optimality contraction

Prove $\|T_*V-T_*W\|_\infty\leq\gamma\|V-W\|_\infty$. Use the elementary inequality
$$|\max_a u_a-\max_a v_a|\leq\max_a|u_a-v_a|.$$
Then derive $\|V_k-V_*\|_\infty\leq\gamma^k\|V_0-V_*\|_\infty$ for synchronous value iteration.

**Hint**

First bound the difference between the two backups for each fixed action; reward terms cancel.



**Solution**

For a fixed action,
$$|q_V(s,a)-q_W(s,a)|\leq\gamma\sum_{s',r}p(s',r\mid s,a)|V(s')-W(s')|\leq\gamma\|V-W\|_\infty.$$
The supplied maximum inequality transfers this bound to $|(T_*V)(s)-(T_*W)(s)|$. Maximizing over states proves contraction. Since $V_*=T_*V_*$,
$$\|V_{k+1}-V_*\|_\infty\leq\gamma\|V_k-V_*\|_\infty.$$
Induction yields the requested bound. This controls value error; it does not say that every intermediate greedy policy is already optimal.



### Level 4 — Synthesis challenges

### 4T7 — When is an approximate greedy action reliable?

An approximate vector satisfies $\|\widehat V-V^\pi\|_\infty\leq\delta$. Compute lookahead values from the exact model, and suppose the estimated best action $a$ exceeds every other estimated action value by at least $m>0$.

1. Bound $|q_{\widehat V}(s,a)-Q^\pi(s,a)|$ for any action.
2. Give a sufficient inequality involving $m,\gamma,\delta$ for the chosen action to be truly best under $Q^\pi$.
3. With $\gamma=0.9$, $\delta=0.1$, check margins $0.3$ and $0.1$.
4. Does failure of this sufficient condition prove the action is wrong?

**Hint**

The estimated winner can be too high while a competitor is too low. Account for both errors.



**Solution**

Each action's continuation error is bounded by $\gamma\delta$ because outcome probabilities sum to one. For any competitor $b$,
$$Q^\pi(s,a)-Q^\pi(s,b)\geq q_{\widehat V}(s,a)-q_{\widehat V}(s,b)-2\gamma\delta\geq m-2\gamma\delta.$$
Thus $m>2\gamma\delta$ guarantees a strictly best true action; equality guarantees it is at least tied best. Here $2\gamma\delta=0.18$. Margin $0.3$ certifies strict correctness; $0.1$ does not. Failure of the condition means uncertainty, not proof of a wrong action. If every state's improvement is certified relative to the exact $V^\pi$, exact policy-improvement reasoning applies.



### 4T8 — Residuals, value accuracy, and policy quality

Let $\widehat V$ have optimality residual $b=\|\widehat V-T_*\widehat V\|_\infty$. Let $\widehat\pi$ be greedy with respect to $\widehat V$.

1. Derive $e=\|\widehat V-V_*\|_\infty\leq b/(1-\gamma)$.
2. Use $T_{\widehat\pi}\widehat V=T_*\widehat V$ and contraction of both operators to show
$$\|V_*-T_{\widehat\pi}V_*\|_\infty\leq2\gamma e.$$
3. Deduce $\|V_*-V^{\widehat\pi}\|_\infty\leq2\gamma e/(1-\gamma)$ and then a bound in terms of $b$.
4. Evaluate these guarantees at $\gamma=0.9$, $b=0.001$.

**Hint**

For part 1 insert $T_*\widehat V$ and use contraction. For part 3 insert $T_{\widehat\pi}V_*$ between $V_*$ and $V^{\widehat\pi}=T_{\widehat\pi}V^{\widehat\pi}$.



**Solution**

Part 1: $e\leq b+\gamma e$, so $e\leq b/(1-\gamma)$. For part 2, use $V_*=T_*V_*$ and the greedy identity:
$$\|T_*V_*-T_{\widehat\pi}V_*\|_\infty\leq\|T_*V_*-T_*\widehat V\|_\infty+\|T_{\widehat\pi}\widehat V-T_{\widehat\pi}V_*\|_\infty\leq2\gamma e.$$
For policy loss $L=\|V_*-V^{\widehat\pi}\|_\infty$, inserting the indicated term gives $L\leq2\gamma e+\gamma L$. Therefore
$$L\leq\frac{2\gamma e}{1-\gamma}\leq\frac{2\gamma b}{(1-\gamma)^2}.$$
Numerically $e\leq0.01$ and $L\leq0.18$. These are conservative upper bounds, not measured errors. A small residual can support value and policy-quality guarantees without identifying a unique action at every state, especially near ties.




## 10. Common mistakes and mastery checkpoint

Do not put the goal's entry reward into its state value as well: that double-counts it. Do not replace an expectation with a maximum over random outcomes: the agent chooses actions, not which random event occurs. Do not update a terminal state as though further decisions remained. If you mix in-place and synchronous updates, iteration counts change even when both methods converge to the same values.

Before proceeding, explain:

1. What distinguishes evaluation from control?
2. Why does exact greedy policy improvement work?
3. Why is $Q^\pi$ different from $Q_*$?
4. How does value iteration avoid full policy evaluation between improvements?
5. What must change when a known deterministic model becomes stochastic?
6. Why can a planning algorithm consider an action it has never physically tried?

**Next:** Monte Carlo control and temporal-difference learning will address the case where the model is unknown. Exploration returns to center stage: unlike this planner, an experience-based learner cannot back up an unseen action using a supplied model.
