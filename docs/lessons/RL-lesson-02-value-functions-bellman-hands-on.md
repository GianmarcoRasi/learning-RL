# Lesson 2 — Value Functions and the Bellman Equation — Version 2 — Hands-on (Python)

> **Concepts and theoretical exercises:** [RL-lesson-02-value-functions-bellman](RL-lesson-02-value-functions-bellman.md)  
> **Study pace:** Allow 45–75 minutes for the coding lab and practical exercises.  
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

These are the original lesson’s command-line labs and practical exercises. The programs use Python 3.10+ and only the standard library. Run commands from the directory containing your saved script; on systems without `python`, use `python3`.

## 8. Practical lab --- Estimate values from experience

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

The lab uses **every-visit** MC: if an episode visits a state more than once, each visit provides a suffix return starting there. Those returns share parts of the same trip and are correlated. The advanced exercises explore both this dependence and truncation bias.

### Complete command-line lab: sampling and model-based evaluation

Save the following as `lesson02.py`. It is self-contained and uses only Python 3.10+ standard-library modules. Its LineWorld matches Lesson 1: every transition that does not reach the goal earns `-1`, and entering the goal earns `+10`. Terminal value is zero: the `+10` belongs to the transition into the terminal, not to the terminal state itself.

The `mc` mode averages observed suffix returns from **every visit**. Visits within one episode are correlated; do not treat them as independent samples when estimating uncertainty. The `dp` mode computes values from the known model. Both use the same policy, specified by `--right-prob`.

```python
import argparse
import random


def transition(state, action, size):
    next_state = max(0, state - 1) if action == 0 else min(size - 1, state + 1)
    done = next_state == size - 1
    return next_state, 10.0 if done else -1.0, done


def evaluate(size, gamma, right_prob, theta, max_iterations):
    values = [0.0] * size
    for iteration in range(1, max_iterations + 1):
        new = values.copy()
        for state in range(size - 1):
            new[state] = 0.0
            for action, prob in [(0, 1 - right_prob), (1, right_prob)]:
                nxt, reward, done = transition(state, action, size)
                new[state] += prob * (reward + gamma * (0.0 if done else values[nxt]))
        delta = max(abs(a - b) for a, b in zip(new, values))
        values = new
        if delta < theta:
            return values, iteration
    raise RuntimeError("Evaluation did not converge; increase --max-iterations")


def sample(size, gamma, right_prob, episodes, max_steps, rng):
    sums = [0.0] * size
    counts = [0] * size
    truncated = 0
    for _ in range(episodes):
        state = 0
        trajectory = []
        done = False
        for _ in range(max_steps):
            action = int(rng.random() < right_prob)
            nxt, reward, done = transition(state, action, size)
            trajectory.append((state, reward))
            state = nxt
            if done:
                break
        truncated += int(not done)
        # Every-visit MC: each visit contributes its observed suffix return.
        # A truncated suffix is deliberately assigned zero continuation here.
        G = 0.0
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            sums[state] += G
            counts[state] += 1
    values = [sums[s] / counts[s] if counts[s] else None for s in range(size)]
    values[-1] = 0.0
    return values, counts, truncated


def main():
    parser = argparse.ArgumentParser(description="Estimate or compute LineWorld policy values.")
    parser.add_argument("--method", choices=["mc", "dp", "compare"], default="compare")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--right-prob", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.size < 2 or min(args.episodes, args.max_steps, args.max_iterations) < 1:
        parser.error("size >= 2 and positive count arguments required")
    if not 0 <= args.gamma < 1 or not 0 <= args.right_prob <= 1 or not args.theta > 0:
        parser.error("gamma in [0, 1), right-prob in [0, 1], theta > 0 required")
    if args.method in ("dp", "compare"):
        values, iterations = evaluate(args.size, args.gamma, args.right_prob,
                                      args.theta, args.max_iterations)
        print(f"DP iterations={iterations}")
        for state, value in enumerate(values):
            print(f"state={state} dp_value={value:.6f}")
    if args.method in ("mc", "compare"):
        estimates, counts, truncated = sample(args.size, args.gamma, args.right_prob,
                                              args.episodes, args.max_steps,
                                              random.Random(args.seed))
        print(f"MC truncated_episodes={truncated}/{args.episodes}")
        for state, estimate in enumerate(estimates):
            label = "unvisited" if estimate is None else f"{estimate:.6f}"
            print(f"state={state} mc_value={label} visits={counts[state]}")
        if truncated:
            print("WARNING: zero continuation at time limits biases full-episode MC estimates.")


if __name__ == "__main__":
    main()
```

### Run and compare

```bash
python lesson02.py --help
python lesson02.py --method compare --episodes 10000 --seed 7
python lesson02.py --method dp --right-prob 1
python lesson02.py --method compare --gamma 0.5 --episodes 10000
python lesson02.py --method dp --gamma 0.99
python lesson02.py --method mc --max-steps 2 --episodes 100
```

With `--method dp --right-prob 1`, expect the values `[4.58, 6.2, 8.0, 10.0, 0.0]` at the default discount. With a random policy, MC estimates should approach the DP values as sufficient complete experience is collected; individual finite runs need not improve monotonically when you add episodes.

**Time-limit caveat:** the MC code uses zero continuation at a truncated trajectory's end. This omits later rewards and can bias estimates of the full-episode value. It reports how often that happens. Simply discarding all truncated episodes can also bias estimates by selecting shorter trajectories. For this lab, use a generous limit and inspect the truncation count. Later, TD learning will give us a principled way to bootstrap across external limits. If time remaining is part of the task itself, it belongs in the state and you must evaluate that finite-horizon task instead.

## 9. Practical lab — Iterative policy evaluation

Rather than wait for trips, a known model lets us repeatedly revise a table of values. A **Bellman backup** calculates reward plus discounted next-state estimates and averages according to the fixed policy. A **sweep** performs one such update for every state.

For example, a delivery route has A to B with reward $1$, and B to terminal with reward $4$. Set $\gamma=0.5$ and start with estimates $V_0(A)=V_0(B)=0$.

| Sweep | Estimate for A | Estimate for B |
| --- | --- | --- |
| Initial | $0$ | $0$ |
| First | $1+0.5(0)=1$ | $4+0.5(0)=4$ |
| Second | $1+0.5(4)=3$ | $4$ |
| Third | $3$ | $4$ |

The delivery benefit reaches A on the second sweep. These are revisions to predictions, not trips through the environment. The policy stays fixed throughout evaluation.

The `dp` lab uses **synchronous** sweeps: every new value reads from the previous table. Using B's newly updated value immediately when updating A would be an in-place method with different intermediate tables.

The script stops when the largest change in a sweep is below $\theta$:

$$
\max_s|V_{k+1}(s)-V_k(s)|<\theta.
$$

Here $k$ counts sweeps, $|\cdot|$ means absolute difference, and $\max_s$ finds the largest difference over states. For changes $0.02,0.005,0$, the largest is $0.02$; a tolerance of $0.01$ would not stop yet.

In a finite MDP with bounded rewards and $\gamma<1$, discounted policy backups converge, including for policies that loop forever. A policy earning $-1$ forever has value $-1/(1-\gamma)$, even though it never reaches a goal. The lab restricts discounted DP to $\gamma<1$; allowing $\gamma=1$ needs further assumptions.

A small sweep change is a stopping signal rather than an exact value-error measurement. If a value table's **Bellman residual**, its largest disagreement with its own backup, is $b$, then

$$
\|V-V^\pi\|_\infty\leq\frac{b}{1-\gamma}.
$$

The maximum norm $\|\cdot\|_\infty$ means the largest absolute entry. A residual $0.01$ gives an error bound $0.1$ at $\gamma=0.9$, but $1$ at $\gamma=0.99$. These are worst-case bounds, not exact errors. The stretch exercises derive them.

### Theory and coding exercises

1. Derive $V(3)$ under the random policy. Check that $V(3)=0.5[-1+\gamma V(2)]+0.5(10)$.
2. Calculate all values for the always-right policy by hand; verify with the CLI.
3. Compare 100, 1,000 and 10,000 MC episodes at the same seed. Which estimates vary most?
4. Compare DP sweep counts for $\gamma=0.5$, $0.9$ and $0.99$.
5. Run `--right-prob 0`. Why can discounted DP compute values although the goal is never reached? Why does finite truncated MC estimate a different quantity?
6. Add a `--visit-mode first` option to MC, retaining the every-visit default. A first-visit update uses the return following the earliest visit to a state in each episode. Compute returns backward, then choose earliest visits in forward order; selecting the first state encountered backward selects the *last* visit instead. Include a terminal command demonstrating both modes.

------------------------------------------------------------------------

### Solutions — Exercises 1–6

#### 1. Back up the random policy at state three

Left earns $-1$ and reaches state two; right earns $10$ and terminates. Each has probability $0.5$, giving $V(3)=0.5[-1+\gamma V(2)]+0.5(10)$. With the default discount and the converged random-policy value $V(2)\approx-0.565967$, this is $V(3)\approx4.245315$.

#### 2. Verify the always-right values

Starting from terminal value zero, work backward: $V(3)=10$, $V(2)=-1+0.9(10)=8$, $V(1)=-1+0.9(8)=6.2$, and $V(0)=-1+0.9(6.2)=4.58$.

```bash
python lesson02.py --method dp --right-prob 1
```

The printed vector is `[4.58, 6.2, 8.0, 10.0, 0.0]` up to formatting.

#### 3. Compare Monte Carlo estimates

```bash
python lesson02.py --method compare --episodes 100 --seed 7
python lesson02.py --method compare --episodes 1000 --seed 7
python lesson02.py --method compare --episodes 10000 --seed 7
```

One reference run of the original program gives these values, rounded to four decimals; terminal value is zero throughout.

| Episodes | State 0 | State 1 | State 2 | State 3 | Truncated episodes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | -4.0612 | -2.7032 | 0.5190 | 5.4499 | 0 |
| 1,000 | -4.3887 | -3.2251 | -0.5576 | 4.2154 | 0 |
| 10,000 | -4.4840 | -3.2769 | -0.5924 | 4.1731 | 0 |
| DP reference | -4.5025 | -3.2808 | -0.5660 | 4.2453 | — |

State three varies most across these three runs. It has fewer visits than earlier states, but variability also depends on the return distribution and within-episode dependence. These are observations for this seed, not a universal ranking. State two is closer to DP at 1,000 episodes than at 10,000: convergence is not monotonic in individual finite runs.

#### 4. Compare evaluation sweeps

```bash
python lesson02.py --method dp --gamma 0.5
python lesson02.py --method dp --gamma 0.9
python lesson02.py --method dp --gamma 0.99
```

With the original defaults, the respective sweep counts are 31, 134, and 308. A discount closer to one weakens the contraction guarantee; here the evaluation takes more sweeps. Keep the policy, initialization and tolerance fixed when comparing.

#### 5. Evaluate a policy that never finishes

```bash
python lesson02.py --method compare --right-prob 0 --max-steps 3 --episodes 100
```

Every sampled run stays at state zero and truncates. DP computes $V(s)=-1/(1-0.9)=-10$ for every nonterminal state. The three-reward first suffix has return $-2.71$. Every-visit MC averages all three suffixes, $[-2.71,-1.9,-1]$, so the reported state-zero estimate is $-1.87$. Other nonterminal states are `unvisited`. More repetitions of this same cutoff do not recover the infinite-horizon value.

#### 6. Implement first-visit Monte Carlo

Replace the original `sample()` function with the following. It keeps every-visit as the default, computes every suffix backward, and selects the earliest visit by scanning forward.

```python
def sample(size, gamma, right_prob, episodes, max_steps, rng, visit_mode="every"):
    sums = [0.0] * size
    counts = [0] * size
    truncated = 0
    for _ in range(episodes):
        state, trajectory, done = 0, [], False
        for _ in range(max_steps):
            action = int(rng.random() < right_prob)
            nxt, reward, done = transition(state, action, size)
            trajectory.append((state, reward))
            state = nxt
            if done:
                break
        truncated += int(not done)
        returns = [0.0] * len(trajectory)
        G = 0.0
        for t in range(len(trajectory) - 1, -1, -1):
            G = trajectory[t][1] + gamma * G
            returns[t] = G
        seen = set()
        for (state, _), G in zip(trajectory, returns):
            if visit_mode == "first" and state in seen:
                continue
            seen.add(state)
            sums[state] += G
            counts[state] += 1
    values = [sums[s] / counts[s] if counts[s] else None for s in range(size)]
    values[-1] = 0.0
    return values, counts, truncated
```

Add this argument before parsing:

```python
    parser.add_argument("--visit-mode", choices=["every", "first"], default="every")
```

Replace the `sample()` call in `main()` with:

```python
        estimates, counts, truncated = sample(
            args.size, args.gamma, args.right_prob, args.episodes,
            args.max_steps, random.Random(args.seed), args.visit_mode)
```

```bash
python lesson02.py --method mc --visit-mode every --episodes 10000 --seed 7
python lesson02.py --method mc --visit-mode first --episodes 10000 --seed 7
python lesson02.py --method mc --visit-mode first --right-prob 0 --max-steps 3 --episodes 1
```

In the final command, state zero has one counted visit and value $-2.71$; the every-visit counterpart has three counted visits and value $-1.87$. This check distinguishes the earliest visit from the last visit, whose return would be $-1$. Both modes retain zero continuation at truncation.

## 11. Challenge --- Improve the policy

Return to the five-state LineWorld.

You now have estimated $Q(s,a)$ values---or can reason about them.

Construct a new policy:

$$
\pi'(s)\in\arg\max_a Q(s,a).
$$

Then compare the new policy with the random policy.

Measure:

-   average episode return;
-   average episode length;
-   success rate.

### Think carefully

If the estimates of $Q$ are imperfect, is it always safe to stop exploring and choose the largest estimate forever?

This question leads directly toward bandits, exploration strategies, Monte Carlo control, temporal-difference learning, and eventually Q-learning.

### Solution — Improve and measure the policy

Use the known model to compute $Q^\pi(s,a)=r+\gamma V^\pi(s')$ from the random policy’s DP values. At states zero through three, the respective `(left, right)` values are approximately `(-5.0522, -3.9527)`, `(-5.0522, -1.5094)`, `(-3.9527, 2.8208)`, and `(-1.5094, 10)`. Greediness therefore selects right everywhere.

Save the following as `lesson02_compare.py` beside `lesson02.py`. It imports the original lesson’s functions, computes the improved policy, and measures both policies with the same episode and cutoff settings.

```python
import argparse
import random
from lesson02 import evaluate, transition


def measure(policy, episodes, max_steps, gamma, seed):
    rng = random.Random(seed)
    total_return = total_steps = successes = 0
    for _ in range(episodes):
        state, G, done = 0, 0.0, False
        for t in range(max_steps):
            action = int(rng.random() < policy[state])
            state, reward, done = transition(state, action, 5)
            G += gamma ** t * reward
            if done:
                break
        total_return += G
        total_steps += t + 1
        successes += int(done)
    return total_return / episodes, total_steps / episodes, successes / episodes


def main():
    parser = argparse.ArgumentParser(description="Compare random and improved policies.")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.episodes, args.max_steps) < 1 or not 0 <= args.gamma < 1:
        parser.error("positive counts and gamma in [0, 1) required")
    values, _ = evaluate(5, args.gamma, 0.5, 1e-10, 100000)
    improved = []
    for state in range(4):
        q = []
        for action in (0, 1):
            nxt, reward, done = transition(state, action, 5)
            q.append(reward + args.gamma * (0.0 if done else values[nxt]))
        improved.append(float(q[1] > q[0]))
    for name, policy in [("random", [0.5] * 4), ("improved", improved)]:
        G, length, success = measure(policy, args.episodes, args.max_steps,
                                     args.gamma, args.seed)
        print(f"policy={name} mean_return={G:.6f} mean_steps={length:.3f} "
              f"success_rate={success:.3f}")


if __name__ == "__main__":
    main()
```

```bash
python lesson02_compare.py --help
python lesson02_compare.py --episodes 10000 --seed 7
```

At the default discount, the improved policy has return $4.58$, length four and success rate one on every run. The random policy’s mean return approaches $-4.50247$ and its unlimited-run mean length is 20. Imperfect estimates can reverse action rankings; choosing their maximum forever can prevent collecting the evidence needed to correct that mistake. This solution uses a known model rather than claiming that noisy estimates always identify the correct improvement.

For the command above, one reference run reports random-policy mean return `-4.525890`, mean length `20.026`, and success rate `1.000`; the improved policy reports `4.580000`, `4.000`, and `1.000`. Both policies can reliably finish while earning very different returns and using very different numbers of moves.

------------------------------------------------------------------------
