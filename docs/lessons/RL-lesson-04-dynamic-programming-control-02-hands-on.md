# Lesson 4 — Improving Policies with Dynamic Programming — Version 2 — Hands-on (Python)

> **Concepts and theoretical exercises:** [RL-lesson-04-dynamic-programming-control](RL-lesson-04-dynamic-programming-control-01.md)  
> **Study pace:** Allow 45–75 minutes for the coding lab and practical exercises.  
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

Use Python 3.10+ and run every command in this tutorial from the **repository root**. Follow the [environment setup in the README](../../README.md), then install the course in editable mode:

```bash
python -m pip install -e .
```

For Lesson 1 plots, use `python -m pip install -e ".[plots]"`. Text runs require no runtime dependencies beyond the standard library. Editable installation lets commands use your source changes without reinstalling; no copying baseline programs out of Markdown is needed.

The walkthrough uses the supplied baseline program. Exercises and their solutions follow at the end. Solution code remains in this document: modification snippets apply to the linked lesson source; standalone solution programs can be saved under a local `scratch/` directory. Create that directory before saving a program there. It is excluded from version control.

## 1. The environment and assumptions

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

## 2. Guided tutorial — Run the lab

### Find the implementation

The [Lesson 4 source](../../src/learning_rl/lessons/lesson04.py) contains `action_value()`, `evaluate()`, `policy_iteration()`, and `value_iteration()`. It imports the shared [LineWorld model](../../src/learning_rl/environments/lineworld.py). Policy iteration starts with a uniform random policy; both algorithms choose left on exact action-value ties. Numerical evaluation stops at a tolerance.

The CLI compares algorithms and varies route length, discount and tolerance. It counts policy evaluation sweeps separately from policy improvement rounds: these are different units of work.

### Terminal commands

```bash
python -m learning_rl.lessons.lesson04 --help
python -m learning_rl.lessons.lesson04 --algorithm both
python -m learning_rl.lessons.lesson04 --algorithm policy --gamma 0.5
python -m learning_rl.lessons.lesson04 --algorithm value --size 20 --gamma 0.99
python -m learning_rl.lessons.lesson04 --algorithm both --theta 0.000001
```

For the default run, both algorithms should report approximately:

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

### Guided comparisons

Run `--algorithm both` and compare values and actions state by state. Read policy rounds and evaluation sweeps separately. Then increase `--size` to eight and calculate the expected values before running. Change `--gamma` while holding size and tolerance fixed, and inspect the final residual as well as computation counts.

| Option | Meaning |
| --- | --- |
| `--algorithm policy`, `value`, `both` | Which planning algorithm to run |
| `--size` | Number of LineWorld positions |
| `--gamma` | Discount in $[0,1)$ |
| `--theta` | Positive stopping tolerance |
| `--max-iterations` | Limit on evaluation or improvement iterations |

### Reading the distance formula as a delivery bill

If the courier is three moves from the goal, it pays two ordinary movement costs and then earns the goal reward. The formula above becomes

$$
V(s)=-(1+\gamma)+10\gamma^2.
$$

At $\gamma=0.9$, this is $-1-0.9+8.1=6.2$. There are $d-1$ ordinary-cost rewards and one delivery reward. The delivery reward is the $d$-th reward, so its discount exponent is $d-1$. For $d=1$, there is no ordinary movement cost; value is simply $10$.

### Numerical checking

The script prints the final **Bellman optimality residual**:

$$
b=\|V-T_*V\|_\infty.
$$

For the discounted setting,

$$
\|V-V_*\|_\infty\leq\frac{b}{1-\gamma}.
$$

A small residual supports accuracy of the values; it is not, by itself, a claim of exact floating-point equality. For example, $b=10^{-6}$ and $\gamma=0.99$ give a maximum value-error bound of $10^{-4}$. This bound concerns values, not an exact guarantee that every selected action matches a particular optimal policy when actions nearly tie.

## 3. Practical exercises

1. **Propagation:** add `--trace` to value iteration to print the value vector after each sweep. Predict when the goal reward first affects state zero, then verify by command line.
2. **Analytical check:** use `--size 8 --gamma 0.9`, calculate the always-right values with the formula above, and compare both algorithms.
3. **Policy evaluation cost:** compare discounts $0.5$, $0.9$ and $0.99$. Report evaluation sweeps, improvement rounds, value-iteration sweeps and residuals. Do not assume every individual problem requires more sweeps at a larger discount.
4. **Alternative start:** add `--initial-policy left` and `--initial-policy random`. Explain how policy iteration can improve a policy that never initially reaches the goal: the model still reveals what right would do.
5. **Stochastic model:** add `--slip-prob` in $[0,1]$, which executes the opposite direction with that probability. For each intended action, average the two possible outcomes and their rewards in the backup. Retain zero terminal continuation. Demonstrate `--slip-prob 0` and `0.2` with terminal commands.

For the stochastic exercise, test the expected one-step value at state $3$ with zero initial values: intending right gives $(1-p)10+p(-1)$. A transition into the goal earns the goal reward even if it occurs because of a slip.

### Additional challenge — Design a small control problem

Design a tiny MDP in which an immediate attractive reward leads to a worse long-term result. Keep at most four states and two actions per nonterminal state.

Specify all transition probabilities, rewards, terminal conditions and discount. Calculate $Q^\pi$ for an initial policy, improve it once by hand, and calculate the new value. Then adapt the CLI program to your model. Your submission should include a state/action table and a terminal command reproducing your result.

One possible starting point: from a start state, `cash_out` terminates with $+3$ while `continue` earns $-1$ and reaches a state where `finish` terminates with $+10$. At $\gamma=0.9$, continuing returns $-1+0.9(10)=8$, exceeding $3$.

### Solutions — Exercises 1–5

#### 1. Trace value propagation

Use the extended program under exercises 4–5 below; it also implements `--trace`. Save it as `scratch/lesson04_solutions.py`.

```bash
python scratch/lesson04_solutions.py --algorithm value --trace
```

Starting from zeros, the first four synchronous value vectors are `[-1, -1, -1, 10, 0]`, `[-1.9, -1.9, 8, 10, 0]`, `[-2.71, 6.2, 8, 10, 0]`, and `[4.58, 6.2, 8, 10, 0]`. The goal reward first influences state zero on sweep four. Sweep five repeats the fixed point and recognizes convergence with the change-based stopping rule. Updating in place would change this propagation pattern.

#### 2. Check the eight-state route

```bash
python -m learning_rl.lessons.lesson04 --size 8 --gamma 0.9 --algorithm both
```

Working backward from terminal value zero gives `[0.62882, 1.8098, 3.122, 4.58, 6.2, 8, 10, 0]`. Both algorithms should return these values within the tolerance and choose right at every nonterminal state. Each preceding value is $-1+0.9$ times the next value, except the goal-entry transition, which gives $10$ with zero continuation.

#### 3. Compare work and residuals

```bash
python -m learning_rl.lessons.lesson04 --algorithm both --gamma 0.5
python -m learning_rl.lessons.lesson04 --algorithm both --gamma 0.9
python -m learning_rl.lessons.lesson04 --algorithm both --gamma 0.99
```

With the original five-state model, random initial policy, zero initial evaluation values and tolerance $10^{-10}$, the reference results are:

| Discount | Policy evaluation sweeps, total | Policy rounds | Value-iteration sweeps | Final residual, both |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 36 | 2 | 5 | 0 |
| 0.9 | 139 | 2 | 5 | 0 |
| 0.99 | 313 | 2 | 5 | 0 |

Policy evaluation takes more sweeps here as the discount increases. Value iteration still needs five sweeps because this route’s optimal goal-reaching values propagate backward in four sweeps, with one extra sweep to detect stability. Policy rounds include the final round that confirms no change. Zero printed residual reflects this small deterministic example’s computed fixed point; it is not a guarantee for other models.

#### 4–5. Implement alternative starts and stochastic transitions

This complete extension combines `--trace`, `--initial-policy`, and `--slip-prob`. Save it as `scratch/lesson04_solutions.py`; the original lab program can remain as a baseline. `random` means the uniform stochastic policy with right probability $0.5$, matching the original initialization. Every backup, greedy decision and residual uses the same slipped model.

```python
import argparse


def action_value(state, action, values, size, gamma, slip):
    result = 0.0
    for actual, probability in [(action, 1 - slip), (1 - action, slip)]:
        nxt = max(0, state - 1) if actual == 0 else min(size - 1, state + 1)
        done = nxt == size - 1
        reward = 10.0 if done else -1.0
        result += probability * (reward + gamma * (0.0 if done else values[nxt]))
    return result


def greedy(state, values, size, gamma, slip):
    return int(action_value(state, 1, values, size, gamma, slip)
               > action_value(state, 0, values, size, gamma, slip))


def evaluate(policy, size, gamma, theta, limit, slip):
    values = [0.0] * size
    for sweep in range(1, limit + 1):
        new = values.copy()
        for s in range(size - 1):
            new[s] = sum(prob * action_value(s, a, values, size, gamma, slip)
                         for a, prob in [(0, 1 - policy[s]), (1, policy[s])])
        delta = max(abs(x - y) for x, y in zip(new, values))
        values = new
        if delta < theta:
            return values, sweep
    raise RuntimeError("Policy evaluation did not converge")


def policy_iteration(size, gamma, theta, limit, slip, initial):
    policy = [0.0 if initial == "left" else 0.5] * (size - 1)
    total = 0
    for round_number in range(1, limit + 1):
        values, sweeps = evaluate(policy, size, gamma, theta, limit, slip)
        total += sweeps
        new = [float(greedy(s, values, size, gamma, slip)) for s in range(size - 1)]
        if new == policy:
            return values, new, round_number, total
        policy = new
    raise RuntimeError("Policy iteration did not stabilize")


def value_iteration(size, gamma, theta, limit, slip, trace):
    values = [0.0] * size
    for sweep in range(1, limit + 1):
        new = values.copy()
        for s in range(size - 1):
            new[s] = max(action_value(s, a, values, size, gamma, slip) for a in (0, 1))
        delta = max(abs(x - y) for x, y in zip(new, values))
        values = new
        if trace:
            print(f"sweep={sweep} values={[round(v, 6) for v in values]}")
        if delta < theta:
            policy = [float(greedy(s, values, size, gamma, slip)) for s in range(size - 1)]
            return values, policy, sweep, sweep
    raise RuntimeError("Value iteration did not converge")


def main():
    parser = argparse.ArgumentParser(description="DP with tracing, initial policies and slips.")
    parser.add_argument("--algorithm", choices=["policy", "value", "both"], default="both")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--initial-policy", choices=["left", "random"], default="random")
    parser.add_argument("--slip-prob", type=float, default=0.0)
    args = parser.parse_args()
    if args.size < 2 or args.max_iterations < 1:
        parser.error("size >= 2 and max-iterations >= 1 required")
    if not 0 <= args.gamma < 1 or not args.theta > 0:
        parser.error("gamma in [0, 1) and theta > 0 required")
    if not 0 <= args.slip_prob <= 1:
        parser.error("slip-prob must be in [0, 1]")
    common = (args.size, args.gamma, args.theta, args.max_iterations, args.slip_prob)
    for name in ("policy", "value"):
        if args.algorithm not in (name, "both"):
            continue
        if name == "policy":
            values, policy, rounds, sweeps = policy_iteration(*common, args.initial_policy)
        else:
            values, policy, rounds, sweeps = value_iteration(*common, args.trace)
        residual = max(abs(values[s] - max(action_value(s, a, values, args.size,
                                                        args.gamma, args.slip_prob)
                                          for a in (0, 1)))
                       for s in range(args.size - 1))
        print(f"algorithm={name} rounds={rounds} sweeps={sweeps} residual={residual:.3g}")
        for s, value in enumerate(values):
            action = "terminal" if s == args.size - 1 else ("right" if policy[s] else "left")
            print(f"state={s} value={value:.6f} action={action}")


if __name__ == "__main__":
    main()
```

For exercise 4:

```bash
python scratch/lesson04_solutions.py --help
python scratch/lesson04_solutions.py --algorithm policy --initial-policy left
python scratch/lesson04_solutions.py --algorithm policy --initial-policy random
```

Both starts yield `[4.58, 6.2, 8, 10, 0]` and an all-right policy at the default discount. With no slip, the initial left policy has value $-10$ everywhere nonterminal. At state three, the known model reveals that right earns $10$ immediately, so improvement changes that decision even though the initial policy never visits the goal. Subsequent rounds propagate improvement to earlier states. Finite evaluation error can affect decisions whose action values are almost tied; compare final values and residuals rather than requiring identical round counts for the two starts.

For exercise 5:

```bash
python scratch/lesson04_solutions.py --algorithm both --slip-prob 0
python scratch/lesson04_solutions.py --algorithm both --slip-prob 0.2
```

With zero slip, the original results are recovered. With slip $0.2$, both algorithms should agree within numerical tolerance; the value vector is lower because unintended left moves delay delivery. From state three with zero continuation values, intended right has value $0.8(10)+0.2(-1)=7.8$, and intended left has value $0.8(-1)+0.2(10)=1.2$. At slip $0.5$, both intended directions have the same outcome distribution and the stable tie rule selects left. At slip one, intending left executes right with certainty. Reward depends on the actual transition, so a slipped move into the goal still earns $10$.

For slip $0.2$ and the default discount, the reference value vector is `[2.073066, 3.749880, 6.078790, 8.894182, 0]`, with intended right at every nonterminal state.

### Solution — A complete two-decision model

Use two nonterminal states, `start` and `work`, plus terminal state `done`. All listed transitions have probability one; `done` has no available actions and value zero. Set $\gamma=0.9$.

| State | Action | Next state | Reward |
| --- | --- | --- | ---: |
| start | cash_out | done | 3 |
| start | continue | work | -1 |
| work | quit | done | 0 |
| work | finish | done | 10 |

Initially choose `cash_out` at `start` and `quit` at `work`, giving values `(3, 0)`. Under that policy, $Q^\pi(\text{start},\text{cash_out})=3$, $Q^\pi(\text{start},\text{continue})=-1$, $Q^\pi(\text{work},\text{quit})=0$, and $Q^\pi(\text{work},\text{finish})=10$. One simultaneous improvement keeps `cash_out` and selects `finish`, producing values `(3, 10)`. A second improvement now selects `continue` because $-1+0.9(10)=8>3$, yielding the stable values `(8, 10)`.

Save this adapted model-based planner as `scratch/lesson04_challenge.py`. Its acyclic model permits exact evaluation in reverse dependency order instead of iterative evaluation sweeps.

```python
import argparse

MODEL = {
    "start": {"cash_out": ("done", 3.0), "continue": ("work", -1.0)},
    "work": {"quit": ("done", 0.0), "finish": ("done", 10.0)},
}


def q(state, action, values, gamma):
    nxt, reward = MODEL[state][action]
    return reward + gamma * values[nxt]


def main():
    parser = argparse.ArgumentParser(description="Policy iteration on a tiny delivery MDP.")
    parser.add_argument("--gamma", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 <= args.gamma < 1:
        parser.error("gamma must be in [0, 1)")
    policy = {"start": "cash_out", "work": "quit"}
    for round_number in range(1, 5):
        values = {"done": 0.0}
        for state in ("work", "start"):
            values[state] = q(state, policy[state], values, args.gamma)
        print(f"round={round_number} policy={policy} values={values}")
        improved = {s: max(MODEL[s], key=lambda a: q(s, a, values, args.gamma))
                    for s in MODEL}
        if improved == policy:
            break
        policy = improved
    else:
        raise RuntimeError("Policy did not stabilize")


if __name__ == "__main__":
    main()
```

```bash
python scratch/lesson04_challenge.py --help
python scratch/lesson04_challenge.py --gamma 0.9
python scratch/lesson04_challenge.py --gamma 0.2
```

At discount $0.9$, the three printed rounds show `(3, 0)`, `(3, 10)`, then `(8, 10)`. At discount $0.2$, continuing with `finish` gives only $-1+0.2(10)=1$, so `cash_out` remains optimal at `start`. The attraction of the delayed reward depends on the objective’s discount, not just its undiscounted amount.
