# Lesson 4 — Improving Policies with Dynamic Programming — Version 2 — Hands-on (Python)

> **Vocabulary:** Use the [project glossary](../docs/glossary.md) for the terms and notation shared by all lessons and code.

> **Concepts and theoretical exercises:** [RL-lesson-04-dynamic-programming-control](../lessons/RL-lesson-04-dynamic-programming-control-01.md)\
> **Study pace:** Allow 45–75 minutes for the experiment lab and practical exercises.
>
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

After installation, run lesson commands from any directory with the project virtual environment activated. Relative output paths are resolved from your current directory. Follow the [virtual environment setup](../README.md#create-a-local-virtual-environment) and [package installation instructions](../README.md#install-packages) in the README before starting. The standard installation includes Matplotlib for plotting in current and subsequent lessons.

The walkthrough and exercises use the supplied command-line programs. Run experiments with their existing arguments, then report the requested outputs and interpretations. No source edits or new programs are required. Exercises and solutions follow at the end.

```bash
RL lesson04 --help
```

All options are optional. An option that takes a value requires that value when supplied. Flags take no value.

| Option | Required | Values | Role |
| --- | --- | --- | --- |
| `-h`, `--help` | optional | No value; flag. | Display help and exit. |
| `--algorithm` | optional | Placeholder: `ALGORITHM`; one [registered algorithm](#registered-algorithm-values). Default: `both`. | Select the DP control algorithm to run. |
| `--size` | optional | Placeholder: `SIZE`; integer ≥ 2. Default: `5`. | Number of positions, including start and goal. |
| `--gamma` | optional | Placeholder: `GAMMA`; finite real number in [0, 1), dimensionless. Default: `0.9`. | Discount factor used to calculate values and improve policies. |
| `--theta` | optional | Placeholder: `THETA`; real number > 0. Default: `1e-10`. | Stop policy evaluation or value iteration when the largest value change in a sweep is below this tolerance. |
| `--max-iterations` | optional | Placeholder: `MAX_ITERATIONS`; integer ≥ 1. Default: `100000`. | Limit policy improvement rounds, sweeps per policy evaluation, and value iteration sweeps; report failure if convergence is not reached. |
| `--output-dir` | optional | Placeholder: `OUTPUT_DIRECTORY`; relative or absolute directory path. Default: `output/lesson04/TIMESTAMP/`. | Save `summary.csv` and `run.log`, creating the directory if needed. Reusing a directory overwrites these files. |

### Registered algorithm values

| Value of `ALGORITHM` | Role |
| --- | --- |
| `policy` | Alternate policy evaluation and greedy improvement using policy iteration. |
| `value` | Apply Bellman optimality backups using value iteration, then extract a greedy policy. |
| `both` | Run both policy iteration and value iteration. |

### Lesson output

Lesson commands automatically save summary tables to `output/lesson04/TIMESTAMP/summary.csv`
and other output to `run.log` in the same directory. Tables also appear in the terminal;
the log points to the CSV instead of duplicating the tables.
Each run gets a new directory. Use `--output-dir` to choose a directory; reusing it
overwrites its CSV and log. See [output conventions](../docs/command-outputs.md).

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

The [Lesson 4 source](../src/learning_rl/lessons/lesson04.py) contains `action_value()`, `evaluate()`, `policy_iteration()`, and `value_iteration()`. It imports the shared [LineWorld model](../src/learning_rl/environments/lineworld.py). Policy iteration starts with a uniform random policy; both algorithms choose left on exact action-value ties. Numerical evaluation stops at a tolerance.

The CLI compares algorithms and varies route length, discount and tolerance. It counts policy evaluation sweeps separately from policy improvement rounds: these are different units of work.

### Terminal commands

```bash
RL lesson04 --algorithm both
RL lesson04 --algorithm policy --gamma 0.5
RL lesson04 --algorithm value --size 20 --gamma 0.99
RL lesson04 --algorithm both --theta 0.000001
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

Include the commands used and the requested output tables with each answer.

1. **Propagation distance:** run value iteration for sizes 3, 5 and 8. Provide a table of size, value-iteration sweeps and state-zero value. Predict how many synchronous sweeps are needed for the goal reward to affect state zero, and explain the additional sweep used to detect convergence.
2. **Analytical check:** use `--size 8 --gamma 0.9`, calculate the always-right values with the formula above, and provide both algorithms' printed state values and actions alongside your calculation.
3. **Policy evaluation cost:** compare discounts $0.5$, $0.9$ and $0.99$. Report evaluation sweeps, improvement rounds, value-iteration sweeps and residuals. Do not assume every individual problem requires more sweeps at a larger discount.
4. **Tolerance:** compare `--theta 0.01`, `0.000001` and `0.0000000001` with `--algorithm both --size 8 --gamma 0.99`. Provide sweeps, rounds, residuals and state-zero values. Calculate each residual-based value-error bound $b/(1-\gamma)$; explain why a looser tolerance need not change the final output in this deterministic route.
5. **Immediate rewards and ties:** run `--algorithm both --gamma 0` and `--gamma 0.9`. Provide the state/action tables and identify where the selected actions differ. Explain why the zero-discount optimum can choose left away from the goal.

### Additional challenge - Evaluate a longer route

Use the supplied CLI with `--size 20 --algorithm both` at discounts `0.5` and `0.99`. Provide both state-zero values, action tables and residuals. Verify state-zero values using the distance formula, and explain how the discount changes the value of a delayed goal reward. Does a negative state-zero value imply that moving left is better?

### Solutions — Exercises 1–5

#### 1. Compare propagation distances

```bash
RL lesson04 --algorithm value --size 3
RL lesson04 --algorithm value --size 5
RL lesson04 --algorithm value --size 8
```

| Size | Goal distance from state zero | Reported sweeps | State-zero value |
| ---: | ---: | ---: | ---: |
| 3 | 2 | 3 | 8 |
| 5 | 4 | 5 | 4.58 |
| 8 | 7 | 8 | 0.62882 |

Synchronous backups propagate the goal reward one state per sweep. The final extra sweep confirms that values no longer change. The CLI reports final values and sweep counts; these establish the comparison without requiring intermediate traces.

#### 2. Check the eight-state route

```bash
RL lesson04 --size 8 --gamma 0.9 --algorithm both
```

Working backward from terminal value zero gives `[0.62882, 1.8098, 3.122, 4.58, 6.2, 8, 10, 0]`. Both algorithms should return these values within the tolerance and choose right at every nonterminal state. Each preceding value is $-1+0.9$ times the next value, except the goal-entry transition, which gives $10$ with zero continuation.

#### 3. Compare work and residuals

```bash
RL lesson04 --algorithm both --gamma 0.5
RL lesson04 --algorithm both --gamma 0.9
RL lesson04 --algorithm both --gamma 0.99
```

With the original five-state model, random initial policy, zero initial evaluation values and tolerance $10^{-10}$, the reference results are:

| Discount | Policy evaluation sweeps, total | Policy rounds | Value-iteration sweeps | Final residual, both |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 36 | 2 | 5 | 0 |
| 0.9 | 139 | 2 | 5 | 0 |
| 0.99 | 313 | 2 | 5 | 0 |

Policy evaluation takes more sweeps here as the discount increases. Value iteration still needs five sweeps because this route’s optimal goal-reaching values propagate backward in four sweeps, with one extra sweep to detect stability. Policy rounds include the final round that confirms no change. Zero printed residual reflects this small deterministic example’s computed fixed point; it is not a guarantee for other models.

#### 4. Compare tolerances

```bash
RL lesson04 --algorithm both --size 8 --gamma 0.99 --theta 0.01
RL lesson04 --algorithm both --size 8 --gamma 0.99 --theta 0.000001
RL lesson04 --algorithm both --size 8 --gamma 0.99 --theta 0.0000000001
```

Copy each algorithm's rounds, sweeps, residual and state-zero value into a table. Divide each printed residual by `0.01` for its value-error bound, allowing for output rounding. Evaluation work can change with tolerance. Once the all-right policy is selected, its acyclic route can propagate exact values in finitely many sweeps; all three tolerances can therefore produce the same final values and zero printed residual. Zero printed residual describes the computed result, not a general exact-arithmetic guarantee.

#### 5. Inspect immediate-reward ties

```bash
RL lesson04 --algorithm both --gamma 0
RL lesson04 --algorithm both --gamma 0.9
```

At zero discount both algorithms report `[-1, -1, -1, 10, 0]`, with left at states zero through two and right at state three. Away from the goal, both actions have immediate reward `-1`; the stable tie rule chooses left. At discount `0.9`, both report `[4.58, 6.2, 8, 10, 0]` with right at every nonterminal state. Zero discount ignores all delayed rewards, so an optimal policy for that objective need not reach the goal.

### Solution - Evaluate a longer route

```bash
RL lesson04 --size 20 --algorithm both --gamma 0.5
RL lesson04 --size 20 --algorithm both --gamma 0.99
```

State zero is 19 moves from the goal. Substitute $d=19$ into the distance formula and compare with each printed value. Both algorithms choose right at every nonterminal state and should agree within numerical tolerance. At discount `0.5`, state-zero value is approximately `-1.999954`; at `0.99`, it is approximately `-8.203486`. The latter objective counts more of the 18 movement costs as well as more of the delayed goal reward, so a larger discount does not necessarily raise return. Moving left delays completion and adds costs; a negative optimal value does not make left better. Report the measured residuals alongside the state tables.
