# Lesson 2 — Value Functions and the Bellman Equation — Version 2 — Hands-on (Python)

> **Vocabulary:** Use the [project glossary](../docs/glossary.md) for the terms and notation shared by all lessons and code.

> **Concepts and theoretical exercises:** [RL-lesson-02-value-functions-bellman](RL-lesson-02-value-functions-bellman-01.md)  
> **Study pace:** Allow 45–75 minutes for the experiment lab and practical exercises.
>
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

After installation, run lesson commands from any directory with the project virtual environment activated. Relative output paths are resolved from your current directory. Follow the [virtual environment setup](../README.md#create-a-local-virtual-environment) and [package installation instructions](../README.md#install-packages) in the README before starting. The standard installation includes Matplotlib for plotting in current and subsequent lessons.

The walkthrough and exercises use the supplied command-line programs. Run experiments with their existing arguments, then report the requested outputs and interpretations. No source edits or new programs are required. Exercises and solutions follow at the end.

### Lesson output

Lesson commands automatically save printed results to `output/lesson02/TIMESTAMP/run.log`
and display them in the terminal. Each run gets a new directory. Use `--output-dir`
to choose a directory; reusing it overwrites its log. See [output conventions](../docs/command-outputs.md).

## 1. Practical lab --- Estimate values from experience

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

### Find the implementation

The [Lesson 2 source](../src/learning_rl/lessons/lesson02.py) contains `evaluate()` for synchronous Bellman sweeps, `sample()` for every-visit Monte Carlo estimates, and `main()` for the command-line interface. The shared [LineWorld model](../src/learning_rl/environments/lineworld.py) supplies transition outcomes. Ordinary transitions earn `-1`; entry to the goal earns `+10`. Terminal value is zero.

The `mc` mode averages observed suffix returns from every visit. Within-episode returns are correlated. The `dp` mode evaluates the known model. Both use the policy specified by `--right-prob`.

### Run and compare

```bash
RL lesson02 --help
RL lesson02 --method compare --episodes 10000 --seed 7
RL lesson02 --method dp --right-prob 1
RL lesson02 --method compare --gamma 0.5 --episodes 10000
RL lesson02 --method dp --gamma 0.99
RL lesson02 --method mc --max-steps 2 --episodes 100
```

With `--method dp --right-prob 1`, expect the values `[4.58, 6.2, 8.0, 10.0, 0.0]` at the default discount. With a random policy, MC estimates should approach the DP values as sufficient complete experience is collected; individual finite runs need not improve monotonically when you add episodes.

### Guided comparisons

Start with `--method dp --right-prob 1` and check the known value vector. Next run `--method compare`: match state numbers across the DP and MC rows and inspect visit counts and truncation. Finally lower `--max-steps` to two. Predict which states will be unvisited and why collecting more truncated episodes cannot restore omitted rewards.

| Option | What to vary or inspect |
| --- | --- |
| `--method mc`, `dp`, `compare` | Sampled estimates, model values, or both |
| `--right-prob` | The fixed policy being evaluated |
| `--episodes`, `--seed` | Sample amount and reproducibility |
| `--max-steps` | External episode cutoff |
| `--gamma`, `--theta`, `--max-iterations` | Discount, convergence tolerance and sweep limit |

**Time-limit caveat:** the MC code uses zero continuation at a truncated trajectory's end. This omits later rewards and can bias estimates of the full-episode value. It reports how often that happens. Simply discarding all truncated episodes can also bias estimates by selecting shorter trajectories. For this lab, use a generous limit and inspect the truncation count. Later, TD learning will give us a principled way to bootstrap across external limits. If time remaining is part of the task itself, it belongs in the state and you must evaluate that finite-horizon task instead.

## 2. Practical lab — Iterative policy evaluation

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

## Exercises - Run, report and interpret

Include your commands and the requested output rows with every answer.

1. Run default DP, report its state-three value, and verify that $V(3)=0.5[-1+\gamma V(2)]+0.5(10)$.
2. Calculate all values for the always-right policy by hand; provide the CLI value vector and compare it with your calculation.
3. Compare 100, 1,000 and 10,000 MC episodes at the same seed. Provide the MC value vectors, visit counts and truncation counts. Which estimates vary most?
4. Provide a table of DP sweep counts for $\gamma=0.5$, $0.9$ and $0.99$.
5. Run `--method compare --right-prob 0 --max-steps 3 --episodes 100`. Report the DP and MC vectors, visit counts and truncation count. Why can discounted DP compute values although the goal is never reached? Why does finite truncated MC estimate a different quantity?
6. **Measure cutoff bias.** Run `--method compare --right-prob 0 --episodes 100` with `--max-steps 1`, `3` and `10`. Provide a table of state-zero DP value, MC estimate, visit count and truncated episodes. Explain why increasing episode count alone cannot remove a fixed cutoff's bias.

------------------------------------------------------------------------

### Additional challenge — Compare fixed policies

Use the Lesson 2 CLI to report DP values for right probabilities `0.5` and `1`. Use the Lesson 1 CLI to compare their returns, mean episode lengths and success rates with `--compare --episodes 10000 --max-steps 1000 --seed 7 --quiet`. Provide both summary rows and explain why moving right always improves this route. If value estimates are noisy, could a greedy choice still select the wrong action?

### Solutions — Exercises 1–6

#### 1. Back up the random policy at state three

Left earns $-1$ and reaches state two; right earns $10$ and terminates. Each has probability $0.5$, giving $V(3)=0.5[-1+\gamma V(2)]+0.5(10)$. With the default discount and the converged random-policy value $V(2)\approx-0.565967$, this is $V(3)\approx4.245315$.

#### 2. Verify the always-right values

Starting from terminal value zero, work backward: $V(3)=10$, $V(2)=-1+0.9(10)=8$, $V(1)=-1+0.9(8)=6.2$, and $V(0)=-1+0.9(6.2)=4.58$.

```bash
RL lesson02 --method dp --right-prob 1
```

The printed vector is `[4.58, 6.2, 8.0, 10.0, 0.0]` up to formatting.

#### 3. Compare Monte Carlo estimates

```bash
RL lesson02 --method compare --episodes 100 --seed 7
RL lesson02 --method compare --episodes 1000 --seed 7
RL lesson02 --method compare --episodes 10000 --seed 7
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
RL lesson02 --method dp --gamma 0.5
RL lesson02 --method dp --gamma 0.9
RL lesson02 --method dp --gamma 0.99
```

With the original defaults, the respective sweep counts are 31, 134, and 308. A discount closer to one weakens the contraction guarantee; here the evaluation takes more sweeps. Keep the policy, initialization and tolerance fixed when comparing.

#### 5. Evaluate a policy that never finishes

```bash
RL lesson02 --method compare --right-prob 0 --max-steps 3 --episodes 100
```

Every sampled run stays at state zero and truncates. DP computes $V(s)=-1/(1-0.9)=-10$ for every nonterminal state. The three-reward first suffix has return $-2.71$. Every-visit MC averages all three suffixes, $[-2.71,-1.9,-1]$, so the reported state-zero estimate is $-1.87$. Other nonterminal states are `unvisited`. More repetitions of this same cutoff do not recover the infinite-horizon value.

#### 6. Measure cutoff bias

```bash
RL lesson02 --method compare --right-prob 0 --max-steps 1 --episodes 100
RL lesson02 --method compare --right-prob 0 --max-steps 3 --episodes 100
RL lesson02 --method compare --right-prob 0 --max-steps 10 --episodes 100
```

At the default discount, DP gives state-zero value approximately `-10` in each run. Every episode stays at zero and truncates; other nonterminal states are unvisited by MC.

| Cutoff | State-zero MC estimate | State-zero visits | Truncated episodes |
| ---: | ---: | ---: | ---: |
| 1 | -1 | 100 | 100 |
| 3 | -1.87 | 300 | 100 |
| 10 | -4.138106 | 1000 | 100 |

Every-visit MC averages suffixes of lengths 1 through the cutoff. Each suffix omits the infinite continuation. Repeating identical episodes increases counts without changing that average. A larger cutoff includes more rewards, bringing the estimate closer to `-10`.

### Solution — Compare fixed policies

```bash
RL lesson02 --method dp --right-prob 0.5
RL lesson02 --method dp --right-prob 1
RL lesson01 --compare --episodes 10000 --max-steps 1000 --seed 7 --quiet
```

The random-policy DP vector is approximately `[-4.50247, -3.28080, -0.56597, 4.24531, 0]`; always-right gives `[4.58, 6.2, 8, 10, 0]`. The Lesson 1 summary supplies measured returns, lengths and success rates. Always-right has return `4.58`, length four and success rate one. The random policy's mean return approaches `-4.50247` and its unlimited-run mean length is 20. Report your measured random-policy row, which can vary with the Python environment. Both policies can finish reliably while earning different returns. Noisy estimates can reverse action rankings; choosing their maximum forever can prevent collecting evidence needed to correct that mistake.
