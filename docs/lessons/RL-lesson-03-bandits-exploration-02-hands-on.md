# Lesson 3 — Multi-Armed Bandits: Exploration and Exploitation — Version 2 — Hands-on (Python)

> **Concepts and theoretical exercises:** [RL-lesson-03-bandits-exploration](RL-lesson-03-bandits-exploration-01.md)  
> **Study pace:** Allow 45–75 minutes for the coding lab and practical exercises.  
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

Run every command in this tutorial from the **repository root**. Follow the [virtual environment setup](../../README.md#create-a-local-virtual-environment) and [package installation instructions](../../README.md#install-packages) in the README before starting. The README also covers optional plotting dependencies for Lesson 1.

The walkthrough uses the supplied baseline program. Exercises and their solutions follow at the end. Solution code remains in this document: modification snippets apply to the linked lesson source; standalone solution programs can be saved under a local `scratch/` directory. Create that directory before saving a program there. It is excluded from version control.

### Lesson output

Lesson commands automatically save printed results to `output/lesson03/TIMESTAMP/run.log`
and display them in the terminal. Each run gets a new directory. Use `--output-dir`
to choose a directory; reusing it overwrites its log. See [output conventions](../../output/README.md).

## 1. Guided tutorial — Run the lab

### Find the implementation

The [Lesson 3 source](../../src/learning_rl/lessons/lesson03.py) contains `experiment()` for one bandit run and `main()` for comparisons across epsilon settings and repeated runs. Each arm returns a Gaussian reward with its supplied mean and standard deviation one. The agent sees rewards; simulator metrics use hidden means. Random tie breaking avoids favoring arm zero.

The `--initial` estimate affects early action selection. With sample averages, an arm's first reward replaces its initial estimate. Optimism encourages early exploration but cannot guarantee finding the best arm.

The lab reuses seed schedules across strategies. This can correlate their results, so separate standard errors alone are not a test of the difference between strategies. Assess strategy differences with that paired structure in mind.

### Terminal commands

```bash
python -m learning_rl.lessons.lesson03 --help
python -m learning_rl.lessons.lesson03 --epsilons 0 0.1 1 --steps 2000 --runs 200 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 1 --steps 100 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0 --initial 5 --runs 200
python -m learning_rl.lessons.lesson03 --means 0 0.9 1 --epsilons 0 0.01 0.1 --steps 5000
```

Interpret each summary row using all three main metrics. For uniform exploration, default means average to $0.4$; its expected per-pull pseudo-regret is $1-0.4=0.6$. Over 2,000 pulls, cumulative pseudo-regret should average near $1,200$, and optimal fraction near $1/3$ across enough runs. These are statistical checks, not exact outputs.

Pure greedy runs can settle on the wrong arm. Epsilon-greedy often improves average performance over this horizon, but no specific finite noisy experiment guarantees that ranking. Repeating the same command and seed reproduces results within the same Python environment.

### Guided comparisons

First use `--runs 1 --steps 100` to inspect final arm estimates and selection counts. Then compare epsilon settings over many runs. Read mean reward, standard error, pseudo-regret and optimal fraction together. Increase the horizon while keeping means and seed fixed; finally narrow the gap between the best arms. Predict each effect before running it.

| Option | Meaning |
| --- | --- |
| `--means` | Hidden mean reward of each arm |
| `--epsilons` | Exploration probabilities to compare |
| `--steps`, `--runs` | Pulls per run and repeated experiments |
| `--initial` | Initial estimate for every arm |
| `--seed` | Reproducible random stream schedule |

## 2. Practical exercises

1. **Horizon:** compare `--steps 20`, `200` and `2000`, holding other settings fixed. When does exploration start paying for itself? Report means over at least 100 runs.
2. **Difficulty:** compare means `[0,0.2,1]` and `[0,0.9,1]`. Explain why identifying the best arm is harder when the gap is small, even if mistakes then cost less.
3. **Optimism:** compare greedy runs with `--initial 0` and `--initial 5`. Explain why initial values disappear from an arm after its first sample-average update.
4. **Extend the CLI:** add `--alpha`. When omitted, use sample averages; when supplied in $(0,1]$, use a constant step size. Show two terminal commands and compare estimates on the stationary task.
5. **Nonstationarity:** add `--switch-step` to swap the best and worst arm means halfway through a run. Recompute the optimal mean for each time step when measuring regret. Compare sample averages with constant $\alpha=0.1$ at positive epsilon.

For the nonstationary exercise, explicitly document that regret now compares against the best available arm at each time step, a stronger changing comparator.

### Solutions — Exercises 1–5

#### 1. Compare horizons

```bash
python -m learning_rl.lessons.lesson03 --steps 20 --runs 200 --epsilons 0 0.1 1 --seed 7
python -m learning_rl.lessons.lesson03 --steps 200 --runs 200 --epsilons 0 0.1 1 --seed 7
python -m learning_rl.lessons.lesson03 --steps 2000 --runs 200 --epsilons 0 0.1 1 --seed 7
```

The original program gives the following reference results. Seeded random outputs can depend on the Python environment.

| Steps | Epsilon | Mean reward | Mean pseudo-regret | Optimal fraction |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0 | 0.6987 | 6.65 | 0.618 |
| 20 | 0.1 | 0.6949 | 6.72 | 0.617 |
| 20 | 1 | 0.4373 | 11.87 | 0.344 |
| 200 | 0 | 0.7531 | 49.97 | 0.706 |
| 200 | 0.1 | 0.8767 | 25.25 | 0.856 |
| 200 | 1 | 0.4043 | 119.74 | 0.335 |
| 2,000 | 0 | 0.7863 | 427.97 | 0.743 |
| 2,000 | 0.1 | 0.9339 | 132.80 | 0.926 |
| 2,000 | 1 | 0.4000 | 1200.59 | 0.333 |

At 20 steps, epsilon $0.1$ has slightly lower measured mean reward than greedy; by 200 steps it has higher reward and lower pseudo-regret in this experiment. This locates a change between the sampled horizons, not a universal crossover time. Compare cumulative regret across strategies at the same horizon; a longer experiment can earn more reward per pull while accumulating more total regret.

#### 2. Reduce the gap

```bash
python -m learning_rl.lessons.lesson03 --means 0 0.2 1 --steps 2000 --runs 200 --epsilons 0 0.1 --seed 7
python -m learning_rl.lessons.lesson03 --means 0 0.9 1 --steps 2000 --runs 200 --epsilons 0 0.1 --seed 7
```

The best-versus-second-best mean gap falls from $0.8$ to $0.1$, while reward noise still has standard deviation one. More observations are needed to distinguish nearly equal means. However, choosing arm one instead of arm two then costs only $0.1$ per pull instead of $0.8$. A lower optimal fraction can therefore coexist with a high mean reward and low regret. Report all three printed metrics; optimal fraction alone does not describe the cost of mistakes.

#### 3. Use optimistic initial values

```bash
python -m learning_rl.lessons.lesson03 --epsilons 0 --initial 0 --steps 2000 --runs 200 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0 --initial 5 --steps 2000 --runs 200 --seed 7
```

In a reference run, mean reward increases from approximately $0.7863$ to $0.8469$. Untried arms retain their optimistic value and can attract greedy choices after a tried arm receives a lower reward. On an arm’s first update, its count becomes one, so $Q\leftarrow Q+(R-Q)/1=R$: the initial value disappears completely for that arm. Optimism encourages early trials but does not guarantee correct identification, especially with noisy rewards.

#### 4–5. Implement constant step sizes and a mean switch

Apply both extensions below to the source file `src/learning_rl/lessons/lesson03.py`. Replace `experiment()` with this version; the result tuple remains compatible with the original reporting code.

```python
def experiment(means, steps, epsilon, initial, seed, run,
               alpha=None, switch_step=None):
    actions_rng = random.Random(seed + 2 * run)
    rewards_rng = random.Random(seed + 2 * run + 1)
    current_means = list(means)  # Do not mutate the CLI's list across runs.
    estimates = [initial] * len(means)
    counts = [0] * len(means)
    reward_sum = regret = 0.0
    optimal_count = 0
    for t in range(steps):
        if switch_step is not None and t == switch_step:
            worst = min(range(len(current_means)), key=current_means.__getitem__)
            best_arm = max(range(len(current_means)), key=current_means.__getitem__)
            current_means[worst], current_means[best_arm] = (
                current_means[best_arm], current_means[worst])
        if actions_rng.random() < epsilon:
            action = actions_rng.randrange(len(means))
        else:
            highest = max(estimates)
            ties = [a for a, q in enumerate(estimates) if q == highest]
            action = actions_rng.choice(ties)
        reward = rewards_rng.gauss(current_means[action], 1.0)
        counts[action] += 1
        rate = 1.0 / counts[action] if alpha is None else alpha
        estimates[action] += rate * (reward - estimates[action])
        best = max(current_means)
        reward_sum += reward
        regret += best - current_means[action]
        optimal_count += int(current_means[action] == best)
    return reward_sum / steps, regret, optimal_count / steps, estimates, counts
```

Add these arguments before parsing:

```python
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--switch-step", type=int, default=None)
```

After parsing, add validation:

```python
    if args.alpha is not None and not 0 < args.alpha <= 1:
        parser.error("alpha must be in (0, 1]")
    if args.switch_step is not None and not 0 <= args.switch_step < args.steps:
        parser.error("switch-step must be in [0, steps)")
```

Inside the epsilon loop in `main()`, replace the `results = ...` assignment with:

```python
        results = [experiment(args.means, args.steps, epsilon, args.initial,
                              args.seed, run, args.alpha, args.switch_step)
                   for run in range(args.runs)]
```

For exercise 4, compare estimates on the stationary task:

```bash
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 1 --steps 2000 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 1 --steps 2000 --seed 7 --alpha 0.1
```

The update with a supplied step size is $Q\leftarrow Q+\alpha(R-Q)$. Sample averages give equal weight to past observations and become less noisy as an arm’s count grows. Constant step sizes keep responding to recent rewards and retain fluctuations even after many pulls. With $\alpha=0.1$, the initial estimate’s weight after $n$ arm observations is $0.9^n$, so it no longer disappears after the first observation. Since different estimates can lead to different actions, equal seeds do not imply identical trajectories across these strategies.

For the two stationary commands above, one reference run gives final sample-average estimates `[0.1634, 0.1355, 1.0063]` and constant-step estimates `[0.0943, 0.3516, 0.9152]`. Both identify arm two as the current estimated favorite, but their final estimates differ because they weight past rewards differently.

For exercise 5, set the switch halfway through the run:

```bash
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 200 --steps 4000 --switch-step 2000 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 200 --steps 4000 --switch-step 2000 --seed 7 --alpha 0.1
python -m learning_rl.lessons.lesson03 --epsilons 0.1 --runs 1 --steps 4000 --switch-step 2000 --seed 7 --alpha 0.1
```

Pulls are indexed from zero: pulls 0–1999 use the original means, and pull 2000 onward uses the swapped means. The default vector changes from `[0, 0.2, 1]` to `[1, 0.2, 0]`. Sample averages retain the earlier observations with substantial weight; constant step sizes can adapt faster when positive epsilon keeps supplying trials of the newly best arm. Compare mean reward, pseudo-regret and optimal fraction across runs, and inspect the final estimates with `--runs 1`. A finite experiment need not show a guaranteed ranking.

The first two switching commands give these reference results over the whole 4,000-pull horizon:

| Update rule | Mean reward | Mean pseudo-regret | Optimal fraction |
| --- | ---: | ---: | ---: |
| Sample average | 0.5118 | 1953.25 | 0.505 |
| Constant step size 0.1 | 0.9270 | 292.37 | 0.918 |

Here the constant step size adapts much better. Whole-run averages include the stationary first half as well as adaptation after the switch; they are not post-switch-only measurements.

Regret in the extended program is $\sum_t[\max_a q_t(a)-q_t(A_t)]$, measured against the best currently available arm, rather than the best fixed arm over the full horizon. Recomputing `best` inside the loop implements that changing comparator. Although swapping preserves the numeric maximum here, the identity of the best arm and each selected arm’s mean change.
