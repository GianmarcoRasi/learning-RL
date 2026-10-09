# Lesson 3 — Multi-Armed Bandits: Exploration and Exploitation — Version 2 — Hands-on (Python)

> **Vocabulary:** Use the [project glossary](../docs/glossary.md) for the terms and notation shared by all lessons and code.

> **Concepts and theoretical exercises:** [RL-lesson-03-bandits-exploration](RL-lesson-03-bandits-exploration-01.md)  
> **Study pace:** Allow 45–75 minutes for the experiment lab and practical exercises.
>
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

After installation, run lesson commands from any directory with the project virtual environment activated. Relative output paths are resolved from your current directory. Follow the [virtual environment setup](../README.md#create-a-local-virtual-environment) and [package installation instructions](../README.md#install-packages) in the README before starting. The standard installation includes Matplotlib for plotting in current and subsequent lessons.

The walkthrough and exercises use the supplied command-line programs. Run experiments with their existing arguments, then report the requested outputs and interpretations. No source edits or new programs are required. Exercises and solutions follow at the end.

### Lesson output

Lesson commands automatically save printed results to `output/lesson03/TIMESTAMP/run.log`
and display them in the terminal. Each run gets a new directory. Use `--output-dir`
to choose a directory; reusing it overwrites its log. See [output conventions](../docs/command-outputs.md).

## 1. Guided tutorial — Run the lab

### Find the implementation

The [Lesson 3 source](../src/learning_rl/lessons/lesson03.py) contains `experiment()` for one bandit run and `main()` for comparisons across epsilon settings and repeated runs. Each arm returns a Gaussian reward with its supplied mean and standard deviation one. The agent sees rewards; simulator metrics use hidden means. Random tie breaking avoids favoring arm zero.

The `--initial` estimate affects early action selection. With sample averages, an arm's first reward replaces its initial estimate. Optimism encourages early exploration but cannot guarantee finding the best arm.

The lab reuses seed schedules across strategies. This can correlate their results, so separate standard errors alone are not a test of the difference between strategies. Assess strategy differences with that paired structure in mind.

### Terminal commands

```bash
python -m RL lesson03 --help
python -m RL lesson03 --epsilons 0 0.1 1 --steps 2000 --runs 200 --seed 7
python -m RL lesson03 --epsilons 0.1 --runs 1 --steps 100 --seed 7
python -m RL lesson03 --epsilons 0 --initial 5 --runs 200
python -m RL lesson03 --means 0 0.9 1 --epsilons 0 0.01 0.1 --steps 5000
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

Include your commands and a table of mean reward, reward standard error, mean pseudo-regret and optimal fraction for exercises 1-3.

1. **Horizon:** compare `--steps 20`, `200` and `2000`, holding other settings fixed. When does exploration start paying for itself? Report means over at least 100 runs.
2. **Difficulty:** compare means `[0,0.2,1]` and `[0,0.9,1]`. Explain why identifying the best arm is harder when the gap is small, even if mistakes then cost less.
3. **Optimism:** compare greedy runs with `--initial 0` and `--initial 5`. Explain why initial values disappear from an arm after its first sample-average update.
4. **Inspect arm estimates and counts.** Run `--epsilons 0 0.1 1 --runs 1 --steps 100 --seed 7`. Provide the estimates and counts for each epsilon, identify the arm with the largest estimate, and verify that each count vector sums to 100. Explain why the estimated favorite can differ from the true best arm.
5. **Measure uncertainty across runs.** Compare `--runs 20` and `200` with `--epsilons 0.1 --steps 2000 --seed 7`. Provide both summary rows and calculate the ratio of their reward standard errors. Explain why more runs improve precision without adding training steps to any individual agent.

### Solutions — Exercises 1–5

#### 1. Compare horizons

```bash
python -m RL lesson03 --steps 20 --runs 200 --epsilons 0 0.1 1 --seed 7
python -m RL lesson03 --steps 200 --runs 200 --epsilons 0 0.1 1 --seed 7
python -m RL lesson03 --steps 2000 --runs 200 --epsilons 0 0.1 1 --seed 7
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
python -m RL lesson03 --means 0 0.2 1 --steps 2000 --runs 200 --epsilons 0 0.1 --seed 7
python -m RL lesson03 --means 0 0.9 1 --steps 2000 --runs 200 --epsilons 0 0.1 --seed 7
```

The best-versus-second-best mean gap falls from $0.8$ to $0.1$, while reward noise still has standard deviation one. More observations are needed to distinguish nearly equal means. However, choosing arm one instead of arm two then costs only $0.1$ per pull instead of $0.8$. A lower optimal fraction can therefore coexist with a high mean reward and low regret. Report all three printed metrics; optimal fraction alone does not describe the cost of mistakes.

#### 3. Use optimistic initial values

```bash
python -m RL lesson03 --epsilons 0 --initial 0 --steps 2000 --runs 200 --seed 7
python -m RL lesson03 --epsilons 0 --initial 5 --steps 2000 --runs 200 --seed 7
```

In a reference run, mean reward increases from approximately $0.7863$ to $0.8469$. Untried arms retain their optimistic value and can attract greedy choices after a tried arm receives a lower reward. On an arm’s first update, its count becomes one, so $Q\leftarrow Q+(R-Q)/1=R$: the initial value disappears completely for that arm. Optimism encourages early trials but does not guarantee correct identification, especially with noisy rewards.

#### 4. Inspect arm estimates and counts

```bash
python -m RL lesson03 --epsilons 0 0.1 1 --runs 1 --steps 100 --seed 7
```

Each strategy prints a summary, an `estimates=` vector and a `counts=` vector. Copy these vectors into your answer; each count vector sums to 100. The true best arm is index two, with mean one. Reward noise and limited exploration can give another arm the highest estimate. Pure greedy can leave arms untried, while uniform exploration spreads samples across arms. A standard error of zero for one run is a reporting convention, not evidence of zero uncertainty.

#### 5. Measure uncertainty across runs

```bash
python -m RL lesson03 --epsilons 0.1 --runs 20 --steps 2000 --seed 7
python -m RL lesson03 --epsilons 0.1 --runs 200 --steps 2000 --seed 7
```

Report the printed metrics and divide the 20-run standard error by the 200-run standard error. If the empirical standard deviations were equal, the ratio would be $\sqrt{200/20}=\sqrt{10}\approx3.16$; measured standard deviations can differ. Each agent still takes 2,000 pulls. More independent runs improve estimation of strategy performance, rather than continuing one agent's training. Shared seed schedules mean separate standard errors do not directly test differences between epsilon settings.
