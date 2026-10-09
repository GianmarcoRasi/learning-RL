# Lesson 1 — Reinforcement Learning Foundations: Agents, Environments, and MDPs — Version 2 — Hands-on (Python)

> **Vocabulary:** Use the [project glossary](../docs/glossary.md) for the terms and notation shared by all lessons and code.

> **Concepts and theoretical exercises:** [RL-lesson-01-rl-foundations](../lessons/RL-lesson-01-rl-foundations-01.md)\
> **Study pace:** Allow 45–75 minutes for the experiment lab and practical exercises.
>
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

> **Code:** Lesson programs are in `src/learning_rl/lessons/lessonNN.py`, where `NN` is the two-digit lesson number (for this lesson, `lesson01.py`).

## Setup and commands

After installation, run lesson commands from any directory with the project virtual environment activated. Relative output paths are resolved from your current directory. Follow the [virtual environment setup](../README.md#create-a-local-virtual-environment) and [package installation instructions](../README.md#install-packages) in the README before starting.

The walkthrough and exercises use the supplied command-line programs. Run experiments with their existing arguments, then report the requested outputs and interpretations. Exercises and solutions follow at the end.

```bash
RL lesson01 --help
```

All options are optional. An option that takes a value requires that value when supplied. Flags take no value.

| Option | Required | Values | Role |
| --- | --- | --- | --- |
| `-h`, `--help` | optional | No value; flag. | Display help and exit. |
| `--policy` | optional | Placeholder: `POLICY`; one [registered policy](#registered-policy-values). Default: `right`. | Select the fixed policy. |
| `--size` | optional | Placeholder: `SIZE`; integer ≥ 2. Default: `5`. | Number of positions, including start and goal. |
| `--episodes` | optional | Placeholder: `EPISODES`; integer ≥ 1. Default: `1`. | Episodes to run per policy. |
| `--max-steps` | optional | Placeholder: `MAX_STEPS`; integer ≥ 1. Default: `100`. | Maximum transitions per episode before truncation. |
| `--gamma` | optional | Placeholder: `GAMMA`; finite real number in [0, 1], dimensionless. Default: `0.9`. | Discount factor used to calculate returns. |
| `--seed` | optional | Placeholder: `SEED`; any integer. Default: `7`. | Seed for reproducible random actions. |
| `--step-reward` | optional | Placeholder: `STEP_REWARD`; finite real number. Default: `-1`. | Reward for transitions that do not enter the goal. |
| `--goal-reward` | optional | Placeholder: `GOAL_REWARD`; finite real number. Default: `10`. | Reward for entering the goal. |
| `--trace` | optional | No value; flag. Default: disabled. | Print every recorded transition. |
| `--quiet` | optional | No value; flag. Default: disabled. | Suppress episode summaries; retain the final table and any requested trace. |
| `--compare` | optional | No value; flag. Default: disabled. | Run both registered policies; override `--policy`. |
| `--plots` | optional | Placeholder: `PLOTS_DIRECTORY`; relative or absolute directory path. Default: no plots. | Save PNG plots in this directory, creating it if needed. |
| `--plot-episodes` | optional | Placeholder: `PLOT_EPISODES`; integer ≥ 1. Default: `3`. | Maximum episodes per policy for individual trajectory and reward plots; used with `--plots`. |
| `--output-dir` | optional | Placeholder: `OUTPUT_DIRECTORY`; relative or absolute directory path. Default: `output/lesson01/TIMESTAMP/`. | Save `summary.csv` and `run.log`, creating the directory if needed. Reusing a directory overwrites these files. |

### Registered policy values

| Value of `POLICY` | Role |
| --- | --- |
| `right` | Always move right toward the goal. |
| `random` | Choose left or right with equal probability at each step. |

### Lesson output

Lesson commands automatically save summary tables to `output/lesson01/TIMESTAMP/summary.csv`
and other output to `run.log` in the same directory. Tables also appear in the terminal;
the log points to the CSV instead of duplicating the tables.
Each run gets a new directory. Use `--output-dir` to choose a directory; reusing it
overwrites its CSV and log. See [output conventions](../docs/command-outputs.md).

## 1. Guided tutorial — Explore LineWorld

Imagine a courier walking along five numbered stops. Position $0$ is the depot, positions $1,2,3$ are intermediate stops, and position $4$ is the destination.

```text
   Courier starts here
           ↓
       [0] ─── [1] ─── [2] ─── [3] ─── [4]
      Depot    Intermediate stops     Destination
                                        Terminal
```

The environment starts each episode in state $0$. At each decision step, the agent chooses one of two actions: left (coded as $0$) or right (coded as $1$). The next state is the neighboring position in that direction, except that choosing left in state $0$ keeps the agent in state $0$. This still counts as a transition.

Every transition that does not enter state $4$ gives reward $-1$. Entering state $4$ gives reward $+10$ and terminates the episode. State $4$ is therefore the terminal state; no further actions or rewards follow.

```mermaid
flowchart
    S0["0 · Start / depot"]
    S1["1"]
    S2["2"]
    S3["3"]
    S4(["4 · Goal / terminal"])

    S0 -->|"Left (0): −1"| S0
    S0 -->|"Right (1): −1"| S1
    S1 -->|"Left (0): −1"| S0
    S1 -->|"Right (1): −1"| S2
    S2 -->|"Left (0): −1"| S1
    S2 -->|"Right (1): −1"| S3
    S3 -->|"Left (0): −1"| S2
    S3 -->|"Right (1): +10"| S4

    classDef intermediate fill:#1e293b,stroke:#94a3b8,color:#f1f5f9
    classDef start fill:#172554,stroke:#60a5fa,color:#eff6ff
    classDef terminal fill:#052e16,stroke:#4ade80,color:#f0fdf4
    class S1,S2,S3 intermediate
    class S0 start
    class S4 terminal
    linkStyle default stroke:#94a3b8,color:#f1f5f9
```

Each numbered position is a state. Left at state $0$ leaves the agent there and still earns $-1$. Reaching state $4$ ends the episode; it has no outgoing actions. Arrow labels show the action name, its code, and the reward for that transition.

Going straight right gives rewards $-1,-1,-1,10$. The final move gives $10$ total, rather than $10-1$. This small world lets us inspect every decision without the extra details of a real delivery problem.

Before running the program, predict how many transitions it takes to finish and how a left move changes that number. The program implements specified policies.

### Run from your terminal

Run one traced episode and a batch comparison:

```bash
RL lesson01 --policy right --trace
RL lesson01 --compare --episodes 100 --seed 7 --quiet
```

For the default always-right run, expect four steps, a reward sum of `7`, and discounted return `4.58` at $\gamma=0.9$:

$$
G_0=-1-0.9-0.9^2+10(0.9^3)=4.58.
$$

Every command ends with a summary table. `reward_sum`, `return`, and `steps` are means over **all** recorded episodes, including truncated ones. `success_rate` is the fraction reaching the goal; `truncated` is a count. `--quiet` suppresses individual episode summaries; an explicitly requested `--trace` still prints transitions. `--compare` runs both right and random policies with the same environment settings and episode count, overriding `--policy`.

`--seed` reproduces random runs in the same Python environment. Recording and plotting consume no random draws. Nothing is learning: the policies are fixed.

### Generate plots

Generate plots with:

```bash
RL lesson01 --compare --episodes 100 --seed 7 --quiet --plots output/lesson01/comparison/plots
```

`--plots DIRECTORY` saves PNG files without opening windows. Open the saved images in an image viewer. Each command uses the same filenames within its output directory, so use distinct directories for experiments you want to keep. Without `--plots`, the command produces text output only.

| Function | Saved file | What to inspect |
| --- | --- | --- |
| `plot_trajectory(episode, goal, label)` | `right-1-trajectory.png`, etc. | Position against decision step, including initial state $S_0$. A star marks termination; an X marks truncation. The dashed line is the goal. |
| `plot_rewards(episode, gamma, label)` | `right-1-rewards.png`, etc. | Immediate rewards and cumulative discounted reward in separate panels. |
| `plot_comparison(experiments, gamma)` | `comparison.png` | Box plots of recorded returns and lengths, and bars for success rates. The title includes truncation counts. |
| `plot_discount(episode)` | `right-1-discount.png`, etc. | The same recorded trajectory scored at discounts from zero to one. |
| `plot_running_mean(experiments, gamma)` | `running-mean.png` | Running estimates of each fixed policy's mean recorded return. This is not a learning curve. |

By default, trajectory and reward plots are saved for the first three episodes per policy. `--plot-episodes 1` limits these to the first episode. The discount plot always uses the first episode; comparison and running-mean plots use every episode. An experiment is a dictionary mapping policy labels to lists of `Episode` records. Each record contains states, actions, rewards, and termination status. Plot functions return a Matplotlib figure, so they can also be called directly from Python; close the figure after saving it.

The cumulative reward panel shows

$$
C_n=\sum_{t=0}^{n-1}\gamma^t R_{t+1},\qquad C_0=0.
$$

Its final value equals the reported recorded return. Intermediate points accumulate rewards from the start; they are not future returns $G_t$. For a truncated episode, the final point excludes unobserved future rewards.

Compare policies under identical size, reward rule, discount and step limit. Box plots show variation across episodes; always-right produces a constant value. Lengths of truncated episodes equal the cutoff, so the length plot does not estimate how long those episodes would eventually take to finish.

### Interpretation

A uniform random walk on this finite line reaches the goal with probability one if allowed to run indefinitely, but it has no fixed upper bound on episode length. A finite safety limit can therefore truncate episodes. A policy that always moves left never reaches the goal.

With sparse rewards and $\gamma=1$, all successful trajectories earn the same return regardless of length. With $\gamma<1$, earlier goal rewards are worth more. Reward design changes the objective as well as the feedback available to learning algorithms.

------------------------------------------------------------------------

## Hands-on convention for this course

Every hands-on lab lives in `lessons-hands-on/` and provides a complete command-line entry point, `--help`, configurable experiment arguments, and example terminal commands. Under "Setup and commands", place the help command before a table of all options with Required, Values, and Role columns. Define placeholders without angle brackets and include accepted values and defaults in Values; place any linked tables of registered values immediately afterward. Exercises ask you to run the supplied code and provide specific outputs and explanations. Plots supplement traces and summary tables; plotting dependencies are installed by default. Future hands-on lessons should follow this convention.

## 2. Practical exercises — Inspect and compare fixed policies

For all exercises, use the supplied program. Include your commands, summary rows and requested plots or traces in your answers. Plots are optional: the summary tables and traces provide a text alternative.

1. **Compare policies and estimate stability.** Run `--compare` with 100 and then 1,000 episodes. Compare mean return, length, success rate and truncation count. Inspect the comparison and running-mean plots if available. Does collecting more episodes train either policy? Explain why a more stable estimate need not be a better score.
2. **Recognize truncation visually.** Run always-right with `--max-steps 2`. Inspect its trajectory and reward plots, or its trace and summary table. Identify the last state and ending status. Is the final cumulative reward the full task return?
3. **Change the discount while preserving a trajectory.** Run always-right with discounts 0, 0.5 and 1. Compare the tables and reward plots; inspect its discount plot. Verify that states and actions do not change. Repeat with a random policy using the same seed and all other settings unchanged.
4. **Compare reward rules.** Compare movement cost / goal reward `(-1, 10)` with sparse rewards `(0, 1)` at $\gamma=1$, using both policies and identical seeds. Use separate plot directories, or compare the tables. Which rule distinguishes fast and slow successful episodes? Then try $\gamma=0.9$ with sparse rewards. Do the fixed policies change their actions?
5. **Compare route lengths.** Run `--compare --episodes 1000 --seed 7 --quiet` for sizes 3, 5 and 8. Provide a table with size, policy, mean return, mean steps, success rate and truncation count. Include a trajectory plot for always-right at each size. Explain why route length changes the return even though neither policy learns.

### Solutions — Exercises 1–5

Try each exercise before reading its solution. Append the indicated `--plots` option to save plots; omit it for text-only runs. Matplotlib is already included in the standard setup.

#### 1. Compare policies and estimate stability

```bash
RL lesson01 --compare --episodes 100 --seed 7 --quiet --plots output/lesson01/100/plots
RL lesson01 --compare --episodes 1000 --seed 7 --quiet --plots output/lesson01/1000/plots
```

Always-right takes four steps, earns total reward `7`, and has return `4.58` at the default discount, with success rate one. Random episodes have variable lengths and returns. More episodes generally give a more stable estimate; they do not train this fixed policy or guarantee a better mean. The running mean can still fluctuate.

Without a cutoff, the expected number of moves from position zero for a uniform random policy is $4(4+1)=20$, and the expected **undiscounted** reward sum is $10-(20-1)=-9$. With the 100-step cutoff, lengths are capped and some runs may fail to finish. These unlimited-run quantities need not equal the reported means; also, discounted return and undiscounted reward sum differ.

#### 2. Recognize truncation

```bash
RL lesson01 --policy right --max-steps 2 --trace --plots output/lesson01/cutoff/plots
```

The trajectory is $0\to1\to2$, with rewards $-1,-1$. Expect `steps=2`, `reward_sum=-2.000`, `return=-1.900`, `terminated=False`, `truncated=True`, and success rate zero. The trajectory ends with an X at position two, below the goal. The final cumulative reward is a partial sum, not an estimate of the unobserved continuation. Reaching position four requires four moves.

#### 3. Change the discount

```bash
RL lesson01 --policy right --gamma 0 --trace --plots output/lesson01/gamma0/plots
RL lesson01 --policy right --gamma 0.5 --trace --plots output/lesson01/gamma05/plots
RL lesson01 --policy right --gamma 1 --trace --plots output/lesson01/gamma1/plots
```

Actions and states remain identical because this policy does not use the discount. For rewards $[-1,-1,-1,10]$, the returns are respectively $-1$, $-0.5$, and $7$. The reward sum remains $7$. The immediate reward panels match, but cumulative discounted reward changes. The discount plot rescales the same recorded rewards without resampling actions. With a random policy, keeping the same seed and all other arguments also preserves trajectories when only the discount changes.

#### 4. Change reward design

```bash
RL lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 1 --step-reward -1 --goal-reward 10 --plots output/lesson01/cost/plots
RL lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 1 --step-reward 0 --goal-reward 1 --plots output/lesson01/sparse/plots
RL lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 0.9 --step-reward 0 --goal-reward 1 --plots output/lesson01/sparse-discounted/plots
```

Always-right earns `7` with movement costs and `1` with sparse rewards at $\gamma=1$. For a successful trajectory of length $T$, the respective undiscounted returns are $11-T$ and $1$. The first rule favors shorter successful paths; the second gives all successes the same return. A truncated sparse-reward run earns zero, so its mean return at $\gamma=1$ equals its success rate. At $0<\gamma<1$, a successful sparse-reward trajectory has return $\gamma^{T-1}$, favoring earlier completion. Changing rewards or discount alone does not change either fixed policy's actions. Compare returns within each reward rule; their scales differ across rules.

#### 5. Compare route lengths

```bash
RL lesson01 --compare --size 3 --episodes 1000 --seed 7 --quiet --plots output/lesson01/size3/plots
RL lesson01 --compare --size 5 --episodes 1000 --seed 7 --quiet --plots output/lesson01/size5/plots
RL lesson01 --compare --size 8 --episodes 1000 --seed 7 --quiet --plots output/lesson01/size8/plots
```

Copy both summary rows for each size into your results table and attach the `right-1-trajectory.png` images. Always-right takes 2, 4 and 7 moves respectively, succeeds every time, and has discounted returns `8`, `4.58` and `0.62882` at the default discount. Its reward sums are `9`, `7` and `4`. Longer routes add movement costs and delay the goal reward. Random runs vary; report the measured means and truncation counts rather than assuming every episode finishes within the default 100-step limit. The policies remain fixed throughout.
