# Lesson 1 — Reinforcement Learning Foundations: Agents, Environments, and MDPs — Version 2 — Hands-on (Python)

> **Concepts and theoretical exercises:** [RL-lesson-01-rl-foundations](RL-lesson-01-rl-foundations-01.md)  
> **Study pace:** Allow 45–75 minutes for the coding lab and practical exercises.  
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

## Setup and commands

Use Python 3.10+ and run every command in this tutorial from the **repository root**. Follow the [environment setup in the README](../../README.md), then install the course in editable mode:

```bash
python -m pip install -e .
```

For Lesson 1 plots, use `python -m pip install -e ".[plots]"`. Text runs require no runtime dependencies beyond the standard library. Editable installation lets commands use your source changes without reinstalling; no copying baseline programs out of Markdown is needed.

The walkthrough uses the supplied baseline program. Exercises and their solutions follow at the end. Solution code remains in this document: modification snippets apply to the linked lesson source; standalone solution programs can be saved under a local `scratch/` directory. Create that directory before saving a program there. It is excluded from version control.

### Save this lesson's results

Group experiments under `output/lesson01/RUN_NAME/`. Create a run directory
before redirecting output. For example, in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path output/lesson01/comparison | Out-Null
```

On macOS/Linux, use `mkdir -p output/lesson01/comparison`. Save a comparison log:

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet > output/lesson01/comparison/run.log
```

Redirection saves printed output; it does not display it. Use a new run name when
changing settings to preserve the previous result. See [output conventions](../../output/README.md)
for displaying and saving simultaneously. Add `--plots output/lesson01/comparison/plots` before the redirection to save the figures beside the log.

## 1. Guided tutorial — Explore LineWorld

Imagine a courier walking along five numbered stops. Position $0$ is the depot, positions $1,2,3$ are intermediate stops, and position $4$ is the destination.

| Rule | Meaning |
| --- | --- |
| Start | Position $0$ |
| Actions | Left (coded as $0$) or right (coded as $1$) |
| Left at position $0$ | Remain at $0$; it still counts as a transition |
| Transition not reaching $4$ | Reward $-1$ |
| Transition entering $4$ | Reward $+10$, then terminate |

Going straight right gives rewards $-1,-1,-1,10$. The final move gives $10$ total, rather than $10-1$. This small world lets us inspect every decision without the extra details of a real delivery problem.

Before running the program, predict how many transitions it takes to finish and how a left move changes that number. The program implements specified policies; it does not learn a better one yet.

### Find the implementation

| File | Responsibility |
| --- | --- |
| [Lesson 1 runner](../../src/learning_rl/lessons/lesson01.py) | `Episode` records, `run_episode()`, summaries, and CLI arguments |
| [LineWorld](../../src/learning_rl/environments/lineworld.py) | Environment reset, step, and deterministic transition model |
| [Lesson 1 plots](../../src/learning_rl/plotting/lesson01.py) | Five plotting functions consuming recorded episodes |

The runner records actions and outcomes before calculating summaries or generating plots. Recording and plotting do not change the random action sequence. The environment reports termination; the runner marks a cutoff as truncation. A cutoff stops the recorded return and does not estimate missing continuation.

### Run from your terminal

First inspect the CLI, then run one traced episode and a batch comparison:

```bash
python -m learning_rl.lessons.lesson01 --help
python -m learning_rl.lessons.lesson01 --policy right --trace
python -m learning_rl.lessons.lesson01 --compare --episodes 100 --seed 7 --quiet
```

For the default always-right run, expect four steps, a reward sum of `7`, and discounted return `4.58` at $\gamma=0.9$:

$$
G_0=-1-0.9-0.9^2+10(0.9^3)=4.58.
$$

Every command ends with a summary table. `reward_sum`, `return`, and `steps` are means over **all** recorded episodes, including truncated ones. `success_rate` is the fraction reaching the goal; `truncated` is a count. `--quiet` suppresses individual episode summaries; an explicitly requested `--trace` still prints transitions. `--compare` runs both right and random policies with the same environment settings and episode count, overriding `--policy`.

`--seed` reproduces random runs in the same Python environment. Recording and plotting consume no random draws. Nothing is learning: the policies are fixed.

### Optional plotting

Install Matplotlib in the Python environment you use for this lab:

```bash
python -m pip install -e ".[plots]"
python -m learning_rl.lessons.lesson01 --compare --episodes 100 --seed 7 --quiet --plots output/lesson01/comparison/plots
```

`--plots DIRECTORY` saves PNG files without opening windows. Open the saved images in an image viewer. Each command uses the same filenames within its output directory, so use distinct directories for experiments you want to keep. Without `--plots`, Matplotlib is neither imported nor required.

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

## Coding convention for this course

Every coding lab provides a complete command-line entry point, `--help`, configurable experiment arguments, and example terminal commands. Coding exercises should retain this interface when you extend a program. Optional plots supplement traces and summary tables; text runs remain available without plotting dependencies.

## 2. Practical exercises — Inspect and compare fixed policies

For exercises 1–4, use the supplied program. Plots are optional: the summary tables and traces provide a text alternative.

1. **Compare policies and estimate stability.** Run `--compare` with 100 and then 1,000 episodes. Compare mean return, length, success rate and truncation count. Inspect the comparison and running-mean plots if available. Does collecting more episodes train either policy? Explain why a more stable estimate need not be a better score.
2. **Recognize truncation visually.** Run always-right with `--max-steps 2`. Inspect its trajectory and reward plots, or its trace and summary table. Identify the last state and ending status. Is the final cumulative reward the full task return?
3. **Change the discount while preserving a trajectory.** Run always-right with discounts 0, 0.5 and 1. Compare the tables and reward plots; inspect its discount plot. Verify that states and actions do not change. Repeat with a random policy using the same seed and all other settings unchanged.
4. **Compare reward rules.** Compare movement cost / goal reward `(-1, 10)` with sparse rewards `(0, 1)` at $\gamma=1$, using both policies and identical seeds. Use separate plot directories, or compare the tables. Which rule distinguishes fast and slow successful episodes? Then try $\gamma=0.9$ with sparse rewards. Do the fixed policies change their actions?
5. **Extend the random policy.** Add `--right-prob`, choosing right with `rng.random() < args.right_prob`. Validate the probability, retain the plotting and summary interfaces, and include a run command. Compare probabilities 0, 0.5, 0.8 and 1 using tables or separate plot directories. Test the extension with `--compare` too.

### Solutions — Exercises 1–5

Try each exercise before reading its solution. To use plots, install Matplotlib and append the indicated `--plots` option; omit it for text-only runs.

#### 1. Compare policies and estimate stability

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 100 --seed 7 --quiet --plots output/lesson01/100/plots
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --plots output/lesson01/1000/plots
```

Always-right takes four steps, earns total reward `7`, and has return `4.58` at the default discount, with success rate one. Random episodes have variable lengths and returns. More episodes generally give a more stable estimate; they do not train this fixed policy or guarantee a better mean. The running mean can still fluctuate.

Without a cutoff, the expected number of moves from position zero for a uniform random policy is $4(4+1)=20$, and the expected **undiscounted** reward sum is $10-(20-1)=-9$. With the 100-step cutoff, lengths are capped and some runs may fail to finish. These unlimited-run quantities need not equal the reported means; also, discounted return and undiscounted reward sum differ.

#### 2. Recognize truncation

```bash
python -m learning_rl.lessons.lesson01 --policy right --max-steps 2 --trace --plots output/lesson01/cutoff/plots
```

The trajectory is $0\to1\to2$, with rewards $-1,-1$. Expect `steps=2`, `reward_sum=-2.000`, `return=-1.900`, `terminated=False`, `truncated=True`, and success rate zero. The trajectory ends with an X at position two, below the goal. The final cumulative reward is a partial sum, not an estimate of the unobserved continuation. Reaching position four requires four moves.

#### 3. Change the discount

```bash
python -m learning_rl.lessons.lesson01 --policy right --gamma 0 --trace --plots output/lesson01/gamma0/plots
python -m learning_rl.lessons.lesson01 --policy right --gamma 0.5 --trace --plots output/lesson01/gamma05/plots
python -m learning_rl.lessons.lesson01 --policy right --gamma 1 --trace --plots output/lesson01/gamma1/plots
```

Actions and states remain identical because this policy does not use the discount. For rewards $[-1,-1,-1,10]$, the returns are respectively $-1$, $-0.5$, and $7$. The reward sum remains $7$. The immediate reward panels match, but cumulative discounted reward changes. The discount plot rescales the same recorded rewards without resampling actions. With a random policy, keeping the same seed and all other arguments also preserves trajectories when only the discount changes.

#### 4. Change reward design

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 1 --step-reward -1 --goal-reward 10 --plots output/lesson01/cost/plots
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 1 --step-reward 0 --goal-reward 1 --plots output/lesson01/sparse/plots
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --gamma 0.9 --step-reward 0 --goal-reward 1 --plots output/lesson01/sparse-discounted/plots
```

Always-right earns `7` with movement costs and `1` with sparse rewards at $\gamma=1$. For a successful trajectory of length $T$, the respective undiscounted returns are $11-T$ and $1$. The first rule favors shorter successful paths; the second gives all successes the same return. A truncated sparse-reward run earns zero, so its mean return at $\gamma=1$ equals its success rate. At $0<\gamma<1$, a successful sparse-reward trajectory has return $\gamma^{T-1}$, favoring earlier completion. Changing rewards or discount alone does not change either fixed policy's actions. Compare returns within each reward rule; their scales differ across rules.

#### 5. Add a probability of moving right

In `src/learning_rl/lessons/lesson01.py`, add this argument in `main()` before parsing:

```python
    parser.add_argument("--right-prob", type=float, default=0.5)
```

After parsing, validate it:

```python
    if not 0 <= args.right_prob <= 1:
        parser.error("right-prob must be in [0, 1]")
```

Extend the runner signature:

```python
def run_episode(env, policy, rng, max_steps, right_prob=0.5):
```

Replace its action-selection line with:

```python
        action = 1 if policy == "right" else int(rng.random() < right_prob)
```

In `main()`, pass the argument into the runner:

```python
            episode = run_episode(env, policy, rng, args.max_steps, args.right_prob)
```

Run separate experiments (omit `--plots` for tables only):

```bash
python -m learning_rl.lessons.lesson01 --help
python -m learning_rl.lessons.lesson01 --compare --right-prob 0 --episodes 1000 --quiet --plots output/lesson01/p0/plots
python -m learning_rl.lessons.lesson01 --compare --right-prob 0.5 --episodes 1000 --quiet --plots output/lesson01/p05/plots
python -m learning_rl.lessons.lesson01 --compare --right-prob 0.8 --episodes 1000 --quiet --plots output/lesson01/p08/plots
python -m learning_rl.lessons.lesson01 --compare --right-prob 1 --episodes 1000 --quiet --plots output/lesson01/p1/plots
```

Probability one reproduces the four-step right trajectory; probability zero stays at position zero and truncates. Probability 0.8 should generally shorten episodes relative to 0.5. The probability affects only the random policy, including when invoked through `--compare`. Existing plot and table functions consume the same episode records and need no changes. Using `rng.random()` instead of `rng.randrange(2)` changes the seeded draw sequence, so the extension need not reproduce the original random trajectories even at probability 0.5.
