# Lesson 1 — Reinforcement Learning Foundations: Agents, Environments, and MDPs — Version 2 — Hands-on (Python)

> **Concepts and theoretical exercises:** [RL-lesson-01-rl-foundations](RL-lesson-01-rl-foundations.md)  
> **Study pace:** Allow 45–75 minutes for the coding lab and practical exercises.  
> **Prerequisites:** Read the companion lesson; Python loops, lists and functions.

These are the original lesson’s command-line labs and practical exercises. The programs use Python 3.10+ and only the standard library. Run commands from the directory containing your saved script; on systems without `python`, use `python3`.

## 6. Practical lab --- Build a tiny environment

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

### Complete command-line program

Save the following as `lesson01.py`. It uses only the Python 3.10+ standard library; no packages or notebook are required. Run commands from the directory containing your saved script. On systems without `python`, use `python3`.

The environment reports true termination. The runner separately reports **truncation** when its safety limit is reached. A time limit is not evidence that the goal was reached. The reported discounted return stops at that limit; it does not estimate unobserved future rewards.

```python
import argparse
import random


class LineWorld:
    def __init__(self, size=5, step_reward=-1.0, goal_reward=10.0):
        if size < 2:
            raise ValueError("size must be at least 2")
        self.size = size
        self.goal = size - 1
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == self.goal:
            raise RuntimeError("Episode terminated: call reset()")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (left) or 1 (right)")
        self.state = max(0, self.state - 1) if action == 0 else self.state + 1
        terminated = self.state == self.goal
        reward = self.goal_reward if terminated else self.step_reward
        return self.state, reward, terminated


def main():
    parser = argparse.ArgumentParser(description="Run a fixed or random LineWorld agent.")
    parser.add_argument("--policy", choices=["right", "random"], default="right")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-reward", type=float, default=-1.0)
    parser.add_argument("--goal-reward", type=float, default=10.0)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.size < 2 or args.episodes < 1 or args.max_steps < 1:
        parser.error("size >= 2, episodes >= 1, and max-steps >= 1 required")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must be in [0, 1]")
    rng = random.Random(args.seed)
    env = LineWorld(args.size, args.step_reward, args.goal_reward)
    totals, returns, lengths = [], [], []
    successes = 0
    for episode in range(args.episodes):
        state = env.reset()
        total = discounted = 0.0
        terminated = False
        for t in range(args.max_steps):
            action = 1 if args.policy == "right" else rng.randrange(2)
            next_state, reward, terminated = env.step(action)
            total += reward
            discounted += args.gamma ** t * reward
            if args.trace:
                print(f"episode={episode + 1} state={state} action={action} "
                      f"next_state={next_state} reward={reward:g}")
            state = next_state
            if terminated:
                break
        steps = t + 1
        successes += int(terminated)
        totals.append(total)
        returns.append(discounted)
        lengths.append(steps)
        print(f"episode={episode + 1} steps={steps} reward_sum={total:.3f} "
              f"return={discounted:.3f} terminated={terminated} "
              f"truncated={not terminated}")
    print(f"mean_reward_sum={sum(totals)/args.episodes:.3f} "
          f"mean_return={sum(returns)/args.episodes:.3f} "
          f"mean_steps={sum(lengths)/args.episodes:.3f} "
          f"success_rate={successes/args.episodes:.3f}")


if __name__ == "__main__":
    main()
```

### Run from your terminal

```bash
python lesson01.py --help
python lesson01.py --policy right --trace
python lesson01.py --policy random --episodes 100 --max-steps 100 --seed 7
python lesson01.py --policy random --episodes 100 --step-reward 0 --goal-reward 1
```

For the default always-right run, expect four steps, a reward sum of `7`, and discounted return `4.58` at $\gamma=0.9$:

$$
G_0=-1-0.9-0.9^2+10(0.9^3)=4.58.
$$

The script reports every episode and a final summary. `--seed` reproduces random runs in the same Python environment. Nothing is learning yet: the policy is specified by the user.

## 7. Practical exercise — Random agent

1. Run 100 random episodes, then 1,000. Compare mean return, length and success rate.
2. Use `--max-steps 2` with the right policy. Why are all episodes truncated?
3. Compare `--gamma 0`, `--gamma 0.5` and `--gamma 1`. Does the trajectory change?
4. Compare step reward `-1`, goal reward `10` with step reward `0`, goal reward `1`.
5. Add a `--right-prob` argument, using `rng.random()` to choose right with that probability. Preserve `--help`, validate the probability and show a run command.

### Solutions — Exercises 1–5

Try each exercise before using these solutions. Commands below use the original program unless explicitly described as an extension.

#### 1. Compare random runs

```bash
python lesson01.py --policy random --episodes 100 --seed 7
python lesson01.py --policy random --episodes 1000 --seed 7
```

Compare the final `mean_return`, `mean_steps` and `success_rate` lines. More episodes generally give a more stable estimate; they do not train this fixed random policy or guarantee a better mean. Without a cutoff, the expected number of moves from position zero is $4(4+1)=20$, and the expected undiscounted reward sum is $10-(20-1)=-9$. With the 100-step cutoff, lengths are capped and some runs may fail to finish, so the reported means need not equal these unlimited-run quantities.

#### 2. Explain truncation

```bash
python lesson01.py --policy right --max-steps 2 --trace
```

The trajectory is $0\to1\to2$, with two rewards of $-1$. Reaching position four requires four moves. Expect `steps=2`, `reward_sum=-2.000`, `return=-1.900`, `terminated=False`, `truncated=True`, and a success rate of zero.

#### 3. Change the discount

```bash
python lesson01.py --policy right --gamma 0 --trace
python lesson01.py --policy right --gamma 0.5 --trace
python lesson01.py --policy right --gamma 1 --trace
```

The actions and states remain identical because this policy does not use the discount. For rewards $[-1,-1,-1,10]$, the returns are respectively $-1$, $-0.5$, and $7$. The reward sum remains $7$. With a random policy, keeping the same seed and other arguments also preserves the trajectory when only the discount changes.

#### 4. Change reward design

```bash
python lesson01.py --policy right --gamma 1 --step-reward -1 --goal-reward 10
python lesson01.py --policy right --gamma 1 --step-reward 0 --goal-reward 1
python lesson01.py --policy random --episodes 1000 --seed 7 --gamma 1 --step-reward 0 --goal-reward 1
```

The right policy earns $7$ with movement costs and $1$ with sparse rewards. For a successful trajectory of length $T$, the respective undiscounted returns are $11-T$ and $1$. A truncated sparse-reward run earns zero, so its mean return at $\gamma=1$ equals its success rate. At $\gamma<1$, a successful sparse-reward trajectory has return $\gamma^{T-1}$. Changing rewards alone does not change this fixed policy’s actions.

#### 5. Add a probability of moving right

In `main()`, add this argument before `args = parser.parse_args()`:

```python
    parser.add_argument("--right-prob", type=float, default=0.5)
```

After parsing, add validation:

```python
    if not 0 <= args.right_prob <= 1:
        parser.error("right-prob must be in [0, 1]")
```

Replace the existing action-selection line inside the step loop with:

```python
            action = 1 if args.policy == "right" else int(rng.random() < args.right_prob)
```

```bash
python lesson01.py --help
python lesson01.py --policy random --right-prob 0.8 --episodes 1000 --seed 7
python lesson01.py --policy random --right-prob 1 --trace
python lesson01.py --policy random --right-prob 0 --max-steps 5 --trace
```

Probability one reproduces the four-step right trajectory; probability zero stays at position zero and truncates. The probability affects only `--policy random`. Using `rng.random()` instead of `rng.randrange(2)` changes the seeded draw sequence, so the extended program need not reproduce the original random trajectories even at probability $0.5$.

### Interpretation

A uniform random walk on this finite line reaches the goal with probability one if allowed to run indefinitely, but it has no fixed upper bound on episode length. A finite safety limit can therefore truncate episodes. A policy that always moves left never reaches the goal.

With sparse rewards and $\gamma=1$, all successful trajectories earn the same return regardless of length. With $\gamma<1$, earlier goal rewards are worth more. Reward design changes the objective as well as the feedback available to learning algorithms.

------------------------------------------------------------------------

## Coding convention for this course

Every coding lab provides a complete command-line entry point, `--help`, configurable experiment arguments, and example terminal commands. Coding exercises should retain this interface when you extend a program.
