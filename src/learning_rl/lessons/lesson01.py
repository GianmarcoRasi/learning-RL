import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from learning_rl.lessons.output import add_output_argument, lesson_output, save_summary, terminal_output

from learning_rl.environments.lineworld import LineWorld
from learning_rl.plotting.lesson01 import (
    plotting, plot_trajectory, plot_rewards, plot_comparison,
    plot_discount, plot_running_mean,
)


@dataclass
class Episode:
    """Recorded trajectory; truncation excludes unobserved continuation."""
    states: list[int]  # Includes S_0: one more state than rewards.
    actions: list[int]
    rewards: list[float]
    terminated: bool

    @property
    def truncated(self):
        return not self.terminated

    def discounted_return(self, gamma):
        """Return the recorded return, not a policy's expected state value."""
        return sum(gamma ** t * r for t, r in enumerate(self.rewards))


def run_episode(env, policy, rng, max_steps):
    states, actions, rewards = [env.reset()], [], []
    terminated = False
    for _ in range(max_steps):
        action = 1 if policy == "right" else rng.randrange(2)
        state, reward, terminated = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if terminated:
            break
    return Episode(states, actions, rewards, terminated)


def print_summary(experiments, gamma):
    rows = []
    print(f"\nSummary (gamma={gamma:g}; returns stop at termination or cutoff)")
    print(f"{'policy':<10} {'episodes':>8} {'reward_sum':>12} {'return':>12} "
          f"{'steps':>10} {'success_rate':>13} {'truncated':>10}")
    for label, episodes in experiments.items():
        row = {
            "policy": label, "episodes": len(episodes),
            "reward_sum": mean(sum(e.rewards) for e in episodes),
            "return": mean(e.discounted_return(gamma) for e in episodes),
            "steps": mean(len(e.actions) for e in episodes),
            "success_rate": mean(e.terminated for e in episodes),
            "truncated": sum(e.truncated for e in episodes),
        }
        rows.append(row)
        print(f"{label:<10} {row['episodes']:>8} "
              f"{row['reward_sum']:>12.3f} {row['return']:>12.3f} "
              f"{row['steps']:>10.3f} {row['success_rate']:>13.3f} "
              f"{row['truncated']:>10}")
    print("reward_sum, return and steps are means over all recorded episodes.")
    return rows


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Run and compare fixed LineWorld policies.")
    parser.add_argument("--policy", choices=["right", "random"], default="right")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-reward", type=float, default=-1.0)
    parser.add_argument("--goal-reward", type=float, default=10.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Suppress episode summaries; keep table and any requested trace.")
    parser.add_argument("--compare", action="store_true", help="Run right and random; overrides --policy.")
    parser.add_argument("--plots", type=Path, help="Save five kinds of PNG plots in this directory.")
    parser.add_argument("--plot-episodes", type=int, default=3,
                        help="Episodes per policy to plot individually (default: 3).")
    add_output_argument(parser)
    args = parser.parse_args(argv)
    if args.size < 2 or args.episodes < 1 or args.max_steps < 1 or args.plot_episodes < 1:
        parser.error("size >= 2; episodes, max-steps and plot-episodes >= 1 required")
    if not 0 <= args.gamma <= 1:
        parser.error("gamma must be in [0, 1]")
    if not all(math.isfinite(r) for r in (args.step_reward, args.goal_reward)):
        parser.error("rewards must be finite")
    if args.plots:
        try:
            plotting()
        except ImportError:
            parser.error("Plotting requires Matplotlib: python -m pip install matplotlib")
    with lesson_output("lesson01", args.output_dir) as directory:
        experiments = {}
        for policy in (["right", "random"] if args.compare else [args.policy]):
            rng = random.Random(args.seed)
            env = LineWorld(args.size, args.step_reward, args.goal_reward)
            episodes = []
            for number in range(1, args.episodes + 1):
                episode = run_episode(env, policy, rng, args.max_steps)
                episodes.append(episode)
                if args.trace:
                    for t, (action, reward) in enumerate(zip(episode.actions, episode.rewards)):
                        print(f"policy={policy} episode={number} state={episode.states[t]} "
                              f"action={action} next_state={episode.states[t+1]} reward={reward:g}")
                if not args.quiet:
                    print(f"policy={policy} episode={number} steps={len(episode.actions)} "
                          f"reward_sum={sum(episode.rewards):.3f} "
                          f"return={episode.discounted_return(args.gamma):.3f} "
                          f"terminated={episode.terminated} truncated={episode.truncated}")
            experiments[policy] = episodes
        with terminal_output():
            rows = print_summary(experiments, args.gamma)
        save_summary(directory,
                     ["policy", "episodes", "reward_sum", "return", "steps", "success_rate", "truncated"],
                     rows)
        if args.plots:
            args.plots.mkdir(parents=True, exist_ok=True)
            plt = plotting()

            def save(fig, name):
                fig.savefig(args.plots / name, dpi=150)
                plt.close(fig)

            for policy, episodes in experiments.items():
                for number, episode in enumerate(episodes[:args.plot_episodes], 1):
                    label = f"{policy}, episode {number}"
                    save(plot_trajectory(episode, args.size - 1, label), f"{policy}-{number}-trajectory.png")
                    save(plot_rewards(episode, args.gamma, label), f"{policy}-{number}-rewards.png")
                save(plot_discount(episodes[0]), f"{policy}-1-discount.png")
            save(plot_comparison(experiments, args.gamma), "comparison.png")
            save(plot_running_mean(experiments, args.gamma), "running-mean.png")
            print(f"Plots saved to {args.plots.resolve()}")


if __name__ == "__main__":
    main()
