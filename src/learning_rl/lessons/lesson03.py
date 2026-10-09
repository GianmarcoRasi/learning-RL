import argparse
import random
import statistics

from learning_rl.lessons.output import add_output_argument, lesson_output


def experiment(means, steps, epsilon, initial, seed, run):
    """Learn arm values using epsilon-greedy and sample-average updates.

    True means are used for reward generation and pseudo-regret evaluation,
    never for action selection. Each step is one bandit pull.
    """
    # Separate streams: reward noise is independent of action-selection draws.
    actions_rng = random.Random(seed + 2 * run)
    rewards_rng = random.Random(seed + 2 * run + 1)
    estimates = [initial] * len(means)
    counts = [0] * len(means)
    reward_sum = pseudo_regret = 0.0
    optimal_count = 0
    best = max(means)
    for _ in range(steps):
        if actions_rng.random() < epsilon:
            action = actions_rng.randrange(len(means))
        else:
            highest = max(estimates)
            ties = [a for a, q in enumerate(estimates) if q == highest]
            action = actions_rng.choice(ties)
        reward = rewards_rng.gauss(means[action], 1.0)
        counts[action] += 1
        estimates[action] += (reward - estimates[action]) / counts[action]
        reward_sum += reward
        pseudo_regret += best - means[action]
        optimal_count += int(means[action] == best)
    return reward_sum / steps, pseudo_regret, optimal_count / steps, estimates, counts


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Compare epsilon-greedy Gaussian bandits.")
    parser.add_argument("--means", type=float, nargs="+", default=[0.0, 0.2, 1.0])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.1, 1.0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--initial", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    add_output_argument(parser)
    args = parser.parse_args(argv)
    import math
    if args.steps < 1 or args.runs < 1:
        parser.error("steps and runs must be positive")
    if not all(math.isfinite(m) for m in args.means) or not math.isfinite(args.initial):
        parser.error("means and initial must be finite")
    if not all(0 <= e <= 1 for e in args.epsilons):
        parser.error("epsilons must be in [0, 1]")
    with lesson_output("lesson03", args.output_dir):
        for epsilon in args.epsilons:
            results = [experiment(args.means, args.steps, epsilon, args.initial, args.seed, run)
                       for run in range(args.runs)]
            rewards = [r[0] for r in results]
            se = statistics.stdev(rewards) / args.runs ** 0.5 if args.runs > 1 else 0.0
            print(f"epsilon={epsilon:g} mean_reward={statistics.mean(rewards):.4f} "
                  f"reward_SE={se:.4f} "
                  f"mean_pseudo_regret={statistics.mean(r[1] for r in results):.2f} "
                  f"optimal_fraction={statistics.mean(r[2] for r in results):.3f}")
            if args.runs == 1:
                print("estimates=", [round(q, 4) for q in results[0][3]])
                print("counts=", results[0][4])


if __name__ == "__main__":
    main()
