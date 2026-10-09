import argparse

from learning_rl.lessons.output import add_output_argument, lesson_output

from learning_rl.environments.lineworld import transition


def action_value(state, action, values, size, gamma):
    """Compute one-step lookahead Q_V using the supplied state-value table."""
    next_state, reward, terminated = transition(state, action, size)
    return reward + gamma * (0.0 if terminated else values[next_state])


def greedy(state, values, size, gamma):
    left = action_value(state, 0, values, size, gamma)
    right = action_value(state, 1, values, size, gamma)
    return int(right > left)  # Stable tie rule: choose left on exact equality.


def evaluate(policy, size, gamma, theta, max_iterations):
    """Evaluate the fixed policy with synchronous Bellman expectation sweeps."""
    values = [0.0] * size
    for iteration in range(1, max_iterations + 1):
        new = values.copy()
        for state in range(size - 1):
            p_right = policy[state]
            new[state] = ((1 - p_right) * action_value(state, 0, values, size, gamma)
                          + p_right * action_value(state, 1, values, size, gamma))
        delta = max(abs(x - y) for x, y in zip(new, values))
        values = new
        if delta < theta:
            return values, iteration
    raise RuntimeError("Policy evaluation did not converge")


def policy_iteration(size, gamma, theta, max_iterations):
    """Alternate policy evaluation and greedy policy improvement."""
    policy = [0.5] * (size - 1)
    sweeps = 0
    for round_number in range(1, max_iterations + 1):
        values, count = evaluate(policy, size, gamma, theta, max_iterations)
        sweeps += count
        improved = [float(greedy(s, values, size, gamma)) for s in range(size - 1)]
        if improved == policy:
            return values, improved, round_number, sweeps
        policy = improved
    raise RuntimeError("Policy iteration did not stabilize")


def value_iteration(size, gamma, theta, max_iterations):
    """Perform synchronous optimality sweeps, then extract a greedy policy."""
    values = [0.0] * size
    for iteration in range(1, max_iterations + 1):
        new = values.copy()
        for state in range(size - 1):
            new[state] = max(action_value(state, a, values, size, gamma) for a in (0, 1))
        delta = max(abs(x - y) for x, y in zip(new, values))
        values = new
        if delta < theta:
            policy = [float(greedy(s, values, size, gamma)) for s in range(size - 1)]
            return values, policy, iteration, iteration
    raise RuntimeError("Value iteration did not converge")


def main(argv=None, *, prog=None):
    parser = argparse.ArgumentParser(prog=prog, description="Find optimal LineWorld policies with DP.")
    parser.add_argument("--algorithm", choices=["policy", "value", "both"], default="both")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100000)
    add_output_argument(parser)
    args = parser.parse_args(argv)
    if args.size < 2 or args.max_iterations < 1:
        parser.error("size >= 2 and max-iterations >= 1 required")
    if not 0 <= args.gamma < 1 or not args.theta > 0:
        parser.error("gamma in [0, 1) and theta > 0 required")
    with lesson_output("lesson04", args.output_dir):
        for name, algorithm in [("policy", policy_iteration), ("value", value_iteration)]:
            if args.algorithm not in (name, "both"):
                continue
            values, policy, rounds, sweeps = algorithm(args.size, args.gamma, args.theta,
                                                      args.max_iterations)
            residual = max(abs(values[s] - max(action_value(s, a, values, args.size,
                                                            args.gamma) for a in (0, 1)))
                           for s in range(args.size - 1))
            print(f"algorithm={name} rounds={rounds} sweeps={sweeps} residual={residual:.3g}")
            for state, value in enumerate(values):
                action = "terminal" if state == args.size - 1 else ("right" if policy[state] else "left")
                print(f"state={state} value={value:.6f} action={action}")


if __name__ == "__main__":
    main()
