import argparse
import random

from learning_rl.lessons.output import add_output_argument, lesson_output

from learning_rl.environments.lineworld import transition


def evaluate(size, gamma, right_prob, theta, max_iterations):
    values = [0.0] * size
    for iteration in range(1, max_iterations + 1):
        new = values.copy()
        for state in range(size - 1):
            new[state] = 0.0
            for action, prob in [(0, 1 - right_prob), (1, right_prob)]:
                nxt, reward, done = transition(state, action, size)
                new[state] += prob * (reward + gamma * (0.0 if done else values[nxt]))
        delta = max(abs(a - b) for a, b in zip(new, values))
        values = new
        if delta < theta:
            return values, iteration
    raise RuntimeError("Evaluation did not converge; increase --max-iterations")


def sample(size, gamma, right_prob, episodes, max_steps, rng):
    sums = [0.0] * size
    counts = [0] * size
    truncated = 0
    for _ in range(episodes):
        state = 0
        trajectory = []
        done = False
        for _ in range(max_steps):
            action = int(rng.random() < right_prob)
            nxt, reward, done = transition(state, action, size)
            trajectory.append((state, reward))
            state = nxt
            if done:
                break
        truncated += int(not done)
        # Every-visit MC: each visit contributes its observed suffix return.
        # A truncated suffix is deliberately assigned zero continuation here.
        G = 0.0
        for state, reward in reversed(trajectory):
            G = reward + gamma * G
            sums[state] += G
            counts[state] += 1
    values = [sums[s] / counts[s] if counts[s] else None for s in range(size)]
    values[-1] = 0.0
    return values, counts, truncated


def main():
    parser = argparse.ArgumentParser(description="Estimate or compute LineWorld policy values.")
    parser.add_argument("--method", choices=["mc", "dp", "compare"], default="compare")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--right-prob", type=float, default=0.5)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    add_output_argument(parser)
    args = parser.parse_args()
    if args.size < 2 or min(args.episodes, args.max_steps, args.max_iterations) < 1:
        parser.error("size >= 2 and positive count arguments required")
    if not 0 <= args.gamma < 1 or not 0 <= args.right_prob <= 1 or not args.theta > 0:
        parser.error("gamma in [0, 1), right-prob in [0, 1], theta > 0 required")
    with lesson_output("lesson02", args.output_dir):
        if args.method in ("dp", "compare"):
            values, iterations = evaluate(args.size, args.gamma, args.right_prob,
                                          args.theta, args.max_iterations)
            print(f"DP iterations={iterations}")
            for state, value in enumerate(values):
                print(f"state={state} dp_value={value:.6f}")
        if args.method in ("mc", "compare"):
            estimates, counts, truncated = sample(args.size, args.gamma, args.right_prob,
                                                  args.episodes, args.max_steps,
                                                  random.Random(args.seed))
            print(f"MC truncated_episodes={truncated}/{args.episodes}")
            for state, estimate in enumerate(estimates):
                label = "unvisited" if estimate is None else f"{estimate:.6f}"
                print(f"state={state} mc_value={label} visits={counts[state]}")
            if truncated:
                print("WARNING: zero continuation at time limits biases full-episode MC estimates.")


if __name__ == "__main__":
    main()
