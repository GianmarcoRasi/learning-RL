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
