"""Project commands for the supplied reinforcement learning lessons."""

import argparse
import importlib
import sys


LESSONS = {
    "lesson01": "Reinforcement learning foundations: fixed LineWorld policies",
    "lesson02": "Value functions: Monte Carlo and iterative policy evaluation",
    "lesson03": "Multi-armed bandits: epsilon-greedy exploration",
    "lesson04": "Dynamic programming control: policy and value iteration",
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Run the reinforcement learning course lessons.",
        epilog="Use %(prog)s lesson01 --help for a lesson's experiment arguments.",
    )
    parser.add_argument("lesson", choices=LESSONS, help="; ".join(
        f"{name}: {description}" for name, description in LESSONS.items()))
    # Parse only the lesson selector; the lesson owns all remaining arguments.
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.parse_args(arguments)
        return
    selected = parser.parse_args(arguments[:1])
    module = importlib.import_module(f"learning_rl.lessons.{selected.lesson}")
    module.main(arguments[1:])
