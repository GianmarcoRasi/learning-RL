"""Checks for shared rules, published reference results, and module entry points."""

import contextlib
import importlib
import io
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning_rl.environments.lineworld import LineWorld, transition
from learning_rl.lessons import lesson01, lesson02, lesson03, lesson04


@contextlib.contextmanager
def temporary_working_directory():
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            yield
        finally:
            os.chdir(previous)


class LessonChecks(unittest.TestCase):
    def test_shared_model_and_goal_reward(self):
        env = LineWorld()
        self.assertEqual(env.step(0), (0, -1.0, False))
        for state in range(4):
            self.assertEqual(env.step(1), transition(state, 1, 5))
        with self.assertRaises(RuntimeError):
            env.step(1)
        with self.assertRaises(ValueError):
            transition(4, 1, 5)

    def test_recorded_return_and_cutoff(self):
        episode = lesson01.run_episode(LineWorld(), "right", random.Random(7), 100)
        self.assertEqual(episode.states, [0, 1, 2, 3, 4])
        self.assertEqual(episode.rewards, [-1, -1, -1, 10])
        for gamma, expected in [(0, -1), (0.5, -0.5), (0.9, 4.58), (1, 7)]:
            self.assertAlmostEqual(episode.discounted_return(gamma), expected)
        cutoff = lesson01.run_episode(LineWorld(), "right", random.Random(7), 2)
        self.assertTrue(cutoff.truncated)
        self.assertEqual(cutoff.states, [0, 1, 2])
        self.assertAlmostEqual(cutoff.discounted_return(0.9), -1.9)

    def test_rewards_do_not_change_fixed_policy(self):
        a = lesson01.run_episode(LineWorld(), "random", random.Random(7), 100)
        b = lesson01.run_episode(LineWorld(step_reward=0, goal_reward=1),
                                "random", random.Random(7), 100)
        self.assertEqual(a.states, b.states)
        self.assertEqual(a.actions, b.actions)

    def test_dp_reference_values_and_sweeps(self):
        values, _ = lesson02.evaluate(5, 0.9, 1, 1e-10, 100000)
        for actual, expected in zip(values, [4.58, 6.2, 8, 10, 0]):
            self.assertAlmostEqual(actual, expected)
        for gamma, sweeps in [(0.5, 31), (0.9, 134), (0.99, 308)]:
            _, actual = lesson02.evaluate(5, gamma, 0.5, 1e-10, 100000)
            self.assertEqual(actual, sweeps)

    def test_mc_truncation_suffixes(self):
        values, counts, truncated = lesson02.sample(5, 0.9, 0, 1, 3, random.Random(7))
        self.assertAlmostEqual(values[0], -1.87)
        self.assertEqual(counts, [3, 0, 0, 0, 0])
        self.assertEqual(truncated, 1)
        self.assertIsNone(values[1])

    def test_bandit_seed_and_counts(self):
        arguments = ([0, 0.2, 1], 100, 0.1, 0, 7, 0)
        result = lesson03.experiment(*arguments)
        self.assertEqual(result, lesson03.experiment(*arguments))
        self.assertEqual(sum(result[4]), 100)
        self.assertGreaterEqual(result[1], 0)

    def test_control_algorithms_agree(self):
        for size in (5, 8):
            pi = lesson04.policy_iteration(size, 0.9, 1e-10, 100000)
            vi = lesson04.value_iteration(size, 0.9, 1e-10, 100000)
            for a, b in zip(pi[0], vi[0]):
                self.assertAlmostEqual(a, b)
            self.assertEqual(pi[1], [1.0] * (size - 1))
            self.assertEqual(pi[1], vi[1])
        self.assertEqual(lesson04.value_iteration(5, 0.9, 1e-10, 100000)[2], 5)

    def test_all_clis_help_and_examples(self):
        examples = [(lesson01, ["--compare", "--episodes", "10", "--quiet"], "4.580"),
                    (lesson02, ["--method", "dp", "--right-prob", "1"], "4.580000"),
                    (lesson03, ["--runs", "1", "--steps", "20"], "mean_pseudo_regret"),
                    (lesson04, ["--algorithm", "both"], "residual=")]
        for module, arguments, expected in examples:
            with self.subTest(module=module.__name__):
                output = io.StringIO()
                with patch.object(sys, "argv", [module.__name__, "--help"]), \
                     contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit:
                    module.main()
                self.assertEqual(exit.exception.code, 0)
                self.assertIn("usage:", output.getvalue())
                output = io.StringIO()
                with patch.object(sys, "argv", [module.__name__] + arguments), \
                     contextlib.redirect_stdout(output):
                    module.main()
                self.assertIn(expected, output.getvalue())

    def test_cli_logs_are_saved_by_lesson_and_run(self):
        examples = [(lesson01, ["--quiet"]),
                    (lesson02, ["--method", "dp"]),
                    (lesson03, ["--runs", "1", "--steps", "10"]),
                    (lesson04, [])]
        with temporary_working_directory():
            for module, arguments in examples:
                with self.subTest(module=module.__name__):
                    root = Path("output") / module.__name__.rsplit(".", 1)[-1]
                    with patch.object(sys, "argv", [module.__name__, "--help"]), \
                         contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                        module.main()
                    self.assertFalse(root.exists())
                    for _ in range(2):
                        output = io.StringIO()
                        with patch.object(sys, "argv", [module.__name__] + arguments), \
                             contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                            module.main()
                        self.assertTrue(any(log.read_text(encoding="utf-8") == output.getvalue()
                                            for log in root.glob("*/run.log")))
                    self.assertEqual(len(list(root.glob("*/run.log"))), 2)
                    output = io.StringIO()
                    with patch.object(sys, "argv", [module.__name__] + arguments +
                                      ["--output-dir", str(root / "custom")]), \
                         contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                        module.main()
                    self.assertEqual((root / "custom/run.log").read_text(encoding="utf-8"),
                                     output.getvalue())


class PlotChecks(unittest.TestCase):
    def test_all_plot_types_and_saved_cli_output(self):
        import matplotlib
        matplotlib.use("Agg")
        plots = importlib.import_module("learning_rl.plotting.lesson01")
        right = lesson01.run_episode(LineWorld(), "right", random.Random(7), 100)
        cutoff = lesson01.run_episode(LineWorld(), "right", random.Random(7), 2)
        experiments = {"right": [right], "cutoff": [cutoff]}
        figures = [plots.plot_trajectory(cutoff, 4), plots.plot_rewards(right, 0.9),
                   plots.plot_comparison(experiments, 0.9), plots.plot_discount(right),
                   plots.plot_running_mean(experiments, 0.9)]
        self.assertAlmostEqual(figures[1].axes[1].lines[0].get_ydata()[-1], 4.58)
        with tempfile.TemporaryDirectory() as output:
            for i, fig in enumerate(figures):
                fig.savefig(Path(output) / f"plot-{i}.png")
                plots.plotting().close(fig)
            with patch.object(sys, "argv", ["lesson01", "--compare", "--episodes", "2",
                                            "--quiet", "--plot-episodes", "1", "--plots", output]), \
                 contextlib.redirect_stdout(io.StringIO()):
                lesson01.main()
            self.assertEqual(len(list(Path(output).glob("*.png"))), 13)
        self.assertFalse(plots.plotting().get_fignums())


if __name__ == "__main__":
    unittest.main()
