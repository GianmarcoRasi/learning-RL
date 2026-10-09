"""Checks for shared rules, published reference results, and module entry points."""

import contextlib
import csv
import importlib
import io
import os
import random
import sys
import subprocess
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning_rl.environments.lineworld import LineWorld, transition
from learning_rl.lessons import lesson01, lesson02, lesson03, lesson04
from learning_rl import cli


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
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.test_home = Path(home.name)
        home_patch = patch.object(Path, "home", return_value=self.test_home)
        home_patch.start()
        self.addCleanup(home_patch.stop)

    def test_saved_paths_default_and_verbose_preserve_saved_results(self):
        examples = [
            (lesson01, [], "success_rate", "episode=1 steps=4"),
            (lesson01, ["--trace", "--quiet"], "success_rate", "next_state=4 reward=10"),
            (lesson02, ["--method", "mc", "--right-prob", "0", "--episodes", "1",
                        "--max-steps", "1"], "mc_value=", "WARNING: zero continuation"),
            (lesson03, ["--runs", "1", "--steps", "20"], "mean_reward=", "counts="),
            (lesson04, ["--algorithm", "both"], "action=", "Summary saved to"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for module, arguments, table_marker, log_marker in examples:
                with self.subTest(lesson=module.__name__):
                    directory = Path(temporary) / module.__name__.rsplit(".", 1)[-1]
                    saved = None
                    for verbose in (False, True):
                        output, errors = io.StringIO(), io.StringIO()
                        options = arguments + ["--output-dir", str(directory)]
                        if verbose:
                            options.append("--verbose")
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                            cli.main([directory.name] + options)
                            self.assertIs(sys.stdout, output)
                        log = (directory / "run.log").read_text(encoding="utf-8")
                        csv_text = (directory / "summary.csv").read_text(encoding="utf-8")
                        self.assertIn(log_marker, log)
                        self.assertNotIn(table_marker, log)
                        self.assertEqual(output.getvalue().count("Summary saved to"), 1)
                        self.assertEqual(errors.getvalue(), f"Output saved to {directory / 'run.log'}\n")
                        if verbose:
                            self.assertIn(table_marker, output.getvalue())
                            self.assertIn(log_marker, output.getvalue())
                            self.assertIn("Output saved to", errors.getvalue())
                            self.assertEqual((log, csv_text), saved)
                        else:
                            self.assertEqual(output.getvalue(), f"Summary saved to {directory / 'summary.csv'}\n")
                            saved = (log, csv_text)

    def test_csv_summary_values_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def run(module, arguments):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    module.main(arguments + ["--output-dir", str(directory)])
                with (directory / "summary.csv").open(encoding="utf-8", newline="") as file:
                    return list(csv.DictReader(file))

            rows = run(lesson01, ["--compare", "--episodes", "3", "--quiet"])
            self.assertEqual([row["policy"] for row in rows], ["right", "random"])
            self.assertEqual(int(rows[0]["episodes"]), 3)
            self.assertAlmostEqual(float(rows[0]["return"]), 4.58)
            self.assertEqual(float(rows[0]["reward_sum"]), 7)
            self.assertEqual(float(rows[0]["steps"]), 4)
            self.assertEqual(float(rows[0]["success_rate"]), 1)
            rows = run(lesson01, ["--quiet", "--max-steps", "2"])
            self.assertEqual(len(rows), 1)  # Reusing a directory replaces the previous CSV.
            self.assertEqual(int(rows[0]["truncated"]), 1)
            self.assertAlmostEqual(float(rows[0]["return"]), -1.9)
            self.assertEqual(float(rows[0]["success_rate"]), 0)

            rows = run(lesson02, ["--method", "compare", "--right-prob", "1", "--episodes", "2"])
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertAlmostEqual(float(row["value"]), [4.58, 6.2, 8, 10, 0][int(row["state"])])
            self.assertEqual(rows[0]["visits"], "")
            self.assertEqual(rows[5]["visits"], "2")
            self.assertEqual(rows[5]["truncated_episodes"], "0")
            rows = run(lesson02, ["--method", "mc", "--right-prob", "0", "--episodes", "1",
                                  "--max-steps", "3"])
            self.assertEqual(len(rows), 5)
            self.assertAlmostEqual(float(rows[0]["value"]), -1.87)
            self.assertEqual(rows[0]["truncated_episodes"], "1")
            self.assertEqual(rows[1]["value"], "")
            self.assertEqual(rows[-1]["value"], "0.0")
            self.assertEqual(rows[0]["iterations"], "")
            rows = run(lesson02, ["--method", "dp"])
            self.assertEqual({row["method"] for row in rows}, {"dp"})

            rows = run(lesson03, ["--runs", "1", "--steps", "20", "--epsilons", "0.1"])
            expected = lesson03.experiment([0, 0.2, 1], 20, 0.1, 0, 7, 0)
            self.assertEqual(len(rows), 1)
            for column, value in zip(["mean_reward", "mean_pseudo_regret", "optimal_fraction"], expected):
                self.assertEqual(float(rows[0][column]), value)
            self.assertEqual(float(rows[0]["reward_SE"]), 0)

            for selection, count in [("both", 10), ("policy", 5), ("value", 5)]:
                rows = run(lesson04, ["--algorithm", selection])
                self.assertEqual(len(rows), count)
                for row in rows:
                    state = int(row["state"])
                    self.assertAlmostEqual(float(row["value"]), [4.58, 6.2, 8, 10, 0][state])
                    self.assertEqual(row["action"], "terminal" if state == 4 else "right")
                    self.assertLess(float(row["residual"]), 1e-9)

    def test_project_dispatch_forwards_lesson_arguments(self):
        examples = [("lesson01", ["--quiet"], "4.580"),
                    ("lesson02", ["--method", "dp", "--right-prob", "1"], "4.580000"),
                    ("lesson03", ["--runs", "1", "--steps", "20"], "mean_pseudo_regret"),
                    ("lesson04", ["--algorithm", "both"], "residual=")]
        with temporary_working_directory():
            for name, arguments, expected in examples:
                with self.subTest(lesson=name):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        cli.main([name] + arguments + ["--verbose"])
                    self.assertIn(expected, output.getvalue())
                    root = self.test_home / "learning-rl-output" / name if name == "lesson01" else Path("output") / name
                    self.assertTrue(list(root.glob("*/run.log")))
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
                        cli.main([name, "--help"])
                    self.assertEqual(caught.exception.code, 0)

    def test_project_module_entry_points_outside_repository(self):
        # Use the installed package, without injecting repository paths.
        with temporary_working_directory():
            for module in ("RL", "learning_rl"):
                for options in ([], ["--verbose"]):
                    result = subprocess.run(
                        [sys.executable, "-m", module, "lesson01", "--quiet", "--output-dir", "run"] + options,
                        capture_output=True, text=True, check=True,
                    )
                    if options:
                        self.assertIn("4.580", result.stdout)
                    else:
                        self.assertRegex(result.stdout, r"^Summary saved to .+summary\.csv\n$")
            command = shutil.which("RL", path=str(Path(sys.executable).parent))
            self.assertIsNotNone(command, "Install the project to create the RL command")
            for options in ([], ["--verbose"]):
                result = subprocess.run(
                    [command, "lesson01", "--quiet", "--output-dir", "run"] + options,
                    capture_output=True, text=True, check=True,
                )
                if options:
                    self.assertIn("4.580", result.stdout)
                else:
                    self.assertRegex(result.stdout, r"^Summary saved to .+summary\.csv\n$")

    def test_project_help_and_invalid_arguments_do_not_create_logs(self):
        with temporary_working_directory():
            for arguments, expected_code in [(["--help"], 0), ([], 2),
                                              (["lesson99"], 2),
                                              (["lesson01", "--unknown"], 2)]:
                with self.subTest(arguments=arguments), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    cli.main(arguments)
                self.assertEqual(caught.exception.code, expected_code)
                self.assertFalse(Path("output").exists())

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
                with patch.object(sys, "argv", [module.__name__] + arguments + ["--verbose"]), \
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
                    root = (self.test_home / "learning-rl-output/lesson01" if module is lesson01 else
                            Path("output") / module.__name__.rsplit(".", 1)[-1])
                    with patch.object(sys, "argv", [module.__name__, "--help"]), \
                         contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                        module.main()
                    self.assertFalse(root.exists())
                    for _ in range(2):
                        output = io.StringIO()
                        with patch.object(sys, "argv", [module.__name__] + arguments + ["--verbose"]), \
                             contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                            module.main()
                        logs = [log.read_text(encoding="utf-8") for log in root.glob("*/run.log")]
                        self.assertTrue(any("Summary saved to " in log and
                                            all(line in output.getvalue() for line in log.splitlines())
                                            for log in logs))
                    self.assertEqual(len(list(root.glob("*/run.log"))), 2)
                    self.assertEqual(len(list(root.glob("*/summary.csv"))), 2)
                    output = io.StringIO()
                    with patch.object(sys, "argv", [module.__name__] + arguments +
                                      ["--output-dir", str(root / "custom"), "--verbose"]), \
                         contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                        module.main()
                    log = (root / "custom/run.log").read_text(encoding="utf-8")
                    self.assertIn(f"Summary saved to {root / 'custom' / 'summary.csv'}", log)
                    table_marker = {lesson01: "success_rate", lesson02: "dp_value=",
                                    lesson03: "mean_reward=", lesson04: "action="}[module]
                    self.assertIn(table_marker, output.getvalue())
                    self.assertNotIn(table_marker, log)
                    self.assertTrue(all(line in output.getvalue() for line in log.splitlines()))

    def test_logs_keep_traces_and_warnings_without_summary_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                lesson01.main(["--trace", "--quiet", "--output-dir", str(directory)])
            log = (directory / "run.log").read_text(encoding="utf-8")
            self.assertIn("state=3 action=1 next_state=4 reward=10", log)
            self.assertNotIn("Summary (gamma=", log)
            self.assertIn("summary.csv", log)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                lesson02.main(["--method", "mc", "--right-prob", "0", "--episodes", "1",
                               "--max-steps", "1", "--output-dir", str(directory)])
            log = (directory / "run.log").read_text(encoding="utf-8")
            self.assertIn("WARNING: zero continuation", log)
            self.assertNotIn("mc_value=", log)
            self.assertIn("summary.csv", log)


class PlotChecks(unittest.TestCase):
    def test_selection_preserves_batch_data_and_results(self):
        import matplotlib
        matplotlib.use("Agg")
        plots = importlib.import_module("learning_rl.plotting.lesson01")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["--compare", "--episodes", "7", "--quiet"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                lesson01.main(common + ["--output-dir", str(root / "baseline")])
                self.assertFalse((root / "baseline/plots").exists())
                with patch.object(lesson01, "plot_comparison", wraps=plots.plot_comparison) as comparison:
                    lesson01.main(common + ["--output-dir", str(root / "selected"),
                                           "--plot-types",
                                           "trajectory", "discount", "comparison",
                                           "--plot-episode-ids", "2", "7"])
            self.assertEqual({p.name for p in (root / "selected/plots").glob("*.png")},
                             {f"{policy}-{number}-{kind}.png"
                              for policy in ("right", "random") for number in (2, 7)
                              for kind in ("trajectory", "discount")} | {"comparison.png"})
            self.assertEqual([len(es) for es in comparison.call_args.args[0].values()], [7, 7])
            self.assertEqual((root / "baseline/summary.csv").read_bytes(),
                             (root / "selected/summary.csv").read_bytes())

    def test_default_figure_directory_is_independent_of_launch_directory(self):
        import matplotlib
        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(lesson01.Path, "home", return_value=root / "home"), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for _ in range(2):
                    with temporary_working_directory():
                        launched = Path.cwd()
                        lesson01.main(["--plot-types", "rewards"])
                        self.assertFalse(list(launched.rglob("*.png")))
            figures = list((root / "home/learning-rl-output/lesson01").glob("*/plots/right-1-rewards.png"))
            self.assertEqual(len(figures), 2)
            for figure in figures:
                self.assertTrue((figure.parent.parent / "summary.csv").exists())
                self.assertTrue((figure.parent.parent / "run.log").exists())

    def test_invalid_plot_selection_creates_no_output(self):
        invalid = [["--plot-episodes", "1"],
                   ["--plot-types", "unknown"],
                   ["--plot-types", "all", "--plot-episode-ids", "0"],
                   ["--plot-types", "all", "--episodes", "2", "--plot-episode-ids", "3"],
                   ["--plot-types", "all", "--plot-episodes", "1", "--plot-episode-ids", "1"],
                   ["--plots"]]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit), \
                     contextlib.redirect_stderr(io.StringIO()):
                    lesson01.main(arguments + ["--output-dir", str(output)])
                self.assertFalse(output.exists())

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
            terminal, errors = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", ["lesson01", "--compare", "--episodes", "2",
                                            "--quiet", "--plot-episodes", "1", "--plot-types", "all", "--output-dir", output]), \
                 contextlib.redirect_stdout(terminal), contextlib.redirect_stderr(errors):
                lesson01.main()
            self.assertEqual(len(terminal.getvalue().splitlines()), 2)
            self.assertRegex(terminal.getvalue().splitlines()[0], r"^Summary saved to .+summary\.csv$")
            self.assertEqual(terminal.getvalue().splitlines()[1], f"Plots saved to {Path(output).resolve() / 'plots'}")
            self.assertRegex(errors.getvalue(), r"^Output saved to .+run\.log\n$")
            self.assertEqual(len(list(Path(output).glob("*.png"))), 5)
            self.assertEqual(len(list((Path(output) / "plots").glob("*.png"))), 8)
        self.assertFalse(plots.plotting().get_fignums())


if __name__ == "__main__":
    unittest.main()
