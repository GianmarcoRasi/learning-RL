# Command outputs

Run commands from the repository root. Store generated results by lesson and run:

```text
output/
  README.md
  lesson01/
    comparison/
      run.log
      plots/
        comparison.png
        running-mean.png
        ...
  lesson02/
    comparison/
      run.log
  lesson03/
    comparison/
      run.log
  lesson04/
    comparison/
      run.log
```

Only this README is tracked by Git. Generated results are local. Record the command
and its settings alongside the log when keeping an experiment for later, for example
in `command.txt`. Use distinct run names for different episode counts, seeds, reward
rules, or discounts. Reusing a run name overwrites its files.

## Create run directories

PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path output/lesson01/comparison, output/lesson02/comparison, output/lesson03/comparison, output/lesson04/comparison | Out-Null
```

macOS/Linux:

```bash
mkdir -p output/lesson01/comparison output/lesson02/comparison output/lesson03/comparison output/lesson04/comparison
```

## Save printed results

These commands work in PowerShell and bash after activating the environment and
installing the course. `>` saves standard output to a file; errors stay visible in
the terminal. The directory must exist before redirecting output.

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet > output/lesson01/comparison/run.log
python -m learning_rl.lessons.lesson02 --method compare --episodes 10000 --seed 7 > output/lesson02/comparison/run.log
python -m learning_rl.lessons.lesson03 --epsilons 0 0.1 1 --steps 2000 --runs 200 --seed 7 > output/lesson03/comparison/run.log
python -m learning_rl.lessons.lesson04 --algorithm both > output/lesson04/comparison/run.log
```

To generate Lesson 1 figures as well, install the optional plotting dependency and
add `--plots output/lesson01/comparison/plots` before the `>` symbol. The program
creates the plots directory automatically, but the redirected log's parent still
needs to exist first.

## Display and save simultaneously

PowerShell:

```powershell
python -m learning_rl.lessons.lesson04 --algorithm both | Tee-Object -FilePath output/lesson04/comparison/run.log
```

macOS/Linux:

```bash
python -m learning_rl.lessons.lesson04 --algorithm both | tee output/lesson04/comparison/run.log
```

## Existing results

Plots previously generated in `plots/comparison/` have been preserved under
`lesson01/previous-root-plots/plots/`. The separate set formerly in
`lesson01/plots/comparison/` is under `lesson01/previous-lesson01-plots/plots/`.
Their original settings were not recorded, so do not assume they describe the
same experiment or compare them as if all settings were identical.
