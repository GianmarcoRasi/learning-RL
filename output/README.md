# Command outputs

Run lesson commands from the repository root after following the
[environment setup](../README.md#create-a-local-virtual-environment) and
[package installation instructions](../README.md#install-packages).

Each lesson command automatically creates a directory and saves its printed
results to `output/lessonNN/TIMESTAMP/run.log`. Results also appear in the terminal.
Timestamps include microseconds so separate runs preserve previous results.
Help and argument validation errors do not create logs; errors stay in the terminal.

```text
output/
  README.md
  lesson01/
    TIMESTAMP/
      run.log
  lesson02/
    TIMESTAMP/
      run.log
  lesson03/
    TIMESTAMP/
      run.log
  lesson04/
    TIMESTAMP/
      run.log
```

Only this README is tracked by Git. Generated results are local.

## Choose a run directory

Use `--output-dir` to give an experiment a descriptive name. The program creates
the directory automatically; no shell redirection or directory creation is needed:

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --output-dir output/lesson01/comparison
python -m learning_rl.lessons.lesson02 --method compare --episodes 10000 --seed 7 --output-dir output/lesson02/comparison
python -m learning_rl.lessons.lesson03 --epsilons 0 0.1 1 --steps 2000 --runs 200 --seed 7 --output-dir output/lesson03/comparison
python -m learning_rl.lessons.lesson04 --algorithm both --output-dir output/lesson04/comparison
```

Use distinct names for different settings. Reusing a directory overwrites `run.log`.
Record the command and its settings alongside the log when keeping an experiment
for later, for example in `command.txt`.

## Optional Lesson 1 plots

Follow the [optional plotting installation instructions](../README.md#install-packages).
Use `--plots` to choose where PNG files are saved, for example:

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --output-dir output/lesson01/comparison --plots output/lesson01/comparison/plots
```

Plot directories are created automatically. Reusing a plots directory overwrites
its plot filenames. Without `--plots`, no figures are generated.

## Existing results

Plots previously generated in `plots/comparison/` have been preserved under
`lesson01/previous-root-plots/plots/`. The separate set formerly in
`lesson01/plots/comparison/` is under `lesson01/previous-lesson01-plots/plots/`.
Their original settings were not recorded, so do not assume they describe the
same experiment or compare them as if all settings were identical.
