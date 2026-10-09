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

Generated results are local and ignored by Git.

## Choose a run directory

Add `--output-dir PATH` to a lesson command to choose where its `run.log` is saved.
The program creates the directory automatically:

```bash
python -m learning_rl.lessons.lesson01 --output-dir output/lesson01/my-run
python -m learning_rl.lessons.lesson02 --output-dir output/lesson02/my-run
python -m learning_rl.lessons.lesson03 --output-dir output/lesson03/my-run
python -m learning_rl.lessons.lesson04 --output-dir output/lesson04/my-run
```

Use distinct names for different settings. Reusing a directory overwrites `run.log`.
