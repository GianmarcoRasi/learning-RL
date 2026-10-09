# Command outputs

Run lesson commands from any directory in the activated project environment after following the
[environment setup](../README.md#create-a-local-virtual-environment) and
[package installation instructions](../README.md#install-packages).

Each lesson command automatically creates a directory and saves its printed
summary tables to `output/lessonNN/TIMESTAMP/summary.csv`. Tables also appear
in the terminal. The `run.log` in the same directory records other output,
such as traces, diagnostics, and warnings, plus a message pointing to the saved CSV;
it does not duplicate the summary tables.
Timestamps include microseconds so separate runs preserve previous results.
Help and argument validation errors do not create logs; errors stay in the terminal.

```text
output/
  lesson01/
    TIMESTAMP/
      run.log
      summary.csv
  lesson02/
    TIMESTAMP/
      run.log
      summary.csv
  lesson03/
    TIMESTAMP/
      run.log
      summary.csv
  lesson04/
    TIMESTAMP/
      run.log
      summary.csv
```

Generated results are local and ignored by Git.

## CSV summary tables

CSV files have a header row and preserve numeric precision rather than the rounded
terminal display. Open them in a spreadsheet or read them with Python's `csv` module.

| Lesson | One row per | Columns |
| --- | --- | --- |
| 01 | Policy | `policy`, `episodes`, `reward_sum`, `return`, `steps`, `success_rate`, `truncated` |
| 02 | Method and state | `method`, `state`, `value`, `visits`, `iterations`, `episodes`, `truncated_episodes` |
| 03 | Epsilon | `epsilon`, `mean_reward`, `reward_SE`, `mean_pseudo_regret`, `optimal_fraction` |
| 04 | Algorithm and state | `algorithm`, `state`, `value`, `action`, `rounds`, `sweeps`, `residual` |

Lesson 01 means include truncated episodes. In lesson 02, an unvisited MC state's
`value` is blank; terminal values are zero. Fields that do not apply to a method
are blank (for example, DP has no `visits`). Iteration counts and truncation counts
describe the whole method run and repeat across its state rows. Lesson 04 similarly
repeats algorithm statistics across its state rows; the terminal action is `terminal`.
Traces, warnings, and single-run bandit estimates and counts remain in `run.log`.

## Choose a run directory

Add `--output-dir PATH` to a lesson command to choose where its CSV and log are saved.
The program creates the directory automatically:

```bash
RL lesson01 --output-dir output/lesson01/my-run
RL lesson02 --output-dir output/lesson02/my-run
RL lesson03 --output-dir output/lesson03/my-run
RL lesson04 --output-dir output/lesson04/my-run
```

Use distinct names for different settings. Reusing a directory overwrites `summary.csv` and `run.log`.
