# Command outputs

Run lesson commands from any directory in the activated project environment after following the
[environment setup](../README.md#create-a-local-virtual-environment) and
[package installation instructions](../README.md#install-packages).

Each lesson command automatically creates a directory and saves its printed
summary tables to `output/lessonNN/TIMESTAMP/summary.csv`. Successful runs print
the saved-file paths by default. Add `--verbose` to also display summary tables and other
output in the terminal. The `run.log` in the same directory records other output,
such as traces, diagnostics, and warnings, plus a message pointing to the saved CSV;
it does not duplicate the summary tables.
Lesson 01 defaults to `~/learning-rl-output/lesson01/TIMESTAMP/` instead of a
launch-directory-relative path. Its CSV, log and requested `plots/` all belong to
that same run directory. Other lessons use the `output/` paths shown below.
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

## Show output in the terminal

Add `--verbose` to any lesson command to display its output as well as saving it:

```bash
RL lesson01 --policy right --trace --verbose
RL lesson02 --method dp --verbose
RL lesson03 --runs 1 --steps 20 --verbose
RL lesson04 --algorithm both --verbose
```

Saved-file paths are displayed in both modes, including the plot directory when
plots are requested. Verbose mode does not change the CSV, log,
plots, or experiment results. In lesson 01, `--quiet` still suppresses individual
episode summaries and an explicit `--trace` still records transitions in `run.log`;
`--verbose` also displays those requested transitions and the final summary table.

## Choose a run directory

Add `--output-dir PATH` to a lesson command to choose where its CSV and log are saved, together with any lesson 01 figures in `plots/`.
The program creates the directory automatically:

```bash
RL lesson01 --output-dir output/lesson01/my-run
RL lesson02 --output-dir output/lesson02/my-run
RL lesson03 --output-dir output/lesson03/my-run
RL lesson04 --output-dir output/lesson04/my-run
```

Use distinct names for different settings. Reusing a directory overwrites `summary.csv` and `run.log`.

## Lesson 01 figures

`RL lesson01 --plot-types all` saves all figure types in the run directory’s
`plots/` subdirectory, alongside `summary.csv` and `run.log`. The default run
directory is `~/learning-rl-output/lesson01/TIMESTAMP/`, independently of the
current directory. Each invocation creates a new timestamped run directory.

Use `--output-dir PATH` to choose the directory for all run artifacts. Explicit
relative paths resolve from the current directory. Reusing a directory overwrites
matching files and leaves unrelated files, including figures from earlier selections,
in place.

Use `--plot-types trajectory rewards discount comparison running-mean` to select
one or more figure types, or `--plot-types all` for all five. Without this option,
no figures are saved. Trajectory, reward and discount figures use the first three
available episodes per policy by default; `--plot-episodes N` or
`--plot-episode-ids I J` changes that selection and requires `--plot-types`.
Comparison and running-mean figures always use all episodes of each policy in the
run. `--compare` includes both policies. These commands do not load or compare
summary CSV files from separate runs.
