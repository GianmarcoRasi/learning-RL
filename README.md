# Learning Reinforcement Learning

Lessons and hands-on Python exercises covering reinforcement learning foundations,
value functions and the Bellman equation, multi-armed bandits, and dynamic programming.

## Requirements

- Python 3.10 or newer, with `venv` and `pip` available.
- The text-based labs use only the Python standard library.
- **Matplotlib** is the only third-party package needed for the optional plots in Lesson 1.

Run the setup and example commands below from the project root (`learning-RL`).

## Create a local virtual environment

A virtual environment keeps this project's packages separate from other Python projects.
The following commands create it in `.venv` inside the project directory.

### Windows (PowerShell)

Check your installed Python version, create the environment, and activate it:

```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Ensure the reported version is at least 3.10. If you have multiple Python versions,
you can select one explicitly, for example `py -3.12 -m venv .venv`.
If `py` is unavailable but `python --version` works, use `python -m venv .venv`.

If PowerShell blocks the activation script, you can use the environment directly
without activating it. Replace `python` in the commands below with
`.\.venv\Scripts\python.exe`, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\.venv\Scripts\python.exe -m learning_rl.lessons.lesson01 --trace
```

If neither `py` nor `python` works, install Python with its launcher and `pip`,
then reopen your terminal. A `python` command pointing to the Microsoft Store
app alias may require using `py` or your installed Python executable instead.

### macOS / Linux (bash or zsh)

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

Ensure the reported version is at least 3.10. If creating the environment reports
that `venv` or `ensurepip` is unavailable, install the Python virtual environment
package provided by your operating system, then retry.

## Install packages

With the virtual environment activated, update `pip`:

```bash
python -m pip install --upgrade pip
```

Install the course in editable mode. Text runs require no third-party runtime packages:

```bash
python -m pip install -e .
```

To also generate Lesson 1 plots, install the optional plotting dependencies:

```bash
python -m pip install -e ".[plots]"
```

The `pyproject.toml` declares Python support and optional dependencies. An editable
installation uses the files under `src/`, so source edits take effect on the next run.

## Run Lesson 1

Display available options and run a single episode with a transition trace:

```bash
python -m learning_rl.lessons.lesson01 --help
python -m learning_rl.lessons.lesson01 --policy right --trace
```

With the default settings, the right-moving policy reaches the goal in four steps
with a total reward of `7.000` and a discounted return of `4.580`.

Compare the fixed right-moving and random policies across 1,000 episodes:

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet
```

After installing Matplotlib, save PNG plots in `outputs/lesson01`:

```bash
python -m learning_rl.lessons.lesson01 --compare --episodes 1000 --seed 7 --quiet --plots outputs/lesson01
```

The program compares fixed policies; it does not train an agent yet.

## Project layout

- `src/learning_rl/lessons/`: runnable labs for Lessons 1–4.
- `src/learning_rl/environments/`: the shared LineWorld interaction and transition model.
- `src/learning_rl/plotting/`: optional plotting functions; currently Lesson 1.
- `docs/lessons/`: theory, usage tutorials, exercises, and their solutions.
- `docs/flashcards/`: study flashcards.
- `tests/`: numeric, CLI, and optional plotting checks.
- `outputs/`: generated experiment results (ignored by Git).
- `scratch/`: local copies of standalone solution programs (ignored by Git).

Start with the [Lesson 1 theory](docs/lessons/RL-lesson-01-rl-foundations-01.md)
and its [hands-on tutorial](docs/lessons/RL-lesson-01-rl-foundations-02-hands-on.md).
The tutorials link to source files and show commands; exercise solutions remain in
those tutorials. Run all examples from the repository root after installing the course.

## Run the other lessons

```bash
python -m learning_rl.lessons.lesson02 --method compare --episodes 10000 --seed 7
python -m learning_rl.lessons.lesson03 --epsilons 0 0.1 1 --steps 2000 --runs 200
python -m learning_rl.lessons.lesson04 --algorithm both
```

Each module supports `--help`. Lessons 2–4 provide text results without requiring
Matplotlib. Lesson 1 supports both tables and optional saved plots.

## Check the code

```bash
python -m unittest discover -s tests -v
```

Plotting checks run when Matplotlib is installed and are skipped otherwise.

## Finish or resume a session

Leave the activated environment with:

```bash
deactivate
```

In a new terminal, return to the project root and run the activation command for
your operating system again. You only need to create `.venv` and install its
packages once, unless you recreate the environment. Keep `.venv/`, `__pycache__/`,
and generated plots out of version control.
