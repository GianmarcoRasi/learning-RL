# Learning Reinforcement Learning

Lessons and hands-on Python exercises covering reinforcement learning foundations,
value functions and the Bellman equation, multi-armed bandits, and dynamic programming.

## Requirements

- Python 3.10 or newer, with `venv` and `pip` available.
- The text-based labs use only the Python standard library.
- **Matplotlib** is installed by default for plotting throughout the course. Lesson 1 currently provides PNG plot commands.

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

If PowerShell blocks the activation script because running scripts is disabled,
allow local scripts for the current terminal session, then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

This policy change expires when you close the terminal. In a new PowerShell
terminal, repeat these two commands if activation is blocked again. No administrator
privileges are required. Organization-managed policies may override this setting.

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

Install the course in editable mode, including Matplotlib:

```bash
python -m pip install -e .
```

The `pyproject.toml` declares Python support and default dependencies. This setup
prepares the environment for plotting in current and subsequent lessons. Plot
files are generated only when requested by a lesson command.

## Project layout

- `src/learning_rl/lessons/`: runnable labs for Lessons 1–4.
- `src/learning_rl/environments/`: the shared LineWorld interaction and transition model.
- `src/learning_rl/plotting/`: plotting functions; currently Lesson 1.
- `lessons/`: theory, usage tutorials, exercises, and their solutions.
- `docs/`: command-output documentation and study flashcards in `docs/flashcards/`.
- `tests/`: numeric, CLI, and plotting checks.
- `output/`: automatically saved command logs, grouped by lesson and run, plus optional plots. See [output conventions](docs/command-outputs.md). Generated results are local and ignored by Git.

Start with the [Lesson 1 theory](lessons/RL-lesson-01-rl-foundations-01.md)
and its [hands-on tutorial](lessons/RL-lesson-01-rl-foundations-02-hands-on.md).
The tutorials link to source files and show commands. Hands-on exercises use the
supplied code and ask for specific outputs and interpretations; they never require
source edits or new programs. Solutions remain in those tutorials. Run all examples from the repository root after installing the course.

## Check the code

```bash
python -m unittest discover -s tests -v
```

Plotting checks run with Matplotlib, included in the standard installation.

## Finish or resume a session

Leave the activated environment with:

```bash
deactivate
```

In a new terminal, return to the project root and run the activation command for
your operating system again. You only need to create `.venv` and install its
packages once, unless you recreate the environment. Keep `.venv/`, `__pycache__/`,
and generated plots out of version control.
