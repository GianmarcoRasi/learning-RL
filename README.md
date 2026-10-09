# Learning Reinforcement Learning

Lessons and hands-on Python exercises covering reinforcement learning foundations,
value functions and the Bellman equation, multi-armed bandits, and dynamic programming.

## Requirements

- Python 3.10 or newer, with `venv` and `pip` available.
- The text-based labs use only the Python standard library.
- **Matplotlib** is installed by default for plotting throughout the course. Lesson 1 currently provides PNG plot commands.

Run setup commands from the project root (`learning-RL`). After installation, lesson commands work from any directory in the activated environment.

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

From the project root, with the virtual environment activated, update `pip`
and install the course:

```bash
pip install --upgrade pip
pip install -e .
```

Installation includes Matplotlib and creates the `RL` command in the virtual
environment's scripts directory. Activating the environment adds that directory
to your terminal's command search path. Editable installation means changes to
the lesson code are available immediately.

Check that the command is available:

```bash
RL --help
```

If `RL` is not recognized, activate the project's virtual environment in the
current terminal and repeat `pip install -e .` from the project root. Repeat
installation after pulling an update that adds or changes project commands.

## Run a lesson

With the environment activated, use `RL lessonNN ...`, replacing `lessonNN`
with `lesson01`, `lesson02`, `lesson03`, or `lesson04`:

```bash
RL lesson01 --policy right --trace
RL lesson02 --method dp --right-prob 1
RL lesson03 --runs 1 --steps 20
RL lesson04 --algorithm both
```

Append `--help` to see a lesson's experiment arguments:

```bash
RL lesson01 --help
```

Lesson commands work from any directory in the activated environment. Relative
log and plot paths are resolved from the directory where you run the command.
Plot files are generated only when requested by a lesson command.
Successful runs save results to `summary.csv` and other output to `run.log`, then
print the saved-file paths. Add `--verbose` to also see results and diagnostics on screen, for example
`RL lesson01 --policy right --trace --verbose`. See [output conventions](docs/command-outputs.md).

## Shared vocabulary

The [project glossary](docs/glossary.md) collects the defined terms and notation
from all four lessons and maps them to code and command output. Use these meanings
consistently in lesson explanations, code names, docstrings, help, and plots.
When a lesson introduces a term, add its definition and lesson reference to the glossary.

## Project layout

- `src/learning_rl/lessons/`: runnable labs for Lessons 1–4.
- `src/learning_rl/environments/`: the shared LineWorld interaction and transition model.
- `src/learning_rl/plotting/`: plotting functions; currently Lesson 1.
- `lessons/`: theory, theoretical exercises, and their solutions.
- `lessons-hands-on/`: command references, usage tutorials, practical exercises, and their solutions.
- `docs/`: command-output documentation and study flashcards in `docs/flashcards/`.
- `tests/`: numeric, CLI, and plotting checks.
- `output/`: automatically saved command logs, grouped by lesson and run, plus optional plots. See [output conventions](docs/command-outputs.md). Generated results are local and ignored by Git.

Start with the [Lesson 1 theory](lessons/RL-lesson-01-rl-foundations-01.md)
and its [hands-on tutorial](lessons-hands-on/RL-lesson-01-rl-foundations-02-hands-on.md).
The tutorials link to source files and show commands. Hands-on exercises use the
supplied code and ask for specific outputs and interpretations; they never require
source edits or new programs. Solutions remain in those tutorials. Run lesson examples from any directory after installing the course and activating its environment.

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
