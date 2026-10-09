"""Save lesson output while keeping it visible in the terminal."""

import csv
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path


class _Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def add_output_argument(parser):
    parser.add_argument(
        "--output-dir", type=Path,
        help="Save summary.csv and run.log here (default: output/lessonNN/TIMESTAMP/). "
             "Reusing a directory overwrites these files.",
    )


def save_summary(directory, fieldnames, rows):
    """Write numeric results directly, without parsing formatted terminal output."""
    path = directory / "summary.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary saved to {path}")


@contextmanager
def terminal_output():
    """Display a summary table without duplicating it in the run log."""
    stream = sys.stdout.terminal if isinstance(sys.stdout, _Tee) else sys.stdout
    with redirect_stdout(stream):
        yield


@contextmanager
def lesson_output(lesson, directory=None):
    if directory is None:
        directory = Path("output") / lesson / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)):
            yield directory
    print(f"Output saved to {directory / 'run.log'}", file=sys.stderr)
