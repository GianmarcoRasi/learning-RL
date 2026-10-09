"""Save lesson output to files, with optional terminal display."""

import csv
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path


class _Tee:
    def __init__(self, terminal, log, verbose):
        self.terminal = terminal
        self.log = log
        self.verbose = verbose

    def write(self, text):
        if self.verbose:
            self.terminal.write(text)
        return self.log.write(text)

    def flush(self):
        if self.verbose:
            self.terminal.flush()
        self.log.flush()


def add_output_argument(parser):
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also show saved output and summary tables in the terminal.",
    )
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
    """Display summary tables only in verbose mode; keep them out of the log."""
    if isinstance(sys.stdout, _Tee):
        if not sys.stdout.verbose:
            with open(os.devnull, "w") as sink, redirect_stdout(sink):
                yield
            return
        stream = sys.stdout.terminal
    else:
        stream = sys.stdout
    with redirect_stdout(stream):
        yield


@contextmanager
def lesson_output(lesson, directory=None, *, verbose=False):
    if directory is None:
        directory = Path("output") / lesson / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log, verbose)):
            yield directory
    if verbose:
        print(f"Output saved to {directory / 'run.log'}", file=sys.stderr)
