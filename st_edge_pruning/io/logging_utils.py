from __future__ import annotations

import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    """Write console output to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name: str):
        """Forward terminal attributes such as isatty() and encoding."""

        return getattr(self.terminal, name)


@contextmanager
def tee_output(log_path: str | Path) -> Iterator[None]:
    """Mirror stdout and stderr into one run-local log file."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        stdout = TeeStream(sys.stdout, log_file)
        stderr = TeeStream(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
