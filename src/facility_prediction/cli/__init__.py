"""The pipeline's single entry point.

Every stage is a subcommand here, and nothing else in the package owns
an argument parser, a logging setup, or a database connection's
lifetime. The modules take data in and return data out; this package is
where a run acquires its configuration, its engine, and its output
paths, and where it releases them again.

That separation is what keeps the stages testable: a test calls
``build_samples`` with a frame, not a process with a command line.
"""

from __future__ import annotations

from facility_prediction.cli.main import main
from facility_prediction.cli.parsers import build_parser

__all__ = ["build_parser", "main"]
