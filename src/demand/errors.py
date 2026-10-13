"""Errors that the command line reports in one line instead of a traceback.

A traceback is for a bug: it shows a programmer where the code went wrong. A missing
input file, an empty zone list or a failed quality gate is not a bug. The person at the
terminal needs to know what happened and what to do next, and the message of the error
says exactly that.

Every error of this kind inherits from ``PipelineError``. ``demand.cli`` catches the
base class, prints the message and exits with code 1. Any other exception is unexpected
and keeps its traceback.
"""

from __future__ import annotations


class PipelineError(Exception):
    """An expected failure of a pipeline stage. The message says what to do next."""


class QualityGateError(PipelineError):
    """A data layer did not pass its quality gate, so the pipeline must stop here."""
