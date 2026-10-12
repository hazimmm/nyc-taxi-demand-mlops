from __future__ import annotations

import json
import logging

import pytest

from demand.logs import JsonFormatter, configure_logging, get_logger


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("demand.test", logging.INFO, __file__, 1, "rows=%d", (3,), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_emits_one_json_object_with_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(_record(stage="silver")))

    assert payload["message"] == "rows=3"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "demand.test"
    assert payload["stage"] == "silver"
    assert "time" in payload


def test_formatter_includes_exception_text() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exception"]


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers, root.level
    try:
        configure_logging()
        configure_logging()
        get_logger("demand.test").info("hello")

        assert len(root.handlers) == 1
        assert json.loads(capsys.readouterr().err.strip())["message"] == "hello"
    finally:
        root.handlers, root.level = previous_handlers, previous_level
