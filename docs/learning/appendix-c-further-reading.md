# Appendix C – Further reading

Official documentation for the tools that the repository uses today. Entries for later tools are added when their chapters are written. A web address is given only for the main page of each project; use the search of each site to find the topics named here.

## Python and packaging

- **Python Packaging User Guide** – `pyproject.toml`, the src layout, editable installs. <https://packaging.python.org>
- **Python documentation** – the modules `venv`, `logging`, `pathlib` and `importlib`. <https://docs.python.org>
- **Hatch documentation** – the hatchling build backend. <https://hatch.pypa.io>
- **uv documentation** – the command `uv pip compile`. <https://docs.astral.sh/uv/>

## Configuration and data

- **Pydantic documentation** – models, validators, and the section on settings management. <https://docs.pydantic.dev>
- **python-dotenv documentation** – the format of `.env` files.
- **Pandera documentation** – schemas, checks, strict mode. <https://pandera.readthedocs.io>
- **pandas documentation** – time series, time zones (`tz_localize` and `tz_convert`, with their options for times that do not exist or exist twice), and `interpolate`. <https://pandas.pydata.org>

## Getting data

- **HTTPX documentation** – clients, timeouts, streaming responses, and mock transports for tests. <https://www.python-httpx.org>
- **Apache Arrow documentation** – the Python library pyarrow and the Parquet format. <https://arrow.apache.org>
- **DuckDB documentation** – reading Parquet files with SQL. <https://duckdb.org>
- **Open-Meteo documentation** – the historical forecast API and the historical weather (archive) API. <https://open-meteo.com>
- **NYC Taxi and Limousine Commission, TLC Trip Record Data** – the page that lists the monthly files and the data dictionary of the columns. Read the dictionary before you write a cleaning rule: it defines `fare_amount`, `total_amount` and the codes of `payment_type`.
- **IANA Time Zone Database** – the source of names such as `America/New_York` and of the dates on which clocks change. <https://www.iana.org/time-zones>

## Command line, tests and code quality

- **Typer documentation** – commands, arguments, testing a CLI. <https://typer.tiangolo.com>
- **pytest documentation** – fixtures, `tmp_path`, `monkeypatch`, parametrised tests. <https://docs.pytest.org>
- **pytest-cov and Coverage.py documentation** – measuring coverage, branch coverage, `fail_under`.
- **Ruff documentation** – the list of rules and the formatter. <https://docs.astral.sh/ruff/>
- **mypy documentation** – type hints and strict mode. <https://mypy.readthedocs.io>
- **pre-commit documentation** – hooks and local hooks. <https://pre-commit.com>

## Git

- **Git documentation** – the pages on `gitattributes` and `gitignore`. <https://git-scm.com>
- **Conventional Commits** – the specification of the message format. <https://www.conventionalcommits.org>

---

[Previous: Appendix B – Glossary](appendix-b-glossary.md) | [Contents](README.md)
