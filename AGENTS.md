# Repository Guidelines

## Project Structure & Module Organization

`researchclaw/` contains the Python package and CLI entry point (`researchclaw.cli:main`). Major subsystems live under focused packages such as `pipeline/`, `experiment/`, `web/`, `llm/`, `hitl/`, `skills/`, `server/`, and `metaclaw_bridge/`. Tests are in `tests/`, with focused integration scripts in `scripts/`. Documentation and translations are in `docs/`; static site files are in `website/`; shared images are in `image/` and `website/assets/`. `frontend-legacy/` is a vanilla JS dashboard with no declared build step. `experiments/arc_bench/` and `external/agents/` hold benchmark and external-agent integrations.

## Build, Test, and Development Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Create a local editable install with pytest and async test support.

```bash
researchclaw init
researchclaw doctor
researchclaw run --config config.arc.yaml --topic "Your research idea" --auto-approve
```

Initialize local config, validate the environment, and run the pipeline. For tests, use:

```bash
pytest tests/
pytest tests/test_rc_cli.py -q
```

## Coding Style & Naming Conventions

Target Python 3.11+. Use 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep modules aligned with existing subsystem boundaries instead of adding broad utility files. Prefer typed, explicit interfaces for config and pipeline contracts. No formatter or linter is configured in `pyproject.toml`; match nearby style and keep imports tidy.

## Testing Guidelines

Use pytest. Add tests under `tests/` as `test_*.py`, and name test functions `test_<behavior>`. Prefer focused unit coverage for new helpers and regression tests for bug fixes. Run `pytest tests/` before opening a PR; run targeted files during development. For LLM, web, sandbox, or Docker paths, keep network/secrets assumptions isolated and document any required local setup.

## Commit & Pull Request Guidelines

Recent history uses short conventional prefixes such as `fix:`, `docs:`, `test:`, `release:`, and scoped forms like `fix(search_strategy): ...`. Keep commits small and imperative. PRs should branch from `main`, address one concern, describe behavior changes, link issues when relevant, include screenshots for UI/site changes, and state the tests run.

## Security & Configuration Tips

Do not commit secrets. Use `config.researchclaw.example.yaml` as the tracked template; keep local settings in `config.arc.yaml` or `config.yaml`, both gitignored. Review sandbox, SSH, web crawling, and LLM provider changes carefully because they affect external execution or network access.
