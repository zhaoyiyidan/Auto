# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Companion docs

Three other files in the repo root cover overlapping ground — read whichever matches your need:
- `README.md` — public-facing overview, news, integration list.
- `RESEARCHCLAW_CLAUDE.md` — deeper map of the package layout and key APIs.
- `RESEARCHCLAW_AGENTS.md` / `AGENTS.md` — orchestration guidance and contributor conventions.

## Common commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + pytest; use ".[all]" for web/pdf/scholarly extras

# Bootstrap local config (writes config.arc.yaml — gitignored)
researchclaw init
researchclaw doctor                # check env, deps, LLM connectivity

# Run the pipeline
researchclaw run --config config.arc.yaml --topic "your idea" --auto-approve
researchclaw run --profile hep_ph --resume                         # resume last checkpoint
researchclaw run --from-stage PAPER_OUTLINE --output artifacts/foo  # restart mid-pipeline
python -m researchclaw run ...                                     # equivalent to the CLI

# Tests
pytest tests/                               # full suite (~2700 tests)
pytest tests/test_rc_cli.py -q              # one file
pytest tests/test_rc_cli.py::test_name -q   # one test
python tests/e2e_real_llm.py                # E2E, needs API key in config

# Other useful subcommands
researchclaw validate --config config.arc.yaml
researchclaw setup                          # install optional tools (OpenCode, etc.)
researchclaw skills list|install|validate
researchclaw profile list|create|...        # named pipeline profiles (e.g. hep_ph)
researchclaw report --run-dir artifacts/<run-id>
researchclaw serve | dashboard              # web UI / monitor
researchclaw attach|status|approve|reject|guide   # HITL co-pilot CLI
```

The `researchclaw` console script is registered in `pyproject.toml` as `researchclaw.cli:main`. `python -m researchclaw` works identically.

## Architecture (big picture)

ResearchClaw is a **23-stage finite-state-machine pipeline** that turns a research topic into a paper. Everything else hangs off that spine.

### The spine — `researchclaw/pipeline/`

- `stages.py` — `Stage` IntEnum (1..23), `StageStatus`, `STAGE_SEQUENCE`, `NONCRITICAL_STAGES`, gate rollback rules (`DECISION_ROLLBACK`, `MAX_DECISION_PIVOTS`). Stages 5 (LITERATURE_SCREEN), 9 (EXPERIMENT_DESIGN), and 20 (QUALITY_GATE) are gates.
- `contracts.py` — per-stage `StageContract` declaring `required_keys`, `produced_keys`, `gate` flag. Used to validate state passed between stages.
- `executor.py` — `execute_stage()` plus a dispatch table (`_STAGE_EXECUTORS`) wiring each `Stage` to an implementation in `stage_impls/`.
- `stage_impls/` — actual stage logic, grouped by phase: `_topic.py`, `_literature.py`, `_synthesis.py`, `_experiment_design.py`, `_code_generation.py`, `_execution.py`, `_analysis.py`, `_paper_writing.py`, `_review_publish.py`.
- `runner.py` — `execute_pipeline()` and `execute_iterative_pipeline()`. Handles checkpointing (`heartbeat.json`, `pipeline.pid`, per-stage state JSONs in the run dir), `--resume`, gate pausing, and rollbacks driven by `RESEARCH_DECISION` (PROCEED/PIVOT/ITERATE).
- `code_agent.py`, `opencode_bridge.py` — delegate Stages 10/13 code generation to external CLI agents (Claude Code, Codex, Copilot, Gemini, Kimi, OpenCode "Beast Mode").
- `experiment_diagnosis.py`, `experiment_repair.py`, `paper_verifier.py`, `verified_registry.py`, `requirements_judge.py` — anti-fabrication / repair loop.

### Configuration — `researchclaw/config.py`

- `RCConfig.load()` / `load_config()` load YAML and validate. `CONFIG_SEARCH_ORDER` controls fallback (`config.arc.yaml` → `config.yaml` → example template).
- Local configs (`config.arc.yaml`, `config.yaml`, `config_*.yaml`) are gitignored. Only `config.researchclaw.example.yaml` is tracked.
- Profiles (`researchclaw profile create ...`) are named overlays stored under the project root and selected via `--profile`.

### Subsystem map (under `researchclaw/`)

- `experiment/` — code execution. `sandbox.py` (local subprocess), `runner.py`, `git_manager.py`, `validator.py` (AST + security scan + import check + auto-repair, retries up to 3×), `visualize.py` (matplotlib charts).
- `domains/` — domain detection and routing for Stages 10–13. `detector.py` picks a domain (HEP, biology, statistics, ML default, etc.); `adapters/` and `profiles/` configure executors per domain. The HEP path drives `external/agents/ColliderAgent/` (Magnus cloud); biology uses COBRApy; statistics uses a simulation-study agent.
- `hitl/` — Human-in-the-Loop co-pilot (v0.4+). 6 intervention modes, branching, claim verification, cost guards, dynamic SmartPause, ALHF learning. `attach`/`status`/`approve`/`reject`/`guide` CLI commands plug into a running pipeline.
- `evolution.py`, `evolution_aevolve.py` — extract lessons from runs; A-Evolve agentic evolution skill.
- `metaclaw_bridge/` — converts pipeline failures into reusable skills injected into all 23 stages (cross-run learning).
- `skills/` + `.claude/skills/` — pluggable SKILL.md packs (loaded via `researchclaw skills install`). Pre-loaded set includes `researchclaw`, `a-evolve`, `scientific-writing`, `chemistry-rdkit`, `biology-biopython`, etc.
- `llm/` — `LLMClient` (OpenAI-compatible), `from_rc_config()` factory; supports retry/backoff and multiple providers.
- `web/`, `literature/` — paper search & retrieval (arxiv + optional scholarly/tavily/crawl4ai).
- `knowledge/`, `memory/` — markdown knowledge-base writes, episodic memory.
- `server/`, `dashboard/`, `frontend-legacy/` — web UI / monitor.
- `agents/` (`benchmark_agent`, `code_searcher`, `figure_agent`) — narrow internal helper agents.
- `mcp/`, `servers/` — MCP integration.
- `overleaf/` — bidirectional sync.

### Run artifacts

Pipeline runs write into `artifacts/<run-id>/` (gitignored): per-stage JSON state, `pipeline_summary.json`, `heartbeat.json`, `pipeline.pid`, generated code, charts, paper drafts, citation reports. `sentinel.sh <run_dir>` is a watchdog that auto-resumes the run if the heartbeat goes stale.

### Benchmarks

- `experiments/arc_bench/` — ARC-Bench, 55-topic benchmark (ML/HEP/quantum/biology/stats). Manifest + rubric per topic; harness scripts under `experiments/arc_bench/scripts/`. Most other `experiments/*/` directories are gitignored research scratch.

## Conventions

- Python 3.11+, 4-space indent. `snake_case` functions/modules, `PascalCase` classes, `UPPER_SNAKE_CASE` constants. No formatter or linter is configured — match the surrounding file.
- Stage logic belongs in `pipeline/stage_impls/`, not in `executor.py` itself; `executor.py` only dispatches.
- New stages must register a `StageContract` in `contracts.py` and an executor in `executor.py`'s dispatch table; the FSM in `stages.py` is the single source of truth for sequencing.
- Commits use conventional prefixes (`fix:`, `docs:`, `test:`, `release:`, scoped forms like `fix(search_strategy): ...`). Branch from `main`, one concern per PR.
- For LLM/web/sandbox/Docker code paths, isolate network and secret assumptions; the test suite must stay runnable without external services.
- `config.researchclaw.example.yaml` is the only tracked config — never commit secrets. Local env loaders go in `scripts/setenv*.sh` (gitignored).
