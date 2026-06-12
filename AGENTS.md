# AGENTS.md

## Cursor Cloud specific instructions

### Repository layout
- The actual project lives in `agent-platform/` (the repo root only holds this file and a stub `README.md`). Run all commands from `agent-platform/`.

### Project state (important)
- This repo is at **Phase 1 / 5**: it contains only architecture docs, ADRs, and domain **interface/contract definitions** (Python `Protocol`/ABC stubs + Pydantic models). There is **no runnable application** yet — no FastAPI app, no `main.py`, no `docker-compose.yml`, no Alembic migrations. So PostgreSQL/Redis/LLM providers are declared as dependencies but are **not needed** to install or run the current checks.

### Environment
- Python managed via `uv` (installed at `~/.local/bin/uv`; the update script keeps the `agent-platform/.venv` synced). Activate with `source agent-platform/.venv/bin/activate`.
- The startup update script already runs `uv pip install -e ".[dev]"`; you normally don't need to reinstall.

### Checks / commands (run from `agent-platform/`, venv activated)
- Lint: `ruff check src tests`
- Type check (strict): `mypy`
- Layered import contracts: `lint-imports`
- Tests: `pytest`

### Known caveat: `pytest` fails by design right now
- `pyproject.toml` sets `--cov-fail-under=80`, but `tests/` has no tests and `src/` has no implementation yet, so `pytest` exits non-zero with `Total coverage: 0.00%`. This is expected at Phase 1, not an environment problem. To verify test collection without the coverage gate, run `pytest -p no:cov` (or `pytest --no-cov`).

### Verifying the package end-to-end
- There's no server to start; "running" the platform today means importing `src.*` and exercising the domain models (e.g. build `src.agents.events.AgentEvent` via the Pydantic discriminated union, call `AgentPlatformError.to_problem()` for RFC 9457 output, aggregate `TokenUsage`).
