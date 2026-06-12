# AGENTS.md

## Cursor Cloud specific instructions

### Layout & current stage
- The Python project lives in the `agent-platform/` subdirectory, **not** the repo root. Run all tooling from there.
- The repo is at **Stage 1/5** (see `agent-platform/README.md`): only domain interfaces/contracts exist (`agent-platform/src/**/base.py`, `core/types.py`, `core/exceptions.py`, `agents/events.py`). There is **no runnable application/API server yet** and **no automated tests** (`tests/` only holds `__init__.py`).

### Environment
- Dependencies install into a venv at `agent-platform/.venv` (created by the startup update script). Activate it before running tools: `source agent-platform/.venv/bin/activate`.
- The update script uses standard `python3 -m venv` + `pip install -e ".[dev]"`. `uv` is also available and matches the README workflow, but is not required.

### Lint / typecheck (run from `agent-platform/`, venv active)
- `ruff check src tests`
- `mypy`
- `lint-imports` (validates the layered-dependency contracts)

### Tests
- `pytest` currently **fails by design**: `pyproject.toml` sets `--cov-fail-under=80`, so with zero tests it reports "Required test coverage of 80% not reached" even though collection succeeds. This is an expected Stage-1 state, not an environment problem. To smoke-run collection without the coverage gate: `pytest --no-cov`.

### Running / smoke check
- There is no server to start at this stage. To verify the package imports and the contracts validate/serialize, exercise the domain models directly, e.g. build a `Principal`/`TokenUsage`, call `AgentPlatformError.to_problem()`, and `model_dump_json()` on an event from `src.agents.events`.
