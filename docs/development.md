# Development Notes

## Project Status

This codebase is already useful for experimentation, but it is not packaged yet
like a polished library release.

Current state:

- no `pyproject.toml` or `setup.py`
- a lightweight `requirements.txt` exists for runtime setup
- test files are a mix of smoke tests and exploratory scripts
- public examples are concentrated in `Runs.py` and `tests/simulator_test.py`

## Recommended Validation Commands

These commands were checked in this workspace:

```bash
python StochasticPBBP/Runs.py --iterations 1 --print-every 1
python StochasticPBBP/Runs.py --iterations 1 --horizon 113 --batch-size 5 --print-every 1
python StochasticPBBP/tests/simulator_test.py
```

If Matplotlib emits cache-permission warnings during imports, use:

```bash
env MPLCONFIGDIR=/tmp/mpl python StochasticPBBP/tests/simulator_test.py
```

## Notes On The Existing Test Scripts

- `tests/simulator_test.py` is the most reliable smoke test in the repository
  right now
- `tests/compiler_test.py` compares Torch and JAX behavior and therefore needs
  the optional JAX stack
- `tests/rollout_test.py` currently has stale path/import assumptions and should
  be treated as exploratory code until cleaned up

## Suggested Next Improvements

If you want to turn this into a cleaner package, the next engineering steps are:

1. add packaging metadata and a real dependency manifest
2. promote the smoke tests into an automated test runner
3. add focused tests for chunked horizon training and state carry-over
4. add a stable top-level package API with `__init__.py` exports
5. separate research experiments from public examples

## Documentation Tooling

This repository now includes:

- `mkdocs.yml`
- `.readthedocs.yaml`
- Markdown pages under `docs/`

Build locally with:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```
