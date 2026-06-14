# Tests

This folder contains smoke tests for the time-distillation pipeline.

## Quick Run
From the project root:

```bash
python -m pytest tests
```

Or run individual tests:

```bash
python -m pytest tests/time_distill_data_smoke.py
python -m pytest tests/time_distill_model_smoke.py
```

## Notes
- Tests require valid dataset paths from config or fixtures.
- If datasets are missing, smoke tests will fail at the data-loading step.
