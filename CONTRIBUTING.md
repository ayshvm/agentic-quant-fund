# Contributing

Create a focused branch, add tests for behavior changes, and run:

```bash
uvx poetry install
uvx poetry run pytest hedge_fund -q
```

Keep model opinions separate from deterministic execution and risk controls.
Never commit API keys, local mandate data, caches, or backtest receipts.
