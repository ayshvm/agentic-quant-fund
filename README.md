# Agentic Quant Fund

An auditable research lab for composing AI and quantitative alpha models,
turning their signals into risk-controlled portfolios, and backtesting the
result from the terminal.

> Educational and research software only. It is not investment advice and is
> not intended for live trading.

## What is different

This derivative adds configurable research workspaces, strict ticker input
validation, net- and short-exposure controls, realistic commission/slippage
modeling, richer trade diagnostics, institutional risk-adjusted metrics,
mandate validation, date-window safety checks, and automated CI.

The engine keeps a clear boundary: agents form views, deterministic portfolio
and risk code decides exposure, and every cycle produces an auditable record.

## Install

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) or Poetry.

```bash
uvx poetry install
uvx poetry run aqf
```

The legacy `aihf` command remains available for compatibility.

## Configure

On first use, credentials and cached data are stored under `~/.hedge-fund`.
Set `AGENTIC_QUANT_HOME` to isolate the workspace:

```bash
export AGENTIC_QUANT_HOME="$PWD/.aqf-data"
```

You need a Financial Datasets API key and one supported LLM provider key for
LLM-powered models. Quant-only strategies do not require an LLM key.

## Run a mandate

```bash
uvx poetry run aqf hedge_fund/fund/example.yaml --validate
uvx poetry run aqf --version
uvx poetry run aqf hedge_fund/fund/example.yaml --tickers AAPL,MSFT
uvx poetry run aqf hedge_fund/fund/example.yaml --tickers AAPL,MSFT --backtest \
  --start 2024-01-01 --date 2025-01-01 --out result.json
```

Use `commission_bps` and `slippage_bps` with `BacktestEngine` when evaluating
an individual alpha model. Mandates may optionally set `max_net_exposure` and
`max_short_exposure` in addition to position and gross limits.

## Development

```bash
git clone https://github.com/ayshvm/agentic-quant-fund.git
cd agentic-quant-fund
uvx poetry install
uvx poetry run pytest hedge_fund -q
```

See [NOTICE.md](NOTICE.md) for upstream attribution. The original MIT license
and copyright notice are preserved in [LICENSE](LICENSE).
