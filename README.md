# trading-research-agent

An open-source **deep research agent for US equities**. Give it a ticker or a
question; it plans the research, pulls data from SEC filings, market data and
the web, and writes a structured report where **every claim links back to its
source**.

> ⚠️ **Not investment advice.** This project produces research summaries, not
> recommendations. It never outputs ratings or price targets. Data may be
> delayed or incomplete. Verify everything before acting on it.

---

## Status

🚧 Early development. See [`docs/design.md`](docs/design.md) for the full
technical plan and milestones.

| Milestone | What it delivers | Status |
|---|---|---|
| M0 | Repo layout, CI, report contract, static report viewer | 🔨 in progress |
| M1 | Tool layer (SEC EDGAR, market data, caching, rate limiting) | ⬜ |
| M2 | First end-to-end report (single agent) | ⬜ |
| M3 | LangGraph plan → parallel research → synthesize | ⬜ |
| M4 | RAG over 10-K/10-Q + citation checking | ⬜ |
| M5 | Eval suite + tuning | ⬜ |
| M6 | HITL, streaming API, full web UI, Docker | ⬜ |

## Repository layout

```
apps/
  api/                  Python — agent, tools, API          (uv, LangGraph)
  web/                  Next.js — report viewer             (pnpm, Tailwind)
packages/
  contracts/            generated report contract, shared by both
docs/
  design.md             technical plan
  decisions/            architecture decision records
Makefile                single entry point for every command
```

The report contract flows one way, and it is generated at every step — so the
frontend types can never drift from the backend schema:

```
apps/api/src/tra/report/schema.py   (Pydantic — source of truth)
        ↓  make schema
packages/contracts/report.schema.json + samples/
        ↓  pnpm run types   (automatic on dev/build)
apps/web/src/types/report.ts
```

CI runs `make schema` and fails on any diff.

## Quick start

Requires Python 3.12+ with [uv](https://docs.astral.sh/uv/), and Node 22+ with
[pnpm](https://pnpm.io/).

```bash
git clone https://github.com/ihaichao/trading-research-agent.git
cd trading-research-agent

cp apps/api/.env.example apps/api/.env   # fill in TRA_LLM_API_KEY, TRA_SEC_USER_AGENT
make setup                               # install backend + frontend deps
make check                               # everything CI runs

make web-dev                             # report viewer at localhost:3000
cd apps/api && uv run python examples/react_from_scratch.py "What is NVDA trading at?"
```

## Commands

Run everything from the repository root.

```bash
make help        # list targets

make setup       # api-setup + web-setup
make check       # api-check + web-check (same as CI)
make schema      # regenerate packages/contracts from the Pydantic schema

make api-fmt     # ruff format + autofix
make api-check   # ruff + mypy + pytest
make api-test    # pytest, offline only

make web-dev     # next dev
make web-check   # eslint + tsc + next build
```

Opt-in tests that hit real APIs: `cd apps/api && uv run pytest -m network`.

## Data sources

| Source | Used for | Notes |
|---|---|---|
| [SEC EDGAR](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) | Fundamentals, 10-K/10-Q text | Free and official. Requires a contact User-Agent; capped at 10 req/s |
| [yfinance](https://github.com/ranaroussi/yfinance) | Quotes, price history | Unofficial and occasionally flaky — always cached, with a fallback provider |
| [Tavily](https://tavily.com) | Web & news search | Used from M4 |

## License

MIT — see [LICENSE](LICENSE).
