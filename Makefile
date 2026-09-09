# 仓库根 Makefile：所有命令都从这里跑，不需要 cd 进子目录。
#   apps/api  —— Python / LangGraph 后端
#   apps/web  —— Next.js 前端
#   packages/contracts —— 两边共享的报告契约产物（由 apps/api 生成）

API := apps/api
WEB := apps/web

.PHONY: help setup check clean eval eval-judge \
        api-setup api-fmt api-lint api-type api-test api-check \
        web-setup web-dev web-lint web-type web-build web-check \
        schema

help:
	@echo "setup       安装前后端依赖"
	@echo "check       前后端全部检查（等同 CI）"
	@echo "schema      从 Pydantic schema 重新生成契约产物"
	@echo "eval        跑评测集（确定性指标，免费）"
	@echo "eval-judge  跑评测集 + LLM 评判（花钱）"
	@echo "api-*       后端：fmt lint type test check"
	@echo "web-*       前端：dev lint type build check"

# ---------------------------------------------------------------- 全局
setup: api-setup web-setup

check: api-check web-check

clean:
	rm -rf $(API)/.pytest_cache $(API)/.mypy_cache $(API)/.ruff_cache $(API)/.coverage
	rm -rf $(WEB)/.next $(WEB)/node_modules/.cache
	find $(API) -name __pycache__ -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- 后端
api-setup:
	cd $(API) && uv sync --extra dev

api-fmt:
	cd $(API) && uv run ruff format . && uv run ruff check --fix .

api-lint:
	cd $(API) && uv run ruff format --check . && uv run ruff check .

api-type:
	cd $(API) && uv run mypy

api-test:
	cd $(API) && uv run pytest -m "not network"

api-check: api-lint api-type api-test

# ---------------------------------------------------------------- 评测
# 确定性指标：不调模型、免费、可以进 CI
eval:
	cd $(API) && uv run python examples/eval_demo.py

# 加上 LLM 评判：花钱、慢，改动前后各跑一次做对比
eval-judge:
	cd $(API) && uv run python examples/eval_demo.py --judge

# ---------------------------------------------------------------- 契约
# 改过 apps/api/src/tra/report/schema.py 之后必须跑这个，
# 否则前端类型和样例数据会和后端契约漂移（CI 会拦下来）。
schema:
	cd $(API) && uv run python -c "from tra.report.schema import export_json_schema; print(export_json_schema())" > ../../packages/contracts/report.schema.json
	cd $(API) && uv run python examples/sample_report.py

# ---------------------------------------------------------------- 前端
# 前端的 TS 类型和样例数据由 predev / prebuild 自动从 packages/contracts 生成，
# 不需要单独的目标。
web-setup:
	cd $(WEB) && pnpm install

web-dev:
	cd $(WEB) && pnpm run dev

web-lint:
	cd $(WEB) && pnpm run lint

web-type:
	cd $(WEB) && pnpm run typecheck

web-build:
	cd $(WEB) && pnpm run build

web-check: web-lint web-type web-build
