# contracts

前后端共享的**报告契约产物**。这里的文件全部是**生成的，不要手写**。

| 文件 | 生成自 | 消费方 |
|---|---|---|
| `report.schema.json` | `apps/api/src/tra/report/schema.py` | `apps/web` 生成 TypeScript 类型 |
| `samples/sample_report.json` | `apps/api/examples/sample_report.py` | `apps/web` 的开发数据源；后端测试的回归基准 |

改过后端的 Pydantic schema 之后，在仓库根目录跑：

```bash
make schema
```

CI 会执行 `make schema` 再 `git diff --exit-code packages/contracts`，
忘了跑就会红。这是有意设的护栏：**前端类型不可能和后端契约漂移**。

之所以放在 `packages/` 而不是 `docs/`：它不是文档，是两个 app 都依赖的构建输入。
