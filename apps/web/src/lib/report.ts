import type { Report, Source } from "@/types/report";

/** 引用角标：给报告里出现的每个 source 分配一个稳定的序号，供 [1] [2] 显示。 */
export function buildCitationIndex(report: Report): Map<string, number> {
  const index = new Map<string, number>();
  let n = 0;
  for (const section of report.sections) {
    const ids = [
      ...(section.claims ?? []).flatMap((c) => c.source_ids),
      ...(section.metrics ?? []).flatMap((m) => m.source_ids),
    ];
    for (const id of ids) {
      if (!index.has(id)) index.set(id, ++n);
    }
  }
  // 未被引用的 source 也给个号，保证参考文献列表完整
  for (const source of report.sources ?? []) {
    if (!index.has(source.id)) index.set(source.id, ++n);
  }
  return index;
}

export function sourceById(report: Report, id: string): Source | undefined {
  return (report.sources ?? []).find((s) => s.id === id);
}

export const SOURCE_KIND_LABEL: Record<Source["kind"], string> = {
  sec_filing: "SEC filing",
  market_data: "Market data",
  web: "Web",
  computed: "Computed",
};

export function formatMetricValue(
  value: number | null | undefined,
  unit: string,
): string {
  if (value === null || value === undefined) return "—";
  switch (unit) {
    case "percent":
      return `${value.toFixed(1)}%`;
    case "USD_millions":
      return `$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}M`;
    case "USD":
      return `$${value.toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
    case "ratio":
      return `${value.toFixed(2)}x`;
    default:
      return value.toLocaleString("en-US");
  }
}
