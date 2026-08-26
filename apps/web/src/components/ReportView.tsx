"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  SOURCE_KIND_LABEL,
  buildCitationIndex,
  formatMetricValue,
  sourceById,
} from "@/lib/report";
import type { Claim, MetricSeries, Report, Section, Source } from "@/types/report";

function formatDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toISOString().slice(0, 10);
}

function Cite({
  ids,
  index,
  onSelect,
}: {
  ids: string[];
  index: Map<string, number>;
  onSelect: (id: string) => void;
}) {
  return (
    <span className="ml-1 inline-flex gap-1 align-super">
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          onClick={() => onSelect(id)}
          className="cursor-pointer rounded px-1 text-[11px] leading-4 font-mono text-[var(--accent)] ring-1 ring-[var(--line)] hover:bg-[var(--panel)]"
          aria-label={`View source ${index.get(id) ?? "?"}`}
        >
          {index.get(id) ?? "?"}
        </button>
      ))}
    </span>
  );
}

function ClaimList({
  claims,
  index,
  onSelect,
}: {
  claims: Claim[];
  index: Map<string, number>;
  onSelect: (id: string) => void;
}) {
  if (claims.length === 0) return null;
  return (
    <ul className="space-y-2">
      {claims.map((claim, i) => (
        <li key={i} className="flex gap-2 text-[15px] leading-relaxed">
          <span
            className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--accent)]"
            aria-hidden
          />
          <span>
            {claim.text}
            <Cite ids={claim.source_ids} index={index} onSelect={onSelect} />
            {claim.confidence === "low" && (
              <span className="ml-2 rounded px-1 text-[11px] text-[var(--muted)] ring-1 ring-[var(--line)]">
                low confidence
              </span>
            )}
          </span>
        </li>
      ))}
    </ul>
  );
}

function MetricTable({
  series,
  index,
  onSelect,
}: {
  series: MetricSeries;
  index: Map<string, number>;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse text-sm">
        <caption className="pb-2 text-left text-[13px] text-[var(--muted)]">
          {series.label}
          <Cite ids={series.source_ids} index={index} onSelect={onSelect} />
        </caption>
        <thead>
          <tr className="border-b border-[var(--line)] text-left text-[var(--muted)]">
            {series.points.map((p) => (
              <th key={p.period} className="px-3 py-1.5 font-medium">
                {p.period}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {series.points.map((p) => (
              <td key={p.period} className="px-3 py-1.5 font-mono tabular-nums">
                {formatMetricValue(p.value, series.unit)}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function SectionBlock({
  section,
  index,
  onSelect,
}: {
  section: Section;
  index: Map<string, number>;
  onSelect: (id: string) => void;
}) {
  return (
    <section id={section.kind} className="scroll-mt-20 border-t border-[var(--line)] py-8">
      <h2 className="mb-4 text-lg font-semibold">{section.title}</h2>
      {section.narrative_md && (
        <p className="mb-4 text-[15px] leading-relaxed text-[var(--muted)]">
          {section.narrative_md}
        </p>
      )}
      <ClaimList claims={section.claims ?? []} index={index} onSelect={onSelect} />
      {(section.metrics ?? []).length > 0 && (
        <div className="mt-6 space-y-6">
          {(section.metrics ?? []).map((series) => (
            <MetricTable
              key={series.key}
              series={series}
              index={index}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function SourceDrawer({
  source,
  number,
  onClose,
}: {
  source: Source;
  number: number;
  onClose: () => void;
}) {
  return (
    <aside
      role="dialog"
      aria-label="Source detail"
      className="fixed inset-y-0 right-0 z-20 w-full max-w-md overflow-y-auto border-l border-[var(--line)] bg-[var(--panel)] p-6 shadow-xl"
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="rounded bg-[var(--bg)] px-1.5 py-0.5 font-mono text-xs ring-1 ring-[var(--line)]">
            {number}
          </span>
          <span className="text-xs tracking-wide text-[var(--muted)] uppercase">
            {SOURCE_KIND_LABEL[source.kind]}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="cursor-pointer text-sm text-[var(--muted)] hover:text-[var(--fg)]"
          aria-label="Close"
        >
          Close
        </button>
      </div>

      <h3 className="mb-1 text-[15px] font-semibold">{source.title}</h3>
      {source.locator && (
        <p className="mb-4 font-mono text-xs text-[var(--muted)]">{source.locator}</p>
      )}

      {source.snippet && (
        <blockquote className="mb-4 border-l-2 border-[var(--accent)] pl-3 text-[14px] leading-relaxed">
          {source.snippet}
        </blockquote>
      )}

      <dl className="mb-4 grid grid-cols-2 gap-y-1 text-xs text-[var(--muted)]">
        <dt>Published</dt>
        <dd className="text-right">{formatDate(source.published_at)}</dd>
        <dt>Retrieved</dt>
        <dd className="text-right">{formatDate(source.retrieved_at)}</dd>
      </dl>

      <a
        href={source.url}
        target="_blank"
        rel="noreferrer noopener"
        className="text-sm text-[var(--accent)] underline underline-offset-2"
      >
        Open primary source →
      </a>
    </aside>
  );
}

export default function ReportView({ report }: { report: Report }) {
  const [selected, setSelected] = useState<string | null>(null);
  const index = useMemo(() => buildCitationIndex(report), [report]);
  const selectedSource = selected ? sourceById(report, selected) : undefined;

  const close = useCallback(() => setSelected(null), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  return (
    <div className="mx-auto flex max-w-6xl gap-10 px-6 py-10">
      <nav className="sticky top-10 hidden h-fit w-48 shrink-0 lg:block">
        <p className="mb-3 text-xs tracking-wide text-[var(--muted)] uppercase">
          Contents
        </p>
        <ul className="space-y-1.5 text-sm">
          {report.sections.map((s) => (
            <li key={s.kind}>
              <a
                href={`#${s.kind}`}
                className="text-[var(--muted)] hover:text-[var(--fg)]"
              >
                {s.title}
              </a>
            </li>
          ))}
          <li>
            <a href="#sources" className="text-[var(--muted)] hover:text-[var(--fg)]">
              Sources
            </a>
          </li>
        </ul>
      </nav>

      <main className="min-w-0 flex-1">
        <header className="pb-2">
          <div className="mb-2 flex items-baseline gap-3">
            <h1 className="font-mono text-2xl font-bold">{report.ticker}</h1>
            <span className="text-[var(--muted)]">{report.company_name}</span>
          </div>
          <p className="mb-4 text-[15px] text-[var(--muted)]">{report.question}</p>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--muted)]">
            <span>Generated {formatDate(report.generated_at)}</span>
            <span>Data as of {formatDate(report.data_as_of)}</span>
            {report.meta?.model && <span>Model {report.meta.model}</span>}
            {report.meta?.duration_seconds != null && (
              <span>{Math.round(report.meta.duration_seconds)}s</span>
            )}
            {report.meta?.cost_usd != null && (
              <span>${report.meta.cost_usd.toFixed(2)}</span>
            )}
          </div>
        </header>

        {report.sections.map((section) => (
          <SectionBlock
            key={section.kind}
            section={section}
            index={index}
            onSelect={setSelected}
          />
        ))}

        <section id="sources" className="scroll-mt-20 border-t border-[var(--line)] py-8">
          <h2 className="mb-4 text-lg font-semibold">Sources</h2>
          <ol className="space-y-2 text-sm">
            {(report.sources ?? [])
              .slice()
              .sort((a, b) => (index.get(a.id) ?? 0) - (index.get(b.id) ?? 0))
              .map((source) => (
                <li key={source.id} className="flex gap-2">
                  <span className="font-mono text-xs text-[var(--muted)]">
                    {index.get(source.id)}.
                  </span>
                  <button
                    type="button"
                    onClick={() => setSelected(source.id)}
                    className="cursor-pointer text-left hover:underline"
                  >
                    {source.title}
                    {source.locator && (
                      <span className="text-[var(--muted)]"> — {source.locator}</span>
                    )}
                  </button>
                </li>
              ))}
          </ol>
        </section>

        <footer className="border-t border-[var(--line)] py-6 text-xs leading-relaxed text-[var(--muted)]">
          {report.disclaimer}
        </footer>
      </main>

      {selectedSource && (
        <SourceDrawer
          source={selectedSource}
          number={index.get(selectedSource.id) ?? 0}
          onClose={close}
        />
      )}
    </div>
  );
}
