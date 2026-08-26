import "server-only";

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import type { Report } from "@/types/report";

/**
 * 只在服务端使用。目前直接读 public/samples 下的静态样例
 * （由 predev/prebuild 从 docs/samples 同步）。
 * M6 接入后端后，这里换成 fetch(`${API}/reports/${id}`)，页面组件不用改。
 */
export async function loadReport(name = "sample_report.json"): Promise<Report> {
  const path = join(process.cwd(), "public", "samples", name);
  const raw = await readFile(path, "utf-8");
  return JSON.parse(raw) as Report;
}
