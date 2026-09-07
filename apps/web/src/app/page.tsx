import ReportView from "@/components/ReportView";
import { loadReport } from "@/lib/load-report";

export default async function Home() {
  const report = await loadReport("NVDA.json");
  return <ReportView report={report} />;
}
