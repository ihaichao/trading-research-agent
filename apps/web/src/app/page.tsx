import ReportView from "@/components/ReportView";
import { loadReport } from "@/lib/load-report";

export default async function Home() {
  const report = await loadReport();
  return <ReportView report={report} />;
}
