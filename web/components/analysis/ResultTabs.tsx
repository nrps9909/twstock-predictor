"use client";

import type { AnalysisResult, TechnicalResult } from "@/lib/types";
import { OverviewTab } from "./OverviewTab";
import { DetailTab } from "./DetailTab";

export type ResultView = "overview" | "detail";

interface ResultTabsProps {
  result: AnalysisResult;
  technicalData: TechnicalResult | null;
  view: ResultView;
}

export function ResultTabs({ result, technicalData, view }: ResultTabsProps) {
  return (
    <div>
      {view === "overview" && <OverviewTab result={result} />}
      {view === "detail" && <DetailTab result={result} technicalData={technicalData} />}
    </div>
  );
}
