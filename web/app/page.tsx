"use client";

import { Suspense, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { usePipeline } from "@/hooks/usePipeline";
import { api } from "@/lib/api";
import type { TechnicalResult } from "@/lib/types";
import { HeroInput } from "@/components/analysis/HeroInput";
import { SummaryStrip } from "@/components/analysis/SummaryStrip";
import { ResultTabs } from "@/components/analysis/ResultTabs";
import { PipelineProgress } from "@/components/dashboard/PipelineProgress";
import { AlertTriangle, RotateCcw } from "lucide-react";

export default function AnalysisPage() {
  return (
    <Suspense>
      <AnalysisPageInner />
    </Suspense>
  );
}

function AnalysisPageInner() {
  const searchParams = useSearchParams();
  const prefillStock = searchParams.get("stock");

  const {
    state,
    progress,
    phases,
    result,
    error,
    run,
    abort,
    reset,
  } = usePipeline();

  // Auto-run if URL has ?stock=XXXX
  useEffect(() => {
    if (prefillStock && /^\d{4}$/.test(prefillStock) && state === "idle") {
      run(prefillStock);
    }
    // Only on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleAnalyze = useCallback(
    (stockId: string) => {
      run(stockId);
    },
    [run]
  );

  // Fetch technical data when analysis completes (for charts)
  const { data: technicalData = null } = useQuery<TechnicalResult | null>({
    queryKey: ["technical", result?.stock_id],
    queryFn: () =>
      result?.stock_id
        ? api.getTechnical(result.stock_id, 120)
        : Promise.resolve(null),
    enabled: !!result?.stock_id && state === "complete",
    staleTime: 5 * 60 * 1000,
  });

  const isComplete = state === "complete" && result !== null;
  const isRunning = state === "running";
  const isError = state === "error";
  const currentStockId = result?.stock_id || null;

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      <div className="flex-1 overflow-auto">
        <div className="mx-auto max-w-5xl px-6 py-6 space-y-6">
          {/* Hero Input */}
          <HeroInput
            onAnalyze={handleAnalyze}
            onAbort={abort}
            isRunning={isRunning}
            isCompact={isComplete}
            currentStock={currentStockId}
          />

          {/* Pipeline Progress (visible during analysis) */}
          {isRunning && (
            <div className="space-y-4 card-reveal">
              <PipelineProgress phases={phases} progress={progress} />
            </div>
          )}

          {/* Error state */}
          {isError && (
            <div className="glass-card p-8 text-center card-reveal">
              <AlertTriangle
                className="h-8 w-8 mx-auto mb-3"
                style={{ color: "#EF5350" }}
              />
              <p className="text-sm mb-1" style={{ color: "#EF5350" }}>
                分析失敗
              </p>
              <p
                className="text-xs mb-4"
                style={{ color: "var(--text-muted)" }}
              >
                {error || "未知錯誤，請重試"}
              </p>
              <button
                onClick={reset}
                className="inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all"
                style={{
                  background: "rgba(201,168,76,0.08)",
                  border: "1px solid rgba(201,168,76,0.15)",
                  color: "var(--accent-gold)",
                }}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                重新分析
              </button>
            </div>
          )}

          {/* Results section */}
          {isComplete && result && (
            <div className="space-y-6 card-reveal">
              <SummaryStrip result={result} />
              <ResultTabs
                result={result}
                technicalData={technicalData}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
