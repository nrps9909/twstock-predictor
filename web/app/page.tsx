"use client";

import { Suspense, useEffect, useCallback, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { usePipeline } from "@/hooks/usePipeline";
import { api } from "@/lib/api";
import type { TechnicalResult } from "@/lib/types";
import type { ResultView } from "@/components/analysis/ResultTabs";
import { HeroInput } from "@/components/analysis/HeroInput";
import { SummaryStrip } from "@/components/analysis/SummaryStrip";
import { ResultTabs } from "@/components/analysis/ResultTabs";
import { PipelineProgress } from "@/components/dashboard/PipelineProgress";
import { PipelineDetail } from "@/components/dashboard/PipelineDetail";
import { MarketPulse } from "@/components/analysis/MarketPulse";
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
  const forceRerun = searchParams.get("t");

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

  const [view, setView] = useState<ResultView>("overview");

  // Auto-run if URL has ?stock=XXXX
  useEffect(() => {
    if (prefillStock && /^\d{4}$/.test(prefillStock)) {
      const shouldRun = state === "idle"
        || forceRerun
        || (state === "complete" && result?.stock_id !== prefillStock);
      if (shouldRun) run(prefillStock);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleAnalyze = useCallback(
    (stockId: string) => {
      setView("overview"); // Reset to overview on new analysis
      run(stockId);
    },
    [run]
  );

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
  const isIdle = state === "idle";
  const currentStockId = result?.stock_id || null;

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-auto">
        {/* Sticky SummaryStrip */}
        {isComplete && result && (
          <div className="sticky top-0 z-40 backdrop-blur-sticky">
            <div className="mx-auto max-w-[1440px] px-6 xl:px-10">
              <SummaryStrip
                result={result}
                onAnalyze={handleAnalyze}
                onReset={reset}
                currentStock={currentStockId}
                view={view}
                onViewChange={setView}
              />
            </div>
          </div>
        )}

        <div className="mx-auto max-w-[1440px] px-6 xl:px-10 py-6 space-y-6">
          {/* Hero (idle + running) */}
          {!isComplete && (
            <HeroInput
              onAnalyze={handleAnalyze}
              onAbort={abort}
              isRunning={isRunning}
            />
          )}

          {/* MarketPulse — idle */}
          {isIdle && <MarketPulse onAnalyze={handleAnalyze} />}

          {/* Pipeline — running state */}
          {isRunning && (
            <div
              className="grid grid-cols-1 lg:grid-cols-[180px_1fr] gap-3 card-reveal"
              style={{ height: "clamp(280px, 50vh, 420px)" }}
            >
              <div className="glass-card-static p-3 overflow-hidden hidden lg:block">
                <PipelineProgress phases={phases} progress={progress} />
              </div>
              <div className="glass-card-static overflow-hidden flex flex-col">
                {/* Mobile: show progress bar inline */}
                <div className="lg:hidden px-4 pt-3">
                  <div className="relative h-[2px] rounded-full overflow-hidden mb-2" style={{ background: "rgba(255,255,255,0.03)" }}>
                    <div
                      className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                      style={{
                        width: `${progress}%`,
                        background: "linear-gradient(90deg, #B8962F, #C9A84C)",
                      }}
                    />
                  </div>
                </div>
                <PipelineDetail phases={phases} />
              </div>
            </div>
          )}

          {/* Error state */}
          {isError && (
            <div className="glass-card p-8 text-center card-reveal">
              <AlertTriangle className="h-8 w-8 mx-auto mb-3" style={{ color: "#EF5350" }} />
              <p className="text-sm mb-1" style={{ color: "#EF5350" }}>分析失敗</p>
              <p className="text-xs mb-4" style={{ color: "var(--text-muted)" }}>
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

          {/* Results */}
          {isComplete && result && (
            <div className="space-y-6 card-reveal">
              <ResultTabs result={result} technicalData={technicalData} view={view} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
