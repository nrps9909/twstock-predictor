"use client";

import dynamic from "next/dynamic";
import type { PipelineResult, TechnicalResult } from "@/lib/types";
import { TechnicalSnapshot } from "@/components/dashboard/TechnicalSnapshot";
import { TechnicalIndicators } from "@/components/charts/TechnicalIndicators";
import { RadarChart } from "@/components/charts/RadarChart";
import { Loader2 } from "lucide-react";

// Dynamic import for CandlestickChart (avoids SSR issues with lightweight-charts)
const CandlestickChart = dynamic(
  () => import("@/components/charts/CandlestickChart").then((m) => m.CandlestickChart),
  { ssr: false, loading: () => <ChartLoading /> }
);

function ChartLoading() {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-5 w-5 animate-spin" style={{ color: "var(--accent-gold)" }} />
    </div>
  );
}

interface TechnicalTabProps {
  result: PipelineResult;
  technicalData: TechnicalResult | null;
}

export function TechnicalTab({ result, technicalData }: TechnicalTabProps) {
  const chartData = technicalData?.chart_data || [];
  const indicators = result.technical?.indicators || {};
  const signals = result.technical?.signals;

  // Build radar data from signals
  const radarData = signals
    ? Object.entries(signals)
        .filter(([k, v]) => k !== "summary" && v != null)
        .map(([k, v]) => {
          const s = v as { signal: string };
          const score = s.signal === "buy" ? 80 : s.signal === "sell" ? 20 : 50;
          return { name: k.toUpperCase(), value: score, fullMark: 100 };
        })
    : [];

  // Build candlestick data
  const candleData = chartData
    .filter((d) => d.open != null && d.high != null && d.low != null && d.close != null)
    .map((d) => ({
      date: d.date as string,
      open: d.open as number,
      high: d.high as number,
      low: d.low as number,
      close: d.close as number,
      volume: (d.volume as number) || 0,
    }));

  // MA overlays
  const overlays = {
    sma_5: chartData.map((d) => d.sma_5 as number).filter((v) => v != null),
    sma_20: chartData.map((d) => d.sma_20 as number).filter((v) => v != null),
    sma_60: chartData.map((d) => d.sma_60 as number).filter((v) => v != null),
  };

  return (
    <div className="space-y-6">
      {/* K-line chart */}
      {candleData.length > 0 && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            K-LINE CHART
          </div>
          <CandlestickChart
            data={candleData}
            height={380}
            overlays={
              overlays.sma_5.length > 0 || overlays.sma_20.length > 0
                ? overlays
                : undefined
            }
          />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Technical indicators chart */}
        {chartData.length > 0 && (
          <div className="glass-card p-5">
            <div
              className="text-[9px] tracking-[0.15em] font-semibold mb-4"
              style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
            >
              INDICATORS
            </div>
            <TechnicalIndicators data={chartData as any} height={220} />
          </div>
        )}

        {/* Radar chart */}
        {radarData.length > 0 && (
          <div className="glass-card p-5">
            <div
              className="text-[9px] tracking-[0.15em] font-semibold mb-4"
              style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
            >
              SIGNAL RADAR
            </div>
            <RadarChart data={radarData} height={220} />
          </div>
        )}
      </div>

      {/* Signal details */}
      <div className="glass-card p-5">
        <TechnicalSnapshot signals={signals} indicators={indicators} />
      </div>
    </div>
  );
}
