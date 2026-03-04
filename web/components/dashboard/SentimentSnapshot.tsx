"use client";

import type { SentimentSummary } from "@/lib/types";

interface SentimentSnapshotProps {
  sentiment?: SentimentSummary | null;
}

export function SentimentSnapshot({ sentiment }: SentimentSnapshotProps) {
  const hasData = sentiment && sentiment.total_records > 0;
  const avgScore = sentiment?.avg_score ?? 0;
  const label = avgScore > 0.1 ? "BULLISH" : avgScore < -0.1 ? "BEARISH" : "NEUTRAL";
  const color = avgScore > 0.1 ? "var(--signal-buy)" : avgScore < -0.1 ? "var(--signal-sell)" : "var(--signal-hold)";

  return (
    <div className="glass-card p-6 h-full flex flex-col">
      {/* Header */}
      <div className="section-label mb-5">SENTIMENT</div>

      {!hasData ? (
        <div className="flex-1 flex items-center justify-center">
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>NO DATA</span>
        </div>
      ) : (
        <>
          {/* Score label */}
          <div className="flex items-baseline gap-2.5 mb-2">
            <span
              className="text-[10px] font-medium tracking-[0.2em]"
              style={{ color, fontFamily: "'Space Mono', monospace" }}
            >
              {label}
            </span>
          </div>

          {/* Score value */}
          <div className="flex items-baseline gap-1 mb-5">
            <span className="font-num text-4xl font-bold number-glow" style={{ color }}>
              {avgScore > 0 ? "+" : ""}{avgScore.toFixed(2)}
            </span>
          </div>

          {/* Sentiment bar */}
          <div className="flex-1">
            <div
              className="h-1.5 rounded-full flex overflow-hidden mb-3"
              style={{ background: "rgba(255,255,255,0.03)" }}
            >
              <div
                className="h-full transition-all duration-700"
                style={{
                  width: `${sentiment!.bullish_ratio * 100}%`,
                  background: "linear-gradient(90deg, rgba(232,82,74,0.6), rgba(232,82,74,0.9))",
                }}
              />
              <div
                className="h-full transition-all duration-700"
                style={{
                  width: `${sentiment!.neutral_ratio * 100}%`,
                  background: "rgba(255,193,7,0.4)",
                }}
              />
              <div
                className="h-full transition-all duration-700"
                style={{
                  width: `${sentiment!.bearish_ratio * 100}%`,
                  background: "linear-gradient(90deg, rgba(38,166,154,0.6), rgba(38,166,154,0.9))",
                }}
              />
            </div>

            {/* Legend row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--signal-buy)" }} />
                <span
                  className="text-[9px] tracking-wider"
                  style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
                >
                  {Math.round(sentiment!.bullish_ratio * 100)}%
                </span>
              </div>
              <span
                className="font-num text-[10px]"
                style={{ color: "var(--text-muted)" }}
              >
                {sentiment!.total_records} 則
              </span>
              <div className="flex items-center gap-1.5">
                <span
                  className="text-[9px] tracking-wider"
                  style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
                >
                  {Math.round(sentiment!.bearish_ratio * 100)}%
                </span>
                <div className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--signal-sell)" }} />
              </div>
            </div>
          </div>

          {/* Bottom divider + detail */}
          <div className="divider-gold my-4" />
          <div className="flex items-center justify-between">
            <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>情緒傾向</span>
            <span className="font-num text-sm font-medium" style={{ color }}>
              {avgScore > 0.1 ? "偏多" : avgScore < -0.1 ? "偏空" : "中性"}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
