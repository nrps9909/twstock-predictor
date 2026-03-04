"use client";

import type { PipelineResult } from "@/lib/types";
import { SentimentSnapshot } from "@/components/dashboard/SentimentSnapshot";

interface SentimentTabProps {
  result: PipelineResult;
}

export function SentimentTab({ result }: SentimentTabProps) {
  const sentiment = result.sentiment;

  return (
    <div className="space-y-6">
      {/* Sentiment summary card */}
      <div className="max-w-md">
        <SentimentSnapshot sentiment={sentiment} />
      </div>

      {/* Source breakdown */}
      {sentiment && sentiment.by_source && Object.keys(sentiment.by_source).length > 0 && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            SOURCE BREAKDOWN
          </div>
          <div className="space-y-2">
            {Object.entries(sentiment.by_source).map(([source, data]) => {
              const avgColor = data.avg_score > 0.1
                ? "var(--signal-buy)"
                : data.avg_score < -0.1
                ? "var(--signal-sell)"
                : "var(--signal-hold)";

              return (
                <div key={source} className="flex items-center justify-between py-1.5">
                  <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    {source}
                  </span>
                  <div className="flex items-center gap-4">
                    <span className="font-num text-xs" style={{ color: "var(--text-muted)" }}>
                      {data.count} 則
                    </span>
                    <span className="font-num text-xs font-medium" style={{ color: avgColor }}>
                      {data.avg_score > 0 ? "+" : ""}{data.avg_score.toFixed(2)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Sentiment date info */}
      {sentiment?.latest_date && (
        <div className="text-center">
          <span className="text-[10px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            最新資料: {sentiment.latest_date}
          </span>
        </div>
      )}
    </div>
  );
}
