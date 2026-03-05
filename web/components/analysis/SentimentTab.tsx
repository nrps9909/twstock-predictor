"use client";

import type { AnalysisResult } from "@/lib/types";
import { FACTOR_LABELS } from "@/lib/constants";

interface SentimentTabProps {
  result: AnalysisResult;
}

export function SentimentTab({ result }: SentimentTabProps) {
  const newsSentiment = result.factor_details?.news_sentiment;
  const narrative = result.narrative;

  return (
    <div className="space-y-6">
      {/* News sentiment factor */}
      {newsSentiment && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            NEWS SENTIMENT FACTOR
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MiniStat
              label="情緒分數"
              value={newsSentiment.score.toFixed(2)}
              color={
                newsSentiment.score > 0.6
                  ? "#EF5350"
                  : newsSentiment.score < 0.4
                  ? "#26A69A"
                  : "#FFC107"
              }
            />
            <MiniStat
              label="可用性"
              value={newsSentiment.available ? "有效" : "無資料"}
              color={newsSentiment.available ? "rgba(34,197,94,0.8)" : "var(--text-muted)"}
            />
            <MiniStat
              label="權重"
              value={`${((newsSentiment.weight || 0) * 100).toFixed(0)}%`}
            />
            <MiniStat
              label="新鮮度"
              value={`${((newsSentiment.freshness || 0) * 100).toFixed(0)}%`}
            />
          </div>

          {/* Components breakdown */}
          {newsSentiment.components && Object.keys(newsSentiment.components).length > 0 && (
            <div className="mt-4 space-y-2">
              <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
                子分量
              </div>
              {Object.entries(newsSentiment.components).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between py-1">
                  <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    {key}
                  </span>
                  <span className="font-num text-xs" style={{ color: "var(--text-primary)" }}>
                    {typeof val === "number" ? val.toFixed(3) : String(val)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Narrative sentiment info */}
      {narrative && (narrative.key_drivers?.length > 0 || narrative.risks?.length > 0) && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            SENTIMENT FROM NARRATIVE
          </div>
          <div className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
            {narrative.outlook}
          </div>
          {narrative.outlook_horizon && (
            <span className="text-xs mt-1 inline-block" style={{ color: "var(--text-muted)" }}>
              展望期間: {narrative.outlook_horizon}
            </span>
          )}
        </div>
      )}

      {/* Related factors */}
      <div className="glass-card p-5">
        <div
          className="text-[9px] tracking-[0.15em] font-semibold mb-4"
          style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
        >
          SENTIMENT-RELATED FACTORS
        </div>
        <div className="space-y-2">
          {["margin_sentiment", "global_context", "news_sentiment"].map((key) => {
            const detail = result.factor_details?.[key];
            if (!detail) return null;
            const scoreColor =
              detail.score > 0.6 ? "#EF5350" : detail.score < 0.4 ? "#26A69A" : "#FFC107";

            return (
              <div key={key} className="flex items-center gap-3">
                <span className="text-xs w-20 shrink-0" style={{ color: "var(--text-secondary)" }}>
                  {FACTOR_LABELS[key] || key}
                </span>
                <div className="flex-1 relative h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                  <div
                    className="absolute inset-y-0 left-0 rounded-full"
                    style={{
                      width: detail.available ? `${detail.score * 100}%` : "0%",
                      background: scoreColor,
                      opacity: 0.7,
                    }}
                  />
                </div>
                <span className="font-num text-xs w-10 text-right" style={{ color: scoreColor }}>
                  {detail.available ? detail.score.toFixed(2) : "N/A"}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div
      className="rounded-lg p-3"
      style={{
        background: "rgba(255,255,255,0.015)",
        border: "1px solid rgba(255,255,255,0.04)",
      }}
    >
      <div className="text-[8px] tracking-wider uppercase mb-1" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="font-num text-sm font-bold" style={{ color: color || "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}
