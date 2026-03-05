"use client";

import type { AnalysisResult } from "@/lib/types";
import { FACTOR_GROUPS, FACTOR_LABELS, REGIME_LABELS } from "@/lib/constants";

interface FactorTabProps {
  result: AnalysisResult;
}

export function FactorTab({ result }: FactorTabProps) {
  const factors = result.factor_details || {};
  const breakdown = result.confidence_breakdown;

  return (
    <div className="space-y-6">
      {/* Factor groups */}
      {Object.entries(FACTOR_GROUPS).map(([groupName, factorKeys]) => (
        <div key={groupName} className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            {groupName.toUpperCase()}
          </div>
          <div className="space-y-2.5">
            {factorKeys.map((key) => {
              const detail = factors[key];
              if (!detail) return null;
              const score = detail.score ?? 0.5;
              const available = detail.available;
              const weight = detail.weight ?? 0;
              const label = FACTOR_LABELS[key] || key;
              const scoreColor =
                score > 0.6 ? "#EF5350" : score < 0.4 ? "#26A69A" : "#FFC107";

              return (
                <div key={key} className="flex items-center gap-3">
                  {/* Label */}
                  <span
                    className="text-xs w-20 shrink-0 truncate"
                    style={{
                      color: available ? "var(--text-secondary)" : "var(--text-muted)",
                      opacity: available ? 1 : 0.5,
                    }}
                  >
                    {label}
                  </span>

                  {/* Score bar */}
                  <div className="flex-1 relative h-2 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                    <div
                      className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
                      style={{
                        width: available ? `${score * 100}%` : "0%",
                        background: scoreColor,
                        opacity: 0.7,
                      }}
                    />
                    {/* Center marker at 50% */}
                    <div
                      className="absolute top-0 bottom-0 w-px"
                      style={{ left: "50%", background: "rgba(255,255,255,0.1)" }}
                    />
                  </div>

                  {/* Score value */}
                  <span
                    className="font-num text-xs w-10 text-right shrink-0"
                    style={{ color: available ? scoreColor : "var(--text-muted)" }}
                  >
                    {available ? score.toFixed(2) : "N/A"}
                  </span>

                  {/* Weight */}
                  <span
                    className="font-num text-[10px] w-10 text-right shrink-0"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {(weight * 100).toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}

      {/* Bottom row: Confidence breakdown + Regime */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Confidence breakdown */}
        {breakdown && (
          <div className="glass-card p-5">
            <div
              className="text-[9px] tracking-[0.15em] font-semibold mb-3"
              style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
            >
              CONFIDENCE BREAKDOWN
            </div>
            <div className="space-y-2">
              <ConfRow label="一致性" value={breakdown.confidence_agreement} />
              <ConfRow label="強度" value={breakdown.confidence_strength} />
              <ConfRow label="覆蓋率" value={breakdown.confidence_coverage} />
              <ConfRow label="新鮮度" value={breakdown.confidence_freshness} />
              <div className="pt-2 mt-2" style={{ borderTop: "1px solid var(--border)" }}>
                <ConfRow label="風險折扣" value={breakdown.risk_discount} highlight />
              </div>
            </div>
          </div>
        )}

        {/* Regime badge */}
        <div className="glass-card p-5 flex flex-col items-center justify-center gap-3">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            MARKET REGIME
          </div>
          <div
            className="rounded-xl px-6 py-3 text-lg font-bold"
            style={{
              background: result.regime === "bull"
                ? "rgba(239,83,80,0.1)"
                : result.regime === "bear"
                ? "rgba(38,166,154,0.1)"
                : "rgba(255,193,7,0.1)",
              color: result.regime === "bull"
                ? "#EF5350"
                : result.regime === "bear"
                ? "#26A69A"
                : "#FFC107",
              border: `1px solid ${
                result.regime === "bull"
                  ? "rgba(239,83,80,0.2)"
                  : result.regime === "bear"
                  ? "rgba(38,166,154,0.2)"
                  : "rgba(255,193,7,0.2)"
              }`,
            }}
          >
            {REGIME_LABELS[result.regime] || result.regime}
          </div>
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
            HMM 3-state 偵測
          </span>
        </div>
      </div>
    </div>
  );
}

function ConfRow({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{label}</span>
      <span
        className="font-num text-xs font-medium"
        style={{ color: highlight ? "var(--accent-gold)" : "var(--text-primary)" }}
      >
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
