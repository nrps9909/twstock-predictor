"use client";

import type { AnalysisResult } from "@/lib/types";
import { FACTOR_LABELS, REGIME_LABELS, SIGNAL_COLORS, SIGNAL_LABELS } from "@/lib/constants";
import { Shield, ShieldOff, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface OverviewTabProps {
  result: AnalysisResult;
}

export function OverviewTab({ result }: OverviewTabProps) {
  const narrative = result.narrative;
  const risk = result.risk_decision;

  // Top 5 factors by weight
  const topFactors = Object.entries(result.factor_details || {})
    .filter(([, d]) => d.available)
    .sort((a, b) => (b[1].weight || 0) - (a[1].weight || 0))
    .slice(0, 5);

  const regimeColor = result.regime === "bull"
    ? "#EF5350"
    : result.regime === "bear"
    ? "#26A69A"
    : "#FFC107";

  return (
    <div className="space-y-6">
      {/* Risk Decision Card */}
      <div className="glass-card p-5">
        <div
          className="text-[9px] tracking-[0.15em] font-semibold mb-4"
          style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
        >
          RISK DECISION
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MiniStat label="動作" value={SIGNAL_LABELS[risk.action] || risk.action} />
          <MiniStat
            label="倉位"
            value={`${(risk.position_size * 100).toFixed(1)}%`}
          />
          {risk.stop_loss != null && (
            <MiniStat label="停損" value={`$${risk.stop_loss.toFixed(1)}`} />
          )}
          {risk.take_profit != null && (
            <MiniStat label="停利" value={`$${risk.take_profit.toFixed(1)}`} />
          )}
        </div>
        {risk.risk_notes && risk.risk_notes.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {risk.risk_notes.map((note, i) => (
              <span
                key={i}
                className="rounded-md px-2 py-0.5 text-[10px]"
                style={{
                  background: "rgba(255,193,7,0.08)",
                  color: "#FFC107",
                  border: "1px solid rgba(255,193,7,0.15)",
                }}
              >
                {note}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Narrative Card */}
      {narrative && narrative.outlook && (
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div
              className="text-[9px] tracking-[0.15em] font-semibold"
              style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
            >
              NARRATIVE {narrative.source === "llm" ? "(LLM)" : "(Algorithm)"}
            </div>
            <span
              className="rounded-md px-2 py-0.5 text-[10px] font-medium"
              style={{ background: `${regimeColor}15`, color: regimeColor, border: `1px solid ${regimeColor}25` }}
            >
              {REGIME_LABELS[result.regime] || result.regime}
            </span>
          </div>

          <p className="text-sm leading-relaxed mb-4" style={{ color: "var(--text-primary)" }}>
            {narrative.outlook}
            {narrative.outlook_horizon && (
              <span className="ml-2 text-xs" style={{ color: "var(--text-muted)" }}>
                ({narrative.outlook_horizon})
              </span>
            )}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Key drivers */}
            {narrative.key_drivers && narrative.key_drivers.length > 0 && (
              <div>
                <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
                  關鍵驅動
                </div>
                <ul className="space-y-1">
                  {narrative.key_drivers.map((d, i) => (
                    <li key={i} className="text-xs flex items-start gap-1.5" style={{ color: "var(--text-secondary)" }}>
                      <TrendingUp className="h-3 w-3 mt-0.5 shrink-0" style={{ color: "var(--accent-gold)" }} />
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risks */}
            {narrative.risks && narrative.risks.length > 0 && (
              <div>
                <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
                  風險因素
                </div>
                <ul className="space-y-1">
                  {narrative.risks.map((r, i) => (
                    <li key={i} className="text-xs flex items-start gap-1.5" style={{ color: "var(--text-secondary)" }}>
                      <TrendingDown className="h-3 w-3 mt-0.5 shrink-0" style={{ color: "#EF5350" }} />
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Catalysts */}
          {narrative.catalysts && narrative.catalysts.length > 0 && (
            <div className="mt-4">
              <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
                潛在催化劑
              </div>
              <div className="flex flex-wrap gap-2">
                {narrative.catalysts.map((c, i) => (
                  <span
                    key={i}
                    className="rounded-md px-2 py-0.5 text-[10px]"
                    style={{
                      background: "rgba(201,168,76,0.06)",
                      color: "var(--text-secondary)",
                      border: "1px solid rgba(201,168,76,0.12)",
                    }}
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Top 5 factors summary */}
      {topFactors.length > 0 && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-3"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            TOP FACTORS
          </div>
          <div className="space-y-2">
            {topFactors.map(([key, detail]) => {
              const scoreColor =
                detail.score > 0.6 ? "#EF5350" : detail.score < 0.4 ? "#26A69A" : "#FFC107";
              return (
                <div key={key} className="flex items-center gap-3">
                  <span className="text-xs w-20 shrink-0" style={{ color: "var(--text-secondary)" }}>
                    {FACTOR_LABELS[key] || key}
                  </span>
                  <div className="flex-1 relative h-1.5 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                    <div
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${detail.score * 100}%`,
                        background: scoreColor,
                        opacity: 0.6,
                      }}
                    />
                  </div>
                  <span className="font-num text-xs w-8 text-right" style={{ color: scoreColor }}>
                    {detail.score.toFixed(2)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Reasoning */}
      {result.reasoning && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-3"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            ANALYSIS SUMMARY
          </div>
          <p className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
            {result.reasoning}
          </p>
        </div>
      )}
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
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
      <div className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}
