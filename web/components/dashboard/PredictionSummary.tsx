"use client";

import type { PipelineResult } from "@/lib/types";
import { SIGNAL_COLORS, SIGNAL_LABELS } from "@/lib/constants";
import { formatPrice, formatPercent } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus, Shield } from "lucide-react";

interface PredictionSummaryProps {
  result: PipelineResult;
}

export function PredictionSummary({ result }: PredictionSummaryProps) {
  const signal = result.signal || "hold";
  const signalColor = SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;
  const signalLabel = SIGNAL_LABELS[signal] || signal;
  const confidence = result.confidence || 0;
  const change = result.predicted_change || 0;

  const SignalIcon = signal.includes("buy")
    ? TrendingUp
    : signal.includes("sell")
    ? TrendingDown
    : Minus;

  return (
    <div className="glass-card p-6 h-full flex flex-col">
      {/* Header row */}
      <div className="flex items-center justify-between mb-5">
        <div className="section-label">SIGNAL</div>
        {result.agent && (
          <div className="flex items-center gap-1.5" style={{ color: result.agent.approved ? "rgba(34,197,94,0.7)" : "rgba(232,82,74,0.7)" }}>
            <Shield className="h-3 w-3" />
            <span className="text-[10px] font-medium tracking-wider" style={{ fontFamily: "'Space Mono', monospace" }}>
              {result.agent.approved ? "APPROVED" : "DENIED"}
            </span>
          </div>
        )}
      </div>

      {/* Signal badge */}
      <div className="flex items-center gap-3 mb-5">
        <div
          className="flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-bold breath-glow"
          style={{
            backgroundColor: `${signalColor}12`,
            color: signalColor,
            border: `1px solid ${signalColor}20`,
          }}
        >
          <SignalIcon className="h-4 w-4" />
          {signalLabel}
        </div>
      </div>

      {/* Confidence */}
      <div className="flex-1">
        <div className="text-[10px] tracking-wider mb-2" style={{ color: "var(--text-secondary)", fontFamily: "'Space Mono', monospace" }}>
          CONFIDENCE
        </div>
        <div className="flex items-baseline gap-1 mb-3">
          <span className="font-num text-4xl font-bold number-glow" style={{ color: signalColor }}>
            {Math.round(confidence * 100)}
          </span>
          <span className="font-num text-lg" style={{ color: `${signalColor}80` }}>%</span>
        </div>

        {/* Bar */}
        <div className="h-1 rounded-full overflow-hidden mb-5" style={{ background: "rgba(255,255,255,0.03)" }}>
          <div
            className="h-full rounded-full transition-all duration-1000 ease-out"
            style={{
              width: `${confidence * 100}%`,
              background: `linear-gradient(90deg, ${signalColor}60, ${signalColor})`,
              boxShadow: `0 0 8px ${signalColor}40`,
            }}
          />
        </div>
      </div>

      {/* Bottom stats */}
      <div className="divider-gold mb-4" />
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>目標價</span>
          <div className="font-num text-sm font-medium">
            <span>${formatPrice(result.target_price)}</span>
            <span className="ml-2" style={{ color: change >= 0 ? "var(--signal-buy)" : "var(--signal-sell)" }}>
              {formatPercent(change)}
            </span>
          </div>
        </div>

        {result.agent && result.agent.position_size > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>建議倉位</span>
            <span className="font-num text-sm font-medium">
              {(result.agent.position_size * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
