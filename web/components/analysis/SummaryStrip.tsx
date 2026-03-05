"use client";

import type { AnalysisResult } from "@/lib/types";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST, REGIME_LABELS } from "@/lib/constants";
import { TrendingUp, TrendingDown, Minus, Shield, ShieldOff } from "lucide-react";

interface SummaryStripProps {
  result: AnalysisResult;
}

export function SummaryStrip({ result }: SummaryStripProps) {
  const signal = result.signal || "hold";
  const signalColor = SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;
  const signalLabel = SIGNAL_LABELS[signal] || signal;
  const approved = result.risk_decision?.approved;
  const positionSize = result.risk_decision?.position_size;
  const regime = result.regime || "sideways";
  const regimeColor = regime === "bull"
    ? "#EF5350"
    : regime === "bear"
    ? "#26A69A"
    : "#FFC107";

  const SignalIcon = signal.includes("buy")
    ? TrendingUp
    : signal.includes("sell")
    ? TrendingDown
    : Minus;

  return (
    <div
      className="glass-card-static overflow-hidden"
      style={{ borderColor: `${signalColor}15` }}
    >
      <div className="flex items-center gap-6 px-6 py-4 overflow-x-auto">
        {/* Stock info */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="font-num text-lg font-bold" style={{ color: "var(--accent-gold)" }}>
            {result.stock_id}
          </span>
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            {result.stock_name || STOCK_LIST[result.stock_id] || ""}
          </span>
        </div>

        {/* Divider */}
        <div className="h-8 w-px shrink-0" style={{ background: "var(--border)" }} />

        {/* Signal badge */}
        <div className="flex items-center gap-2 shrink-0">
          <div
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold"
            style={{
              background: `${signalColor}12`,
              color: signalColor,
              border: `1px solid ${signalColor}25`,
            }}
          >
            <SignalIcon className="h-3.5 w-3.5" />
            {signalLabel}
          </div>
        </div>

        {/* Confidence */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            信心度
          </div>
          <div className="font-num text-sm font-bold" style={{ color: signalColor }}>
            {Math.round(result.confidence * 100)}%
          </div>
        </div>

        {/* Total Score */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            總分
          </div>
          <div className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
            {result.total_score.toFixed(2)}
          </div>
        </div>

        {/* Current Price */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            現價
          </div>
          <div className="font-num text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            ${result.current_price.toFixed(1)}
            {result.price_change_pct !== 0 && (
              <span
                className="ml-1 text-xs"
                style={{ color: result.price_change_pct >= 0 ? "var(--signal-buy)" : "var(--signal-sell)" }}
              >
                {result.price_change_pct >= 0 ? "+" : ""}{result.price_change_pct.toFixed(2)}%
              </span>
            )}
          </div>
        </div>

        {/* Regime badge */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            體制
          </div>
          <span
            className="inline-block rounded-md px-2 py-0.5 text-[10px] font-bold"
            style={{
              background: `${regimeColor}12`,
              color: regimeColor,
              border: `1px solid ${regimeColor}25`,
            }}
          >
            {REGIME_LABELS[regime] || regime}
          </span>
        </div>

        {/* Risk approval */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            風控
          </div>
          <div className="flex items-center gap-1">
            {approved ? (
              <>
                <Shield className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.8)" }} />
                <span className="text-xs font-medium" style={{ color: "rgba(34,197,94,0.8)" }}>通過</span>
              </>
            ) : (
              <>
                <ShieldOff className="h-3.5 w-3.5" style={{ color: "rgba(232,82,74,0.8)" }} />
                <span className="text-xs font-medium" style={{ color: "rgba(232,82,74,0.8)" }}>否決</span>
              </>
            )}
          </div>
        </div>

        {/* Position size */}
        {positionSize != null && positionSize > 0 && (
          <div className="shrink-0">
            <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
              建議倉位
            </div>
            <div className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
              {(positionSize * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
