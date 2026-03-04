"use client";

import type { PipelineResult } from "@/lib/types";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST } from "@/lib/constants";
import { formatPrice, formatPercent } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus, Shield, ShieldOff } from "lucide-react";

interface SummaryStripProps {
  result: PipelineResult;
}

export function SummaryStrip({ result }: SummaryStripProps) {
  const signal = result.signal || "hold";
  const signalColor = SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;
  const signalLabel = SIGNAL_LABELS[signal] || signal;
  const change = result.predicted_change || 0;
  const approved = result.agent?.approved;
  const positionSize = result.agent?.position_size;

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

        {/* Price → Target */}
        <div className="shrink-0">
          <div className="text-[9px] tracking-wider" style={{ color: "var(--text-muted)" }}>
            現價 → 目標
          </div>
          <div className="font-num text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            ${formatPrice(result.current_price)}
            <span style={{ color: "var(--text-muted)" }}> → </span>
            <span style={{ color: change >= 0 ? "var(--signal-buy)" : "var(--signal-sell)" }}>
              ${formatPrice(result.target_price)}
            </span>
            <span className="ml-1 text-xs" style={{ color: change >= 0 ? "var(--signal-buy)" : "var(--signal-sell)" }}>
              {formatPercent(change)}
            </span>
          </div>
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
