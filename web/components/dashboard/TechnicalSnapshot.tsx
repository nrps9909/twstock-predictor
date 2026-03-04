"use client";

import type { TechnicalSignals } from "@/lib/types";
import { SIGNAL_COLORS } from "@/lib/constants";
import { Activity } from "lucide-react";

interface TechnicalSnapshotProps {
  signals?: TechnicalSignals;
  indicators?: Record<string, number | null>;
}

const INDICATOR_LABELS: Record<string, string> = {
  kd: "KD",
  rsi: "RSI",
  macd: "MACD",
  bias: "BIAS",
  bb: "BOLL",
};

function SignalDot({ signal }: { signal: string }) {
  const color = SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || "rgba(139,144,160,0.5)";
  return (
    <div className="relative flex items-center justify-center">
      <div
        className="h-[6px] w-[6px] rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}60` }}
      />
    </div>
  );
}

export function TechnicalSnapshot({ signals, indicators }: TechnicalSnapshotProps) {
  if (!signals) {
    return (
      <div className="glass-card p-6 h-full flex flex-col">
        <div className="section-label mb-5">TECHNICAL</div>
        <div className="flex-1 flex items-center justify-center">
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>NO DATA</span>
        </div>
      </div>
    );
  }

  const summary = signals.summary;
  const summarySignal = summary?.signal || "hold";
  const summaryColor = SIGNAL_COLORS[summarySignal as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;

  const signalEntries = Object.entries(signals).filter(([k, v]) => k !== "summary" && v != null);

  return (
    <div className="glass-card p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="section-label">TECHNICAL</div>
        {summary && (
          <div className="flex items-center gap-1.5">
            <Activity className="h-3 w-3" style={{ color: summaryColor }} />
            <span
              className="font-num text-[10px] tracking-wider font-medium"
              style={{ color: summaryColor }}
            >
              {summary.raw_score > 0 ? "+" : ""}{summary.raw_score}/{summary.max_score}
            </span>
          </div>
        )}
      </div>

      {/* Signal grid */}
      <div className="flex-1 space-y-2.5">
        {signalEntries.map(([key, sig]) => {
          const s = sig as { signal: string; reason: string };
          const color = SIGNAL_COLORS[s.signal as keyof typeof SIGNAL_COLORS] || "rgba(139,144,160,0.5)";
          const label = s.signal === "buy" ? "BUY" : s.signal === "sell" ? "SELL" : "HOLD";

          return (
            <div key={key} className="flex items-center justify-between group">
              <span
                className="text-[10px] tracking-wider"
                style={{ color: "var(--text-secondary)", fontFamily: "'Space Mono', monospace" }}
              >
                {INDICATOR_LABELS[key] || key.toUpperCase()}
              </span>
              <div className="flex items-center gap-2.5">
                <SignalDot signal={s.signal} />
                <span
                  className="text-[10px] font-medium tracking-wider w-8 text-right"
                  style={{ color, fontFamily: "'Space Mono', monospace" }}
                >
                  {label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Key metrics */}
      {indicators && (indicators.rsi_14 != null || indicators.kd_k != null) && (
        <>
          <div className="divider-gold my-4" />
          <div className="grid grid-cols-2 gap-3">
            {indicators.rsi_14 != null && (
              <div>
                <div
                  className="text-[9px] tracking-wider mb-1"
                  style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
                >
                  RSI-14
                </div>
                <div className="font-num text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                  {typeof indicators.rsi_14 === "number" ? indicators.rsi_14.toFixed(1) : indicators.rsi_14}
                </div>
              </div>
            )}
            {indicators.kd_k != null && (
              <div>
                <div
                  className="text-[9px] tracking-wider mb-1"
                  style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
                >
                  K / D
                </div>
                <div className="font-num text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                  {typeof indicators.kd_k === "number" ? indicators.kd_k.toFixed(0) : indicators.kd_k}
                  <span style={{ color: "var(--text-muted)" }}> / </span>
                  {indicators.kd_d != null
                    ? typeof indicators.kd_d === "number" ? indicators.kd_d.toFixed(0) : indicators.kd_d
                    : "—"}
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
