"use client";

import { useState } from "react";
import Link from "next/link";
import type { AnalysisResult } from "@/lib/types";
import type { ResultView } from "./ResultTabs";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST, REGIME_LABELS } from "@/lib/constants";
import {
  TrendingUp, TrendingDown, Minus, Shield, ShieldOff,
  Search, RotateCcw, ArrowRight, Clock, FileText, BarChart3,
} from "lucide-react";

interface SummaryStripProps {
  result: AnalysisResult;
  onAnalyze?: (stockId: string) => void;
  onReset?: () => void;
  currentStock?: string | null;
  view: ResultView;
  onViewChange: (view: ResultView) => void;
}

export function SummaryStrip({ result, onAnalyze, onReset, currentStock, view, onViewChange }: SummaryStripProps) {
  const [miniInput, setMiniInput] = useState("");
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

  const handleMiniSubmit = () => {
    const trimmed = miniInput.trim();
    if (!trimmed || !/^\d{4}$/.test(trimmed)) return;
    onAnalyze?.(trimmed);
    setMiniInput("");
  };

  return (
    <div
      className="glass-card-static overflow-hidden"
      style={{ borderColor: `${signalColor}15` }}
    >
      <div className="flex items-center gap-4 px-5 py-3 overflow-x-auto">
        {/* Stock info */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="font-num text-base font-bold" style={{ color: "var(--accent-gold)" }}>
            {result.stock_id}
          </span>
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {result.stock_name || STOCK_LIST[result.stock_id] || ""}
          </span>
        </div>

        {/* Divider */}
        <div className="h-7 w-px shrink-0" style={{ background: "var(--border)" }} />

        {/* Signal badge */}
        <div
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] font-bold shrink-0"
          style={{
            background: `${signalColor}12`,
            color: signalColor,
            border: `1px solid ${signalColor}25`,
          }}
        >
          <SignalIcon className="h-3 w-3" />
          {signalLabel}
        </div>

        {/* Confidence */}
        <div className="shrink-0">
          <div className="text-[8px] tracking-wider" style={{ color: "var(--text-muted)" }}>信心</div>
          <div className="font-num text-xs font-bold" style={{ color: signalColor }}>
            {Math.round(result.confidence * 100)}%
          </div>
        </div>

        {/* Total Score */}
        <div className="shrink-0">
          <div className="text-[8px] tracking-wider" style={{ color: "var(--text-muted)" }}>總分</div>
          <div className="font-num text-xs font-bold" style={{ color: "var(--text-primary)" }}>
            {result.total_score.toFixed(2)}
          </div>
        </div>

        {/* Current Price */}
        <div className="shrink-0">
          <div className="text-[8px] tracking-wider" style={{ color: "var(--text-muted)" }}>現價</div>
          <div className="font-num text-xs font-medium" style={{ color: "var(--text-primary)" }}>
            ${result.current_price.toFixed(1)}
            {result.price_change_pct !== 0 && (
              <span
                className="ml-1 text-[10px]"
                style={{ color: result.price_change_pct >= 0 ? "var(--signal-buy)" : "var(--signal-sell)" }}
              >
                {result.price_change_pct >= 0 ? "+" : ""}{result.price_change_pct.toFixed(2)}%
              </span>
            )}
          </div>
        </div>

        {/* Regime badge */}
        <span
          className="inline-block rounded-md px-1.5 py-0.5 text-[9px] font-bold shrink-0"
          style={{
            background: `${regimeColor}12`,
            color: regimeColor,
            border: `1px solid ${regimeColor}25`,
          }}
        >
          {REGIME_LABELS[regime] || regime}
        </span>

        {/* Risk approval */}
        <div className="flex items-center gap-1 shrink-0">
          {approved ? (
            <Shield className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.8)" }} />
          ) : (
            <ShieldOff className="h-3.5 w-3.5" style={{ color: "rgba(232,82,74,0.8)" }} />
          )}
        </div>

        {/* Divider */}
        <div className="h-7 w-px shrink-0" style={{ background: "var(--border)" }} />

        {/* View toggle */}
        <div className="flex items-center shrink-0 rounded-lg overflow-hidden" style={{ border: "1px solid var(--border)" }}>
          <button
            onClick={() => onViewChange("overview")}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-medium transition-all"
            style={{
              background: view === "overview" ? "rgba(201,168,76,0.12)" : "transparent",
              color: view === "overview" ? "var(--accent-gold)" : "var(--text-muted)",
            }}
          >
            <FileText className="h-3 w-3" />
            總覽
          </button>
          <div className="w-px h-5" style={{ background: "var(--border)" }} />
          <button
            onClick={() => onViewChange("detail")}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-medium transition-all"
            style={{
              background: view === "detail" ? "rgba(201,168,76,0.12)" : "transparent",
              color: view === "detail" ? "var(--accent-gold)" : "var(--text-muted)",
            }}
          >
            <BarChart3 className="h-3 w-3" />
            詳細資料
          </button>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Inline mini search + actions */}
        {onAnalyze && (
          <div className="flex items-center gap-2 shrink-0">
            <div className="flex items-center">
              <input
                type="text"
                value={miniInput}
                onChange={(e) => setMiniInput(e.target.value.replace(/\D/g, "").slice(0, 4))}
                onKeyDown={(e) => e.key === "Enter" && handleMiniSubmit()}
                placeholder="代碼"
                className="w-16 rounded-l-lg px-2 py-1.5 text-xs font-num outline-none"
                style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid var(--border)",
                  borderRight: "none",
                  color: "var(--text-primary)",
                }}
              />
              <button
                onClick={handleMiniSubmit}
                disabled={!/^\d{4}$/.test(miniInput.trim())}
                className="rounded-r-lg px-2 py-1.5 transition-all disabled:opacity-30"
                style={{
                  background: "rgba(201,168,76,0.1)",
                  border: "1px solid rgba(201,168,76,0.2)",
                  color: "var(--accent-gold)",
                }}
              >
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
            {currentStock && (
              <button
                onClick={() => onAnalyze(currentStock)}
                className="rounded-lg px-2 py-1.5 text-[10px] font-medium transition-all"
                style={{
                  background: "rgba(201,168,76,0.06)",
                  border: "1px solid rgba(201,168,76,0.12)",
                  color: "var(--accent-gold)",
                }}
                title="重新分析"
              >
                <RotateCcw className="h-3 w-3" />
              </button>
            )}
            {onReset && (
              <button
                onClick={onReset}
                className="rounded-lg px-2 py-1.5 text-[10px] font-medium transition-all"
                style={{
                  background: "rgba(255,255,255,0.02)",
                  border: "1px solid rgba(255,255,255,0.06)",
                  color: "var(--text-secondary)",
                }}
                title="返回首頁"
              >
                <Search className="h-3 w-3" />
              </button>
            )}
            <Link
              href="/history"
              className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-[10px] font-medium transition-all hover:opacity-80"
              style={{
                color: "var(--text-muted)",
              }}
              title="歷史紀錄"
            >
              <Clock className="h-3 w-3" />
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
