"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST } from "@/lib/constants";
import type { PredictionRecord } from "@/lib/types";
import { Loader2, ChevronDown, ChevronRight, Shield, ShieldOff, Filter, RotateCcw } from "lucide-react";

export default function HistoryPage() {
  const router = useRouter();
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filterStock, setFilterStock] = useState<string>("all");

  const sid = filterStock === "all" ? undefined : filterStock;
  const { data: predictions = [], isLoading: loading } = useQuery({
    queryKey: ["predictions", "history", filterStock],
    queryFn: () => api.getPredictionHistory(sid, 50),
    staleTime: 2 * 60 * 1000,
  });

  const getSignalColor = (signal: string) =>
    SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || "#9E9E9E";

  const getSignalLabel = (signal: string) =>
    SIGNAL_LABELS[signal] || signal.toUpperCase();

  const handleReanalyze = (stockId: string) => {
    router.push(`/?stock=${stockId}`);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-56px)]">
      <div className="flex-1 overflow-auto">
        <div className="mx-auto max-w-4xl px-6 py-8 space-y-6">
          {/* Page header */}
          <div className="stagger-in">
            <div className="section-label mb-2">PREDICTION HISTORY</div>
            <h1 className="font-display text-2xl tracking-wide">
              分析<span style={{ color: "var(--accent-gold)" }}>紀錄</span>
            </h1>
          </div>

          {/* Filter bar */}
          <div className="flex items-center gap-4 card-reveal">
            <Filter className="h-4 w-4" style={{ color: "var(--text-secondary)" }} />
            <select
              value={filterStock}
              onChange={(e) => setFilterStock(e.target.value)}
              className="rounded-lg px-3 py-1.5 text-sm"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                outline: "none",
              }}
            >
              <option value="all">全部股票</option>
              {Object.entries(STOCK_LIST).map(([id, name]) => (
                <option key={id} value={id}>
                  {id} {name}
                </option>
              ))}
            </select>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              共 {predictions.length} 筆紀錄
            </span>
          </div>

          {/* Content */}
          {loading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-6 w-6 animate-spin" style={{ color: "var(--accent-gold)" }} />
            </div>
          ) : predictions.length === 0 ? (
            <div className="glass-card p-12 card-reveal text-center">
              <p style={{ color: "var(--text-secondary)" }}>尚無分析紀錄</p>
              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
                執行分析後，結果會自動保存在這裡
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {predictions.map((pred, idx) => {
                const isExpanded = expandedId === pred.id;
                const signalColor = getSignalColor(pred.agent_action || pred.signal);
                const hasActual = pred.actual_price !== null && pred.actual_price !== undefined;
                const pnl = hasActual && pred.predicted_price
                  ? ((pred.actual_price! - pred.predicted_price) / pred.predicted_price * 100)
                  : null;

                return (
                  <div
                    key={pred.id}
                    className="glass-card card-reveal overflow-hidden"
                    style={{ animationDelay: `${0.03 * Math.min(idx, 10)}s` }}
                  >
                    {/* Card header */}
                    <button
                      className="flex w-full items-center gap-4 px-5 py-4 text-left transition-all duration-200"
                      style={{ background: "transparent" }}
                      onClick={() => setExpandedId(isExpanded ? null : pred.id)}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.background = "rgba(201,168,76,0.03)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.background = "transparent";
                      }}
                    >
                      {/* Date */}
                      <span className="font-num text-xs shrink-0" style={{ color: "var(--text-muted)" }}>
                        {pred.prediction_date}
                      </span>

                      {/* Stock */}
                      <span className="font-num text-sm font-bold shrink-0" style={{ color: "var(--accent-gold)" }}>
                        {pred.stock_id}
                      </span>
                      <span className="text-xs shrink-0" style={{ color: "var(--text-secondary)" }}>
                        {STOCK_LIST[pred.stock_id] || pred.stock_name || ""}
                      </span>

                      {/* Signal badge */}
                      <span
                        className="inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-bold shrink-0"
                        style={{
                          background: `${signalColor}15`,
                          color: signalColor,
                          border: `1px solid ${signalColor}30`,
                        }}
                      >
                        {getSignalLabel(pred.agent_action || pred.signal)}
                      </span>

                      {/* Confidence */}
                      <span className="font-num text-xs shrink-0" style={{ color: "var(--text-primary)" }}>
                        {(pred.confidence * 100).toFixed(0)}%
                      </span>

                      {/* Price */}
                      <span className="font-num text-xs shrink-0" style={{ color: "var(--text-secondary)" }}>
                        {pred.predicted_price ? `$${pred.predicted_price.toFixed(0)}` : ""}
                        {hasActual && (
                          <span style={{
                            color: pnl !== null && pnl > 0 ? "#EF5350" : pnl !== null && pnl < 0 ? "#26A69A" : "var(--text-muted)"
                          }}>
                            {" → "}${pred.actual_price!.toFixed(0)}
                            {pnl !== null && (
                              <span className="ml-1 text-[10px]">
                                ({pnl > 0 ? "+" : ""}{pnl.toFixed(1)}%)
                              </span>
                            )}
                          </span>
                        )}
                      </span>

                      {/* Risk */}
                      <span className="shrink-0">
                        {pred.agent_approved ? (
                          <Shield className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.7)" }} />
                        ) : (
                          <ShieldOff className="h-3.5 w-3.5" style={{ color: "rgba(239,68,68,0.5)" }} />
                        )}
                      </span>

                      {/* Spacer + expand icon */}
                      <span className="flex-1 truncate text-xs" style={{ color: "var(--text-muted)" }}>
                        {pred.reasoning || ""}
                      </span>
                      {isExpanded ? (
                        <ChevronDown className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--text-muted)" }} />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 shrink-0" style={{ color: "var(--text-muted)" }} />
                      )}
                    </button>

                    {/* Expanded detail */}
                    {isExpanded && (
                      <div
                        className="px-5 pb-4 pt-2 space-y-3"
                        style={{ borderTop: "1px solid var(--border)" }}
                      >
                        {/* Reasoning */}
                        {pred.reasoning && (
                          <div>
                            <div className="text-[9px] tracking-wider uppercase mb-1" style={{ color: "var(--text-muted)" }}>
                              決策理由
                            </div>
                            <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                              {pred.reasoning}
                            </p>
                          </div>
                        )}

                        {/* Market snapshot */}
                        {pred.market_snapshot && (
                          <div>
                            <div className="text-[9px] tracking-wider uppercase mb-1" style={{ color: "var(--text-muted)" }}>
                              市場快照
                            </div>
                            <div className="grid grid-cols-3 gap-2">
                              {!!pred.market_snapshot.current_price && (
                                <MiniStat label="當前價" value={`$${Number(pred.market_snapshot.current_price).toFixed(1)}`} />
                              )}
                              {!!pred.market_snapshot.target_price && (
                                <MiniStat label="目標價" value={`$${Number(pred.market_snapshot.target_price).toFixed(1)}`} />
                              )}
                              {!!pred.market_snapshot.confidence && (
                                <MiniStat label="信心度" value={`${(Number(pred.market_snapshot.confidence) * 100).toFixed(0)}%`} />
                              )}
                            </div>
                          </div>
                        )}

                        {/* Re-analyze button */}
                        <div className="pt-2">
                          <button
                            onClick={() => handleReanalyze(pred.stock_id)}
                            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-200"
                            style={{
                              background: "rgba(201,168,76,0.08)",
                              border: "1px solid rgba(201,168,76,0.15)",
                              color: "var(--accent-gold)",
                            }}
                          >
                            <RotateCcw className="h-3 w-3" />
                            重新分析
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-lg p-2"
      style={{
        background: "rgba(255,255,255,0.015)",
        border: "1px solid rgba(255,255,255,0.04)",
      }}
    >
      <div className="text-[8px] tracking-wider uppercase" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}
