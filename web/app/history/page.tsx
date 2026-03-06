"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST } from "@/lib/constants";
import type { PredictionRecord, AnalysisResult } from "@/lib/types";
import { ResultTabs } from "@/components/analysis/ResultTabs";
import { Loader2, ChevronDown, ChevronRight, Shield, ShieldOff, Filter, RotateCcw, BarChart3 } from "lucide-react";
import Link from "next/link";

const SIGNAL_FILTERS = [
  { key: "all", label: "全部" },
  { key: "buy", label: "買進" },
  { key: "hold", label: "持有" },
  { key: "sell", label: "賣出" },
] as const;

export default function HistoryPage() {
  const router = useRouter();
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filterStock, setFilterStock] = useState<string>("all");
  const [filterSignal, setFilterSignal] = useState<string>("all");

  const sid = filterStock === "all" ? undefined : filterStock;
  const { data: predictions = [], isLoading: loading } = useQuery({
    queryKey: ["predictions", "history", filterStock],
    queryFn: () => api.getPredictionHistory(sid, 50),
    staleTime: 2 * 60 * 1000,
  });

  // Client-side signal filter
  const filteredPredictions = useMemo(() => {
    if (filterSignal === "all") return predictions;
    return predictions.filter((p) => {
      const action = p.agent_action || p.signal;
      return action.includes(filterSignal);
    });
  }, [predictions, filterSignal]);

  // Stats from predictions
  const stats = useMemo(() => {
    const total = predictions.length;
    const buyCount = predictions.filter((p) => (p.agent_action || p.signal).includes("buy")).length;
    const holdCount = predictions.filter((p) => (p.agent_action || p.signal).includes("hold")).length;
    const sellCount = predictions.filter((p) => (p.agent_action || p.signal).includes("sell")).length;
    const avgConf = total > 0 ? predictions.reduce((s, p) => s + p.confidence, 0) / total : 0;
    return { total, buyCount, holdCount, sellCount, avgConf };
  }, [predictions]);

  const getSignalColor = (signal: string) =>
    SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || "#9E9E9E";

  const getSignalLabel = (signal: string) =>
    SIGNAL_LABELS[signal] || signal.toUpperCase();

  const handleReanalyze = (stockId: string) => {
    router.push(`/?stock=${stockId}&t=${Date.now()}`);
  };

  const handleToggle = (id: number) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  return (
    <div className="flex flex-col h-screen">
      <div className="flex-1 overflow-auto">
        <div className="mx-auto max-w-[1440px] px-6 xl:px-10 py-8 space-y-5">
          {/* Page header */}
          <div className="stagger-in flex items-end justify-between">
            <div>
              <div className="section-label mb-2">PREDICTION HISTORY</div>
              <h1 className="font-display text-2xl tracking-wide">
                分析<span style={{ color: "var(--accent-gold)" }}>紀錄</span>
              </h1>
            </div>
            <Link
              href="/"
              className="flex items-center gap-1.5 text-[11px] font-medium transition-all hover:opacity-80"
              style={{ color: "var(--text-muted)" }}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              分析
            </Link>
          </div>

          {/* Stats summary */}
          {!loading && predictions.length > 0 && (
            <div className="grid grid-cols-5 gap-3 card-reveal">
              <MiniStat label="總分析" value={String(stats.total)} />
              <MiniStat label="買進" value={String(stats.buyCount)} color="var(--signal-buy)" />
              <MiniStat label="持有" value={String(stats.holdCount)} color="#FFC107" />
              <MiniStat label="賣出" value={String(stats.sellCount)} color="var(--signal-sell)" />
              <MiniStat label="平均信心" value={`${(stats.avgConf * 100).toFixed(0)}%`} />
            </div>
          )}

          {/* Filter bar */}
          <div className="flex items-center gap-4 card-reveal flex-wrap">
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

            {/* Signal filter pills */}
            <div className="flex items-center gap-1.5">
              {SIGNAL_FILTERS.map((f) => (
                <button
                  key={f.key}
                  onClick={() => setFilterSignal(f.key)}
                  className="rounded-lg px-3 py-1 text-[11px] font-medium transition-all duration-200"
                  style={{
                    background: filterSignal === f.key ? "rgba(201,168,76,0.1)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${filterSignal === f.key ? "rgba(201,168,76,0.25)" : "rgba(255,255,255,0.06)"}`,
                    color: filterSignal === f.key ? "var(--accent-gold)" : "var(--text-secondary)",
                  }}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <span className="text-xs ml-auto" style={{ color: "var(--text-muted)" }}>
              共 {filteredPredictions.length} 筆紀錄
            </span>
          </div>

          {/* Content */}
          {loading ? (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-6 w-6 animate-spin" style={{ color: "var(--accent-gold)" }} />
            </div>
          ) : filteredPredictions.length === 0 ? (
            <div className="glass-card p-12 card-reveal text-center">
              <p style={{ color: "var(--text-secondary)" }}>尚無分析紀錄</p>
              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
                執行分析後，結果會自動保存在這裡
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {/* Header row */}
              <div
                className="flex items-center gap-4 px-5 py-2 text-[9px] tracking-wider uppercase"
                style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
              >
                <span className="w-20 shrink-0">日期</span>
                <span className="w-12 shrink-0">代碼</span>
                <span className="w-16 shrink-0">名稱</span>
                <span className="w-14 shrink-0">訊號</span>
                <span className="w-10 shrink-0">信心</span>
                <span className="w-24 shrink-0">價格</span>
                <span className="w-6 shrink-0">風控</span>
                <span className="flex-1">摘要</span>
              </div>

              {filteredPredictions.map((pred, idx) => {
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
                    style={{ animationDelay: `${0.02 * Math.min(idx, 10)}s` }}
                  >
                    {/* Card header */}
                    <button
                      className="flex w-full items-center gap-4 px-5 py-3 text-left transition-all duration-200"
                      style={{ background: "transparent" }}
                      onClick={() => handleToggle(pred.id)}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.background = "rgba(201,168,76,0.03)";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.background = "transparent";
                      }}
                    >
                      {/* Date */}
                      <span className="font-num text-xs w-20 shrink-0" style={{ color: "var(--text-muted)" }}>
                        {pred.prediction_date}
                      </span>

                      {/* Stock */}
                      <span className="font-num text-sm font-bold w-12 shrink-0" style={{ color: "var(--accent-gold)" }}>
                        {pred.stock_id}
                      </span>
                      <span className="text-xs w-16 shrink-0 truncate" style={{ color: "var(--text-secondary)" }}>
                        {STOCK_LIST[pred.stock_id] || pred.stock_name || ""}
                      </span>

                      {/* Signal badge */}
                      <span
                        className="inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-bold w-14 shrink-0 justify-center"
                        style={{
                          background: `${signalColor}15`,
                          color: signalColor,
                          border: `1px solid ${signalColor}30`,
                        }}
                      >
                        {getSignalLabel(pred.agent_action || pred.signal)}
                      </span>

                      {/* Confidence */}
                      <span className="font-num text-xs w-10 shrink-0" style={{ color: "var(--text-primary)" }}>
                        {(pred.confidence * 100).toFixed(0)}%
                      </span>

                      {/* Price */}
                      <span className="font-num text-xs w-24 shrink-0" style={{ color: "var(--text-secondary)" }}>
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
                      <span className="w-6 shrink-0">
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

                    {/* Expanded full analysis */}
                    {isExpanded && (
                      <HistoryDetail
                        pred={pred}
                        onReanalyze={handleReanalyze}
                      />
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

function HistoryDetail({
  pred,
  onReanalyze,
}: {
  pred: PredictionRecord;
  onReanalyze: (stockId: string) => void;
}) {
  const { data: detail, isLoading } = useQuery({
    queryKey: ["analysis-detail", pred.stock_id, pred.prediction_date],
    queryFn: async () => {
      const res = await api.getAnalysisDetail(pred.stock_id, pred.prediction_date);
      if ("status" in res && res.status === "not_found") return null;
      return res as AnalysisResult;
    },
    staleTime: 10 * 60 * 1000,
  });

  const hasFullDetail = detail && ("factor_details" in detail || "narrative" in detail);

  return (
    <div
      className="px-5 pb-5 pt-3 space-y-4"
      style={{ borderTop: "1px solid var(--border)" }}
    >
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-5 w-5 animate-spin" style={{ color: "var(--accent-gold)" }} />
          <span className="ml-2 text-xs" style={{ color: "var(--text-muted)" }}>載入完整分析...</span>
        </div>
      ) : hasFullDetail ? (
        /* Full analysis view (same as analysis page) */
        <div className="space-y-4">
          <ResultTabs result={detail} technicalData={null} view="overview" />
        </div>
      ) : (
        /* Fallback: simple view from prediction record */
        <div className="space-y-3">
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
          {pred.market_snapshot && (
            <div>
              <div className="text-[9px] tracking-wider uppercase mb-1" style={{ color: "var(--text-muted)" }}>
                市場快照
              </div>
              <div className="grid grid-cols-3 gap-2">
                {!!pred.market_snapshot.current_price && (
                  <DetailMiniStat label="當前價" value={`$${Number(pred.market_snapshot.current_price).toFixed(1)}`} />
                )}
                {!!pred.market_snapshot.target_price && (
                  <DetailMiniStat label="目標價" value={`$${Number(pred.market_snapshot.target_price).toFixed(1)}`} />
                )}
                {!!pred.market_snapshot.confidence && (
                  <DetailMiniStat label="信心度" value={`${(Number(pred.market_snapshot.confidence) * 100).toFixed(0)}%`} />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Re-analyze button */}
      <div className="pt-2">
        <button
          onClick={() => onReanalyze(pred.stock_id)}
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
      <div className="text-[8px] tracking-wider uppercase" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="font-num text-lg font-bold" style={{ color: color || "var(--text-primary)" }}>
        {value}
      </div>
    </div>
  );
}

function DetailMiniStat({ label, value }: { label: string; value: string }) {
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
