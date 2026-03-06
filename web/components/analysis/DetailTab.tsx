"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type { AnalysisResult, TechnicalResult } from "@/lib/types";
import { FACTOR_GROUPS, FACTOR_LABELS, FACTOR_INSIGHT_GENERATORS, REGIME_LABELS } from "@/lib/constants";
import { TechnicalSnapshot } from "@/components/dashboard/TechnicalSnapshot";
import { TechnicalIndicators } from "@/components/charts/TechnicalIndicators";
import { RadarChart } from "@/components/charts/RadarChart";
import {
  Loader2, ChevronDown, ChevronRight,
  BarChart3, Activity, Brain, MessageSquare, Shield, Layers,
} from "lucide-react";

const CandlestickChart = dynamic(
  () => import("@/components/charts/CandlestickChart").then((m) => m.CandlestickChart),
  { ssr: false, loading: () => <ChartLoading /> }
);

function ChartLoading() {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-5 w-5 animate-spin" style={{ color: "var(--accent-gold)" }} />
    </div>
  );
}

interface DetailTabProps {
  result: AnalysisResult;
  technicalData: TechnicalResult | null;
}

export function DetailTab({ result, technicalData }: DetailTabProps) {
  return (
    <div className="space-y-4">
      {/* ── Section 1: K-Line Chart ──────────────── */}
      <KLineSection technicalData={technicalData} />

      {/* ── Section 2: Technical Signals ─────────── */}
      <TechnicalSection technicalData={technicalData} />

      {/* ── Section 3: 20-Factor Detail ─────────── */}
      <FactorDetailSection result={result} />

      {/* ── Section 4: Each Factor Deep Dive ─────── */}
      <FactorDeepDive result={result} />

      {/* ── Section 5: Sentiment & News ──────────── */}
      <SentimentSection result={result} />

      {/* ── Section 6: Confidence & Regime ────────── */}
      <ConfidenceSection result={result} />

      {/* ── Section 7: Raw Data Dump ─────────────── */}
      <RawDataSection result={result} />
    </div>
  );
}

// ── Shared ───────────────────────────────────────

function SectionHeader({ icon: Icon, title, subtitle }: {
  icon: typeof BarChart3;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <Icon className="h-4 w-4" style={{ color: "var(--accent-gold)" }} />
      <span
        className="text-[11px] tracking-[0.1em] font-semibold"
        style={{ color: "var(--accent-gold)", fontFamily: "'Noto Sans TC', sans-serif" }}
      >
        {title}
      </span>
      {subtitle && (
        <span className="text-[9px] ml-auto" style={{ color: "var(--text-muted)" }}>
          {subtitle}
        </span>
      )}
    </div>
  );
}

// ── Section 1: K-Line ───────────────────────────

function KLineSection({ technicalData }: { technicalData: TechnicalResult | null }) {
  if (!technicalData) return null;
  const chartData = technicalData.chart_data || [];
  const candleData = chartData
    .filter((d) => d.open != null && d.high != null && d.low != null && d.close != null)
    .map((d) => ({
      date: d.date as string,
      open: d.open as number,
      high: d.high as number,
      low: d.low as number,
      close: d.close as number,
      volume: (d.volume as number) || 0,
    }));

  if (candleData.length === 0) return null;

  const overlays = {
    sma_5: chartData.map((d) => d.sma_5 as number).filter((v) => v != null),
    sma_20: chartData.map((d) => d.sma_20 as number).filter((v) => v != null),
    sma_60: chartData.map((d) => d.sma_60 as number).filter((v) => v != null),
  };

  return (
    <div className="glass-card p-5">
      <SectionHeader icon={BarChart3} title="K 線走勢圖" subtitle="120 日" />
      <CandlestickChart
        data={candleData}
        height={420}
        overlays={
          overlays.sma_5.length > 0 || overlays.sma_20.length > 0
            ? overlays
            : undefined
        }
      />
    </div>
  );
}

// ── Section 2: Technical Signals ────────────────

function TechnicalSection({ technicalData }: { technicalData: TechnicalResult | null }) {
  if (!technicalData) return null;
  const chartData = technicalData.chart_data || [];
  const indicators = technicalData.indicators || {};
  const signals = technicalData.signals;

  const radarData = signals
    ? Object.entries(signals)
        .filter(([k, v]) => k !== "summary" && v != null)
        .map(([k, v]) => {
          const s = v as { signal: string };
          const scoreMap: Record<string, number> = { buy: 90, weak_buy: 70, neutral: 50, weak_sell: 30, sell: 10 };
          return { name: k.toUpperCase(), value: scoreMap[s.signal] ?? 50, fullMark: 100 };
        })
    : [];

  return (
    <div className="glass-card p-5">
      <SectionHeader icon={Activity} title="技術指標訊號" />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {chartData.length > 0 && (
          <div>
            <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
              指標走勢
            </div>
            <TechnicalIndicators data={chartData as any} height={220} />
          </div>
        )}
        {radarData.length > 0 && (
          <div>
            <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
              訊號雷達
            </div>
            <RadarChart data={radarData} height={220} />
          </div>
        )}
        <div>
          <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
            訊號明細
          </div>
          <TechnicalSnapshot signals={signals} indicators={indicators} />
        </div>
      </div>
    </div>
  );
}

// ── Section 3: 20-Factor Bar Chart ──────────────

function FactorDetailSection({ result }: { result: AnalysisResult }) {
  const factors = result.factor_details || {};

  return (
    <div className="glass-card p-5">
      <SectionHeader
        icon={Layers}
        title="20 因子評分明細"
        subtitle={`體制: ${REGIME_LABELS[result.regime] || result.regime}`}
      />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {Object.entries(FACTOR_GROUPS).map(([groupName, factorKeys]) => (
          <div key={groupName}>
            <div
              className="text-[9px] tracking-wider uppercase font-medium mb-3"
              style={{ color: "var(--text-muted)" }}
            >
              {groupName}
            </div>
            <div className="space-y-2">
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
                  <div key={key} className="flex items-center gap-2.5">
                    <span
                      className="text-[11px] w-[72px] shrink-0 truncate"
                      style={{
                        color: available ? "var(--text-secondary)" : "var(--text-muted)",
                        opacity: available ? 1 : 0.5,
                        fontFamily: "'Noto Sans TC', sans-serif",
                      }}
                    >
                      {label}
                    </span>
                    <div className="flex-1 relative h-[6px] rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                      <div
                        className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
                        style={{
                          width: available ? `${score * 100}%` : "0%",
                          background: `linear-gradient(90deg, ${scoreColor}60, ${scoreColor})`,
                        }}
                      />
                      <div
                        className="absolute top-0 bottom-0 w-px"
                        style={{ left: "50%", background: "rgba(255,255,255,0.1)" }}
                      />
                    </div>
                    <span
                      className="font-num text-[11px] font-bold w-8 text-right shrink-0"
                      style={{ color: available ? scoreColor : "var(--text-muted)" }}
                    >
                      {available ? (score * 100).toFixed(0) : "—"}
                    </span>
                    <span
                      className="font-num text-[9px] w-8 text-right shrink-0"
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
      </div>
    </div>
  );
}

// ── Section 4: Factor Deep Dive (每個因子的子分量) ─

function FactorDeepDive({ result }: { result: AnalysisResult }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const factors = result.factor_details || {};

  // Sort by absolute distance from 0.5 (most opinionated first)
  const sortedFactors = Object.entries(factors)
    .filter(([, d]) => d.available)
    .sort((a, b) => Math.abs(b[1].score - 0.5) - Math.abs(a[1].score - 0.5));

  return (
    <div className="glass-card p-5">
      <SectionHeader icon={Brain} title="因子深度分析" subtitle="點擊展開子分量" />
      <div className="space-y-1">
        {sortedFactors.map(([key, detail]) => {
          const label = FACTOR_LABELS[key] || key;
          const score = detail.score;
          const scoreColor = score > 0.6 ? "#EF5350" : score < 0.4 ? "#26A69A" : "#FFC107";
          const isOpen = expanded === key;
          const gen = FACTOR_INSIGHT_GENERATORS[key];
          const insight = gen ? gen(score) : null;
          const hasComponents = detail.components && Object.keys(detail.components).length > 0;

          return (
            <div key={key}>
              <button
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors hover:bg-white/[0.02]"
                onClick={() => setExpanded(isOpen ? null : key)}
              >
                {/* Dot */}
                <span
                  className="h-2 w-2 rounded-full shrink-0"
                  style={{ background: scoreColor }}
                />
                {/* Label */}
                <span
                  className="text-[12px] text-left flex-1"
                  style={{ color: "var(--text-primary)", fontFamily: "'Noto Sans TC', sans-serif" }}
                >
                  {label}
                </span>
                {/* Insight short text */}
                {insight && (
                  <span
                    className="text-[10px] hidden md:inline"
                    style={{ color: "var(--text-muted)", fontFamily: "'Noto Sans TC', sans-serif" }}
                  >
                    {insight.text}
                  </span>
                )}
                {/* Score */}
                <span
                  className="font-num text-[12px] font-bold w-10 text-right shrink-0"
                  style={{ color: scoreColor }}
                >
                  {(score * 100).toFixed(0)}
                </span>
                {/* Chevron */}
                {hasComponents ? (
                  isOpen
                    ? <ChevronDown className="h-3 w-3 shrink-0" style={{ color: "var(--text-muted)" }} />
                    : <ChevronRight className="h-3 w-3 shrink-0" style={{ color: "var(--text-muted)" }} />
                ) : (
                  <div className="w-3 shrink-0" />
                )}
              </button>

              {/* Expanded: sub-components */}
              {isOpen && hasComponents && (
                <div
                  className="ml-7 mr-3 mb-2 rounded-lg p-3 space-y-1.5"
                  style={{
                    background: "rgba(255,255,255,0.015)",
                    border: "1px solid rgba(255,255,255,0.04)",
                  }}
                >
                  <div className="text-[9px] tracking-wider uppercase mb-2" style={{ color: "var(--text-muted)" }}>
                    子分量明細
                  </div>
                  {Object.entries(detail.components).map(([compKey, compVal]) => {
                    const val = typeof compVal === "number" ? compVal : 0;
                    const compColor = val > 0.6 ? "#EF5350" : val < 0.4 ? "#26A69A" : "#FFC107";
                    return (
                      <div key={compKey} className="flex items-center gap-2">
                        <span className="text-[10px] w-32 shrink-0 truncate" style={{ color: "var(--text-secondary)" }}>
                          {compKey}
                        </span>
                        <div className="flex-1 relative h-[4px] rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                          <div
                            className="absolute inset-y-0 left-0 rounded-full"
                            style={{ width: `${Math.min(val * 100, 100)}%`, background: compColor, opacity: 0.7 }}
                          />
                        </div>
                        <span className="font-num text-[10px] w-12 text-right shrink-0" style={{ color: compColor }}>
                          {val.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                  {/* Weight & freshness info */}
                  <div className="flex gap-4 mt-2 pt-2" style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                    <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                      權重: {(detail.weight * 100).toFixed(1)}%
                    </span>
                    <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                      新鮮度: {(detail.freshness * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Section 5: Sentiment ────────────────────────

function SentimentSection({ result }: { result: AnalysisResult }) {
  const newsSentiment = result.factor_details?.news_sentiment;
  const marginSentiment = result.factor_details?.margin_sentiment;
  const globalContext = result.factor_details?.global_context;

  const sentimentFactors = [
    { key: "news_sentiment", detail: newsSentiment },
    { key: "margin_sentiment", detail: marginSentiment },
    { key: "global_context", detail: globalContext },
  ].filter((f) => f.detail);

  if (sentimentFactors.length === 0) return null;

  return (
    <div className="glass-card p-5">
      <SectionHeader icon={MessageSquare} title="情緒與市場脈絡" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {sentimentFactors.map(({ key, detail }) => {
          if (!detail) return null;
          const scoreColor = detail.score > 0.6 ? "#EF5350" : detail.score < 0.4 ? "#26A69A" : "#FFC107";
          const gen = FACTOR_INSIGHT_GENERATORS[key];
          const insight = gen ? gen(detail.score) : null;

          return (
            <div
              key={key}
              className="rounded-lg p-4"
              style={{
                background: "rgba(255,255,255,0.015)",
                border: "1px solid rgba(255,255,255,0.04)",
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="text-[11px] font-medium"
                  style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}
                >
                  {FACTOR_LABELS[key] || key}
                </span>
                <span className="font-num text-[13px] font-bold" style={{ color: scoreColor }}>
                  {(detail.score * 100).toFixed(0)}
                </span>
              </div>
              {insight && (
                <p
                  className="text-[11px] leading-relaxed mb-2"
                  style={{ color: "var(--text-primary)", fontFamily: "'Noto Sans TC', sans-serif" }}
                >
                  {insight.text}
                </p>
              )}
              {/* Sub-components */}
              {detail.components && Object.keys(detail.components).length > 0 && (
                <div className="space-y-1 mt-2 pt-2" style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                  {Object.entries(detail.components).map(([ck, cv]) => (
                    <div key={ck} className="flex items-center justify-between">
                      <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>{ck}</span>
                      <span className="font-num text-[9px]" style={{ color: "var(--text-secondary)" }}>
                        {typeof cv === "number" ? cv.toFixed(3) : String(cv)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Section 6: Confidence & Regime ──────────────

function ConfidenceSection({ result }: { result: AnalysisResult }) {
  const breakdown = result.confidence_breakdown;
  const regimeColor = result.regime === "bull"
    ? "#EF5350"
    : result.regime === "bear"
    ? "#26A69A"
    : "#FFC107";

  return (
    <div className="glass-card p-5">
      <SectionHeader icon={Shield} title="信心度分析與市場體制" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confidence breakdown */}
        {breakdown && (
          <div>
            <div className="text-[9px] tracking-wider uppercase mb-3" style={{ color: "var(--text-muted)" }}>
              信心度組成 (最終: {(result.confidence * 100).toFixed(0)}%)
            </div>
            <div className="space-y-2.5">
              {[
                { label: "一致性", desc: "因子方向是否一致", value: breakdown.confidence_agreement, weight: "30%" },
                { label: "強度", desc: "因子偏離中性的程度", value: breakdown.confidence_strength, weight: "30%" },
                { label: "覆蓋率", desc: "可用因子佔比", value: breakdown.confidence_coverage, weight: "25%" },
                { label: "新鮮度", desc: "資料更新時效", value: breakdown.confidence_freshness, weight: "15%" },
              ].map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px]" style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}>
                        {item.label}
                      </span>
                      <span className="text-[8px]" style={{ color: "var(--text-muted)" }}>
                        {item.desc} ({item.weight})
                      </span>
                    </div>
                    <span className="font-num text-[11px] font-bold" style={{ color: "var(--text-primary)" }}>
                      {(item.value * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="relative h-[4px] rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.04)" }}>
                    <div
                      className="absolute inset-y-0 left-0 rounded-full"
                      style={{
                        width: `${item.value * 100}%`,
                        background: "linear-gradient(90deg, rgba(201,168,76,0.4), rgba(201,168,76,0.8))",
                      }}
                    />
                  </div>
                </div>
              ))}
              <div className="pt-2 mt-1" style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                <div className="flex items-center justify-between">
                  <span className="text-[11px]" style={{ color: "var(--accent-gold)", fontFamily: "'Noto Sans TC', sans-serif" }}>
                    風險折扣
                  </span>
                  <span className="font-num text-[11px] font-bold" style={{ color: "var(--accent-gold)" }}>
                    {(breakdown.risk_discount * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-[9px] mt-0.5" style={{ color: "var(--text-muted)" }}>
                  高波動/低量/融資激增/極端估值時折扣，下限 30%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Regime */}
        <div className="flex flex-col items-center justify-center gap-3">
          <div className="text-[9px] tracking-wider uppercase" style={{ color: "var(--text-muted)" }}>
            HMM 3-state 市場體制
          </div>
          <div
            className="rounded-xl px-8 py-4 text-xl font-bold"
            style={{
              background: `${regimeColor}10`,
              color: regimeColor,
              border: `1px solid ${regimeColor}20`,
            }}
          >
            {REGIME_LABELS[result.regime] || result.regime}
          </div>
          <p className="text-[10px] text-center max-w-xs" style={{ color: "var(--text-muted)" }}>
            HMM (Hidden Markov Model) 根據近期波動率、報酬率、成交量辨識市場狀態。
            {result.regime === "bull" && " 多頭體制下因子權重偏向動能與技術面。"}
            {result.regime === "bear" && " 空頭體制下因子權重偏向防禦與價值面，並減碼 50%。"}
            {result.regime === "sideways" && " 盤整體制下使用基準權重，不做額外調整。"}
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Section 7: Raw Data Dump ────────────────────

function RawDataSection({ result }: { result: AnalysisResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="glass-card p-4">
      <button
        className="flex items-center gap-2 w-full text-left"
        onClick={() => setExpanded((v) => !v)}
      >
        <span
          className="text-[10px] tracking-[0.12em] font-semibold"
          style={{ color: "var(--accent-gold)", fontFamily: "'Noto Sans TC', sans-serif" }}
        >
          原始 JSON 資料
        </span>
        {expanded
          ? <ChevronDown className="h-3 w-3" style={{ color: "var(--text-muted)" }} />
          : <ChevronRight className="h-3 w-3" style={{ color: "var(--text-muted)" }} />
        }
        {!expanded && (
          <span className="text-[10px] ml-auto" style={{ color: "var(--text-muted)" }}>
            點擊展開完整分析資料
          </span>
        )}
      </button>
      {expanded && (
        <pre
          className="mt-3 p-3 rounded-lg text-[10px] leading-relaxed overflow-auto max-h-[600px]"
          style={{
            background: "rgba(0,0,0,0.3)",
            color: "var(--text-secondary)",
            border: "1px solid rgba(255,255,255,0.04)",
            fontFamily: "'Space Mono', monospace",
          }}
        >
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
