"use client";

import { useState } from "react";
import type { AnalysisResult } from "@/lib/types";
import {
  FACTOR_LABELS, SIGNAL_COLORS, SIGNAL_LABELS,
  FACTOR_INSIGHT_GENERATORS,
} from "@/lib/constants";
import {
  Shield, ShieldOff, TrendingUp, TrendingDown,
  ChevronDown, ChevronRight, AlertTriangle,
  Zap, Eye, Crosshair, Clock, Lightbulb,
} from "lucide-react";

interface OverviewTabProps {
  result: AnalysisResult;
}

// ── Helpers ──────────────────────────────────────

function getVerdictText(signal: string, confidence: number, approved: boolean, regime: string): string {
  const highConf = confidence >= 0.6;
  const isBuy = signal.includes("buy");
  const isSell = signal.includes("sell");
  const isBear = regime === "bear";
  const isBull = regime === "bull";

  if (isBuy && highConf && approved) return "建議逢低分批佈局，基本面支撐穩健";
  if (isBuy && highConf && !approved) return "多方訊號明確但風控未通過，建議等待更好進場點";
  if (isBuy && !highConf) return "訊號偏多但信心不足，建議小額觀察";
  if (isSell && approved) return "風險升高，建議減碼保護獲利";
  if (isSell && !approved) return "空方壓力浮現，建議降低持倉比重";
  if (isBear) return "多空拉鋸，建議觀望不追高";
  if (isBull) return "盤勢偏多但尚無明確進場訊號，靜待突破";
  return "目前多空不明，建議持有觀望、靜待方向確認";
}

function formatPriceLevel(price: number | string | undefined, currentPrice: number): string {
  const p = Number(price);
  if (price == null || isNaN(p) || p === 0) return "\u2014";
  const pct = ((p - currentPrice) / currentPrice * 100).toFixed(1);
  const sign = Number(pct) >= 0 ? "+" : "";
  return `${p.toFixed(1)} (${sign}${pct}%)`;
}

function getActionLabel(action: string): string {
  return SIGNAL_LABELS[action] || action;
}

function getPositionLabel(size: number): string {
  if (size <= 0) return "暫不建議進場";
  return `投入資金的 ${(size * 100).toFixed(0)}%`;
}

// ── Main Component ───────────────────────────────

export function OverviewTab({ result }: OverviewTabProps) {
  const [rawExpanded, setRawExpanded] = useState(false);
  const narrative = result.narrative;
  const risk = result.risk_decision;
  const signal = result.signal || "hold";
  const signalColor = SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;
  const currentPrice = result.current_price;

  const verdictShort = narrative?.verdict_short || getVerdictText(signal, result.confidence, risk.approved, result.regime);
  const verdictFull = narrative?.verdict || "";
  const riskWarning = narrative?.risk_warning || "";
  const confidenceComment = narrative?.confidence_comment || "";

  // Factor insights grouped by sentiment
  const factorInsights = Object.entries(result.factor_details || {})
    .filter(([, d]) => d.available)
    .map(([key, d]) => {
      const gen = FACTOR_INSIGHT_GENERATORS[key];
      if (!gen) return null;
      const insight = gen(d.score);
      return { key, score: d.score, ...insight };
    })
    .filter(Boolean) as Array<{ key: string; score: number; text: string; sentiment: "bull" | "bear" | "neutral" }>;

  const bullInsights = factorInsights.filter((i) => i.sentiment === "bull");
  const bearInsights = factorInsights.filter((i) => i.sentiment === "bear");
  const neutralInsights = factorInsights.filter((i) => i.sentiment === "neutral");
  const sortedInsights = [...bullInsights, ...bearInsights, ...neutralInsights].slice(0, 6);

  return (
    <div className="space-y-3">
      {/* ── Section 1: VerdictHero ───────────────── */}
      <VerdictHero
        verdictShort={verdictShort}
        verdictFull={verdictFull}
        riskWarning={riskWarning}
        confidenceComment={confidenceComment}
        signalColor={signalColor}
        horizon={narrative?.outlook_horizon}
        isOpus={!!narrative?.verdict}
      />

      {/* ── Section 2: StrategyBox ───────────────── */}
      <StrategyBox
        risk={risk}
        narrative={narrative}
        currentPrice={currentPrice}
        signalColor={signalColor}
      />

      {/* ── Section 3: OutlookCard ───────────────── */}
      {narrative?.outlook && (
        <OutlookCard outlook={narrative.outlook} source={narrative.source} />
      )}

      {/* ── Section 4: BullBearBalance ───────────── */}
      <BullBearBalance narrative={narrative} />

      {/* ── Section 5: FactorInsights ────────────── */}
      {sortedInsights.length > 0 && (
        <FactorInsights insights={sortedInsights} />
      )}

      {/* ── Section 6: RawReasoning ──────────────── */}
      {result.reasoning && (
        <div className="glass-card p-4">
          <button
            className="flex items-center gap-2 w-full text-left group"
            onClick={() => setRawExpanded((v) => !v)}
          >
            <SectionLabel>原始分析紀錄</SectionLabel>
            {rawExpanded ? (
              <ChevronDown className="h-3 w-3 transition-transform" style={{ color: "var(--text-muted)" }} />
            ) : (
              <ChevronRight className="h-3 w-3 transition-transform" style={{ color: "var(--text-muted)" }} />
            )}
            {!rawExpanded && (
              <span className="text-[10px] ml-auto" style={{ color: "var(--text-muted)" }}>
                點擊展開
              </span>
            )}
          </button>
          <p
            className={`text-[12px] leading-[1.8] mt-2 transition-all ${rawExpanded ? "" : "line-clamp-2"}`}
            style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}
          >
            {result.reasoning}
          </p>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ───────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="text-[10px] tracking-[0.12em] font-semibold"
      style={{ color: "var(--accent-gold)", fontFamily: "'Noto Sans TC', sans-serif" }}
    >
      {children}
    </span>
  );
}

/** Section 1: 投資結論 — Opus 詳細版或 fallback 簡短版 */
function VerdictHero({ verdictShort, verdictFull, riskWarning, confidenceComment, signalColor, horizon, isOpus }: {
  verdictShort: string;
  verdictFull: string;
  riskWarning: string;
  confidenceComment: string;
  signalColor: string;
  horizon?: string;
  isOpus: boolean;
}) {
  return (
    <div
      className="glass-card overflow-hidden"
      style={{ borderLeft: `4px solid ${signalColor}` }}
    >
      <div
        className="p-5"
        style={{
          background: `linear-gradient(135deg, ${signalColor}06 0%, transparent 60%)`,
        }}
      >
        {/* Header with source badge */}
        <div className="flex items-center gap-2 mb-3">
          <SectionLabel>投資結論</SectionLabel>
          {isOpus && (
            <span
              className="rounded px-1.5 py-0.5 text-[8px] font-bold tracking-wider uppercase"
              style={{
                background: "rgba(168,85,247,0.08)",
                color: "rgba(168,85,247,0.8)",
                border: "1px solid rgba(168,85,247,0.15)",
              }}
            >
              OPUS
            </span>
          )}
        </div>

        {/* Short verdict — always shown, big text */}
        <p
          className="text-lg font-bold leading-relaxed"
          style={{
            color: signalColor,
            fontFamily: "'Noto Sans TC', sans-serif",
          }}
        >
          {verdictShort}
        </p>

        {/* Full verdict — Opus detailed paragraph */}
        {verdictFull && (
          <p
            className="mt-3 text-[13px] leading-[1.8]"
            style={{ color: "var(--text-primary)", fontFamily: "'Noto Sans TC', sans-serif" }}
          >
            {verdictFull}
          </p>
        )}

        {/* Risk warning */}
        {riskWarning && (
          <div
            className="mt-3 flex items-start gap-2 rounded-lg px-3 py-2"
            style={{
              background: "rgba(239,83,80,0.04)",
              border: "1px solid rgba(239,83,80,0.1)",
            }}
          >
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" style={{ color: "rgba(239,83,80,0.7)" }} />
            <span
              className="text-[11px] leading-relaxed"
              style={{ color: "rgba(239,83,80,0.8)", fontFamily: "'Noto Sans TC', sans-serif" }}
            >
              {riskWarning}
            </span>
          </div>
        )}

        {/* Footer: confidence comment + horizon */}
        <div className="mt-3 flex items-center gap-4 flex-wrap">
          {confidenceComment && (
            <span className="text-[11px]" style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}>
              {confidenceComment}
            </span>
          )}
          {horizon && (
            <span className="text-xs flex items-center gap-1.5 ml-auto" style={{ color: "var(--text-muted)" }}>
              <Clock className="h-3 w-3" />
              展望週期：{horizon}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** Section 2: 操作策略 3x2 grid */
function StrategyBox({ risk, narrative, currentPrice, signalColor }: {
  risk: AnalysisResult["risk_decision"];
  narrative: AnalysisResult["narrative"];
  currentPrice: number;
  signalColor: string;
}) {
  const support = narrative?.key_levels?.support;
  const resistance = narrative?.key_levels?.resistance;

  return (
    <div className="glass-card overflow-hidden">
      {/* Risk warning banner */}
      {!risk.approved && (
        <div
          className="flex items-center gap-2 px-4 py-2 text-[11px]"
          style={{
            background: "rgba(239,83,80,0.06)",
            borderBottom: "1px solid rgba(239,83,80,0.1)",
            color: "rgba(239,83,80,0.8)",
            fontFamily: "'Noto Sans TC', sans-serif",
          }}
        >
          <ShieldOff className="h-3.5 w-3.5 shrink-0" />
          風控系統建議謹慎操作
        </div>
      )}

      <div className="p-4">
        <SectionLabel>操作策略</SectionLabel>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2.5 mt-3">
          <StrategyItem
            icon={<Eye className="h-3.5 w-3.5" />}
            label="建議操作"
            value={getActionLabel(risk.action)}
            valueColor={signalColor}
          />
          <StrategyItem
            icon={<Crosshair className="h-3.5 w-3.5" />}
            label="建議倉位"
            value={getPositionLabel(risk.position_size)}
          />
          <StrategyItem
            icon={<TrendingUp className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.6)" }} />}
            label="參考支撐"
            value={formatPriceLevel(support, currentPrice)}
          />
          <StrategyItem
            icon={<TrendingDown className="h-3.5 w-3.5" style={{ color: "rgba(239,83,80,0.7)" }} />}
            label="停損價位"
            value={risk.stop_loss != null ? `跌破 $${risk.stop_loss.toFixed(1)} 出場 (${((risk.stop_loss - currentPrice) / currentPrice * 100).toFixed(1)}%)` : "\u2014"}
            accent="red"
          />
          <StrategyItem
            icon={<TrendingUp className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.7)" }} />}
            label="停利目標"
            value={risk.take_profit != null ? `目標 $${risk.take_profit.toFixed(1)} (+${((risk.take_profit - currentPrice) / currentPrice * 100).toFixed(1)}%)` : "\u2014"}
            accent="green"
          />
          <StrategyItem
            icon={<Clock className="h-3.5 w-3.5" />}
            label="操作週期"
            value={narrative?.outlook_horizon || "\u2014"}
          />
        </div>

        {/* LLM position suggestion */}
        {narrative?.position_suggestion && (
          <div
            className="mt-3 pt-3 text-[11px] leading-relaxed"
            style={{
              borderTop: "1px solid rgba(255,255,255,0.04)",
              color: "var(--text-secondary)",
              fontFamily: "'Noto Sans TC', sans-serif",
            }}
          >
            <span style={{ color: "var(--accent-gold)", marginRight: 6 }}>AI 建議：</span>
            {narrative.position_suggestion}
          </div>
        )}
      </div>
    </div>
  );
}

function StrategyItem({ icon, label, value, valueColor, accent }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueColor?: string;
  accent?: "red" | "green";
}) {
  const bgMap = {
    red: "rgba(239,83,80,0.04)",
    green: "rgba(34,197,94,0.04)",
  };
  const borderMap = {
    red: "rgba(239,83,80,0.1)",
    green: "rgba(34,197,94,0.1)",
  };

  return (
    <div
      className="rounded-lg p-3"
      style={{
        background: accent ? bgMap[accent] : "rgba(255,255,255,0.02)",
        border: `1px solid ${accent ? borderMap[accent] : "rgba(255,255,255,0.05)"}`,
      }}
    >
      <div className="flex items-center gap-1.5 mb-1.5" style={{ color: "var(--text-muted)" }}>
        {icon}
        <span className="text-[9px] tracking-wider font-medium">{label}</span>
      </div>
      <div
        className="text-[12px] font-bold"
        style={{
          color: valueColor || "var(--text-primary)",
          fontFamily: "'Noto Sans TC', sans-serif",
        }}
      >
        {value}
      </div>
    </div>
  );
}

/** Section 3: AI 分析觀點 */
function OutlookCard({ outlook, source }: { outlook: string; source: string }) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-center gap-2 mb-3">
        <SectionLabel>AI 分析觀點</SectionLabel>
        <span
          className="rounded px-1.5 py-0.5 text-[8px] font-bold tracking-wider uppercase"
          style={{
            background: source === "llm" ? "rgba(201,168,76,0.08)" : "rgba(255,255,255,0.04)",
            color: source === "llm" ? "var(--accent-gold)" : "var(--text-muted)",
            border: `1px solid ${source === "llm" ? "rgba(201,168,76,0.15)" : "rgba(255,255,255,0.06)"}`,
          }}
        >
          {source === "llm" ? "LLM" : "演算法"}
        </span>
      </div>
      <p
        className="text-[13px] leading-[1.8]"
        style={{ color: "var(--text-primary)", fontFamily: "'Noto Sans TC', sans-serif" }}
      >
        {outlook}
      </p>
    </div>
  );
}

/** Section 4: 多空對照 */
function BullBearBalance({ narrative }: { narrative: AnalysisResult["narrative"] }) {
  const hasDrivers = narrative?.key_drivers && narrative.key_drivers.length > 0;
  const hasRisks = narrative?.risks && narrative.risks.length > 0;
  const hasCatalysts = narrative?.catalysts && narrative.catalysts.length > 0;

  if (!hasDrivers && !hasRisks && !hasCatalysts) return null;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Bull side */}
        {hasDrivers && (
          <div
            className="glass-card p-4"
            style={{ background: "rgba(34,197,94,0.02)", borderColor: "rgba(34,197,94,0.08)" }}
          >
            <div className="flex items-center gap-1.5 mb-3">
              <Zap className="h-3.5 w-3.5" style={{ color: "rgba(34,197,94,0.7)" }} />
              <SectionLabel>看多理由</SectionLabel>
            </div>
            <ul className="space-y-2">
              {narrative.key_drivers.map((d, i) => (
                <li
                  key={i}
                  className="text-[12px] leading-snug flex items-start gap-2"
                  style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}
                >
                  <span className="mt-1 shrink-0 h-1.5 w-1.5 rounded-full" style={{ background: "rgba(34,197,94,0.6)" }} />
                  {d}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Bear side */}
        {hasRisks && (
          <div
            className="glass-card p-4"
            style={{ background: "rgba(239,83,80,0.02)", borderColor: "rgba(239,83,80,0.08)" }}
          >
            <div className="flex items-center gap-1.5 mb-3">
              <AlertTriangle className="h-3.5 w-3.5" style={{ color: "rgba(239,83,80,0.7)" }} />
              <SectionLabel>潛在風險</SectionLabel>
            </div>
            <ul className="space-y-2">
              {narrative.risks.map((r, i) => (
                <li
                  key={i}
                  className="text-[12px] leading-snug flex items-start gap-2"
                  style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}
                >
                  <span className="mt-1 shrink-0 h-1.5 w-1.5 rounded-full" style={{ background: "rgba(239,83,80,0.6)" }} />
                  {r}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Catalysts */}
      {hasCatalysts && (
        <div className="glass-card p-4">
          <div className="flex items-center gap-1.5 mb-3">
            <Lightbulb className="h-3.5 w-3.5" style={{ color: "var(--accent-gold-light)" }} />
            <SectionLabel>催化劑</SectionLabel>
          </div>
          <div className="flex flex-wrap gap-2">
            {narrative.catalysts.map((c, i) => (
              <span
                key={i}
                className="rounded-full px-3 py-1 text-[11px]"
                style={{
                  background: "rgba(201,168,76,0.06)",
                  color: "var(--text-secondary)",
                  border: "1px solid rgba(201,168,76,0.12)",
                  fontFamily: "'Noto Sans TC', sans-serif",
                }}
              >
                {c}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Section 5: 關鍵因子解讀 */
function FactorInsights({ insights }: {
  insights: Array<{ key: string; score: number; text: string; sentiment: "bull" | "bear" | "neutral" }>;
}) {
  const dotColors = {
    bull: "rgba(34,197,94,0.7)",
    bear: "rgba(239,83,80,0.7)",
    neutral: "rgba(201,168,76,0.6)",
  };

  return (
    <div className="glass-card p-4">
      <SectionLabel>關鍵因子解讀</SectionLabel>
      <div className="space-y-2.5 mt-3">
        {insights.map((insight) => (
          <div key={insight.key} className="flex items-start gap-2.5">
            <span
              className="mt-1.5 shrink-0 h-2 w-2 rounded-full"
              style={{ background: dotColors[insight.sentiment] }}
            />
            <div>
              <span
                className="text-[12px] leading-relaxed"
                style={{ color: "var(--text-primary)", fontFamily: "'Noto Sans TC', sans-serif" }}
              >
                {insight.text}
              </span>
              <span className="ml-2 text-[9px]" style={{ color: "var(--text-muted)" }}>
                {FACTOR_LABELS[insight.key] || insight.key}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
