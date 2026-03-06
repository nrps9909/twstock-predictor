"use client";

import { useEffect, useRef } from "react";
import { PIPELINE_PHASES } from "@/lib/constants";
import type { AnalysisPhase } from "@/lib/types";
import type { PhaseState } from "@/hooks/usePipeline";
import {
  Check, X, Loader2, Database, Cpu, BarChart3,
  FileText, Shield, Save, ChevronRight,
} from "lucide-react";

const PHASE_ICONS: Record<AnalysisPhase, typeof Database> = {
  data_collection: Database,
  feature_extraction: Cpu,
  scoring: BarChart3,
  narrative: FileText,
  risk_control: Shield,
  finalize: Save,
};

const PHASE_DESC: Record<AnalysisPhase, string> = {
  data_collection: "並行抓取 OHLCV、三大法人、月營收、全球指數、宏觀、情緒、基本面、P/E",
  feature_extraction: "HMM 體制辨識 + ML 集成預測 + LLM 情緒萃取",
  scoring: "20 因子加權 × HMM regime 調整 → 信號 + 信心度",
  narrative: "Opus 生成完整報告: 分析展望 + 投資結論 + 風險評估",
  risk_control: "ATR Stop + Circuit Breaker + Meta-Label + 體制減碼",
  finalize: "寫入資料庫 + 警報系統檢查",
};

interface PipelineDetailProps {
  phases: Record<AnalysisPhase, PhaseState>;
}

function PhaseBlock({ phaseKey, phase }: { phaseKey: AnalysisPhase; phase: PhaseState }) {
  const Icon = PHASE_ICONS[phaseKey];
  const phaseMeta = PIPELINE_PHASES.find((p) => p.key === phaseKey);
  const isDone = phase.status === "done";
  const isRunning = phase.status === "running";
  const isError = phase.status === "error";
  const isPending = phase.status === "pending";
  const subSteps = (phase.data?.sub_steps as string[]) || [];

  if (isPending) return null;

  const accent = isDone ? "rgb(34,197,94)" : isRunning ? "var(--accent-gold)" : isError ? "rgb(232,82,74)" : "var(--text-muted)";

  return (
    <div
      className="px-4 py-2.5 transition-all duration-300"
      style={{
        borderLeft: `2px solid ${isDone ? "rgba(34,197,94,0.25)" : isRunning ? "rgba(201,168,76,0.35)" : isError ? "rgba(232,82,74,0.25)" : "rgba(255,255,255,0.03)"}`,
        background: isRunning ? "rgba(201,168,76,0.02)" : "transparent",
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        <div
          className="flex h-5 w-5 shrink-0 items-center justify-center rounded"
          style={{
            background: isDone ? "rgba(34,197,94,0.08)" : isRunning ? "rgba(201,168,76,0.08)" : isError ? "rgba(232,82,74,0.08)" : "rgba(255,255,255,0.02)",
            color: accent,
          }}
        >
          {isRunning ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : isDone ? <Check className="h-2.5 w-2.5" /> : isError ? <X className="h-2.5 w-2.5" /> : <Icon className="h-2.5 w-2.5" />}
        </div>

        <span className="text-[11px] font-medium" style={{ color: accent, fontFamily: "'Noto Sans TC', sans-serif" }}>
          {phaseMeta?.label}
        </span>

        {isRunning && (
          <span className="text-[9px] ml-auto font-num animate-pulse" style={{ color: "var(--accent-gold-light)" }}>
            {phase.message}
          </span>
        )}
      </div>

      {/* Running: short description */}
      {isRunning && (
        <p className="mt-1 text-[10px] leading-relaxed pl-7 opacity-50" style={{ color: "var(--text-secondary)" }}>
          {PHASE_DESC[phaseKey]}
        </p>
      )}

      {/* Done: sub-steps */}
      {isDone && subSteps.length > 0 && (
        <div className="mt-1 pl-7 space-y-px">
          {subSteps.map((step, i) => (
            <div key={i} className="flex items-start gap-1.5 py-px">
              <ChevronRight className="h-2.5 w-2.5 shrink-0 mt-[2px]" style={{ color: "rgba(34,197,94,0.35)" }} />
              <span className="text-[10px] leading-snug" style={{ color: "var(--text-secondary)", fontFamily: "'Noto Sans TC', sans-serif" }}>
                {step}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function PipelineDetail({ phases }: PipelineDetailProps) {
  const hasActivity = PIPELINE_PHASES.some(({ key }) => phases[key].status !== "pending");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [phases]);

  if (!hasActivity) return null;

  return (
    <>
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-2 shrink-0" style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
        <div className="h-1 w-1 rounded-full" style={{ background: "var(--accent-gold)", boxShadow: "0 0 4px var(--accent-gold)" }} />
        <span className="text-[9px] uppercase tracking-[0.12em] font-medium" style={{ color: "var(--text-secondary)", fontFamily: "'Space Mono', monospace" }}>
          Analysis Log
        </span>
      </div>

      {/* Scrollable log */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-1">
        {PIPELINE_PHASES.map(({ key }) => (
          <PhaseBlock key={key} phaseKey={key} phase={phases[key]} />
        ))}
      </div>
    </>
  );
}
