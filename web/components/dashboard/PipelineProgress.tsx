"use client";

import { useEffect, useState } from "react";
import { PIPELINE_PHASES } from "@/lib/constants";
import type { AnalysisPhase } from "@/lib/types";
import type { PhaseState } from "@/hooks/usePipeline";
import { cn } from "@/lib/utils";
import { Check, Loader2, X } from "lucide-react";

interface PipelineProgressProps {
  phases: Record<AnalysisPhase, PhaseState>;
  progress: number;
}

export function PipelineProgress({ phases, progress }: PipelineProgressProps) {
  const runningPhase = PIPELINE_PHASES.find(({ key }) => phases[key].status === "running");
  const runningKey = runningPhase?.key ?? null;

  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    setElapsed(0);
    if (!runningKey) return;
    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [runningKey]);

  return (
    <div className="flex flex-col gap-1.5">
      {/* Progress bar */}
      <div className="relative h-[2px] rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.03)" }}>
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${progress}%`,
            background: "linear-gradient(90deg, #B8962F, #C9A84C, #E2C87A)",
            boxShadow: "0 0 8px rgba(201,168,76,0.3)",
          }}
        />
      </div>

      {/* Vertical steps */}
      <div className="flex flex-col">
        {PIPELINE_PHASES.map(({ key, label }, i) => {
          const phase = phases[key];
          const isDone = phase.status === "done";
          const isRunning = phase.status === "running";
          const isError = phase.status === "error";
          const isPending = phase.status === "pending";

          return (
            <div key={key} className="flex items-center gap-2 relative">
              {i < PIPELINE_PHASES.length - 1 && (
                <div
                  className="absolute left-[8px] top-[20px] w-[1px] h-[calc(100%-4px)]"
                  style={{
                    background: isDone ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.04)",
                    transition: "background 0.5s ease",
                  }}
                />
              )}

              <div
                className={cn(
                  "flex h-[16px] w-[16px] shrink-0 items-center justify-center rounded-full transition-all duration-500 z-10",
                  isRunning && "step-running",
                )}
                style={{
                  background: isDone
                    ? "rgba(34,197,94,0.15)"
                    : isRunning
                    ? "rgba(201,168,76,0.15)"
                    : isError
                    ? "rgba(232,82,74,0.15)"
                    : "rgba(255,255,255,0.02)",
                  border: `1px solid ${
                    isDone ? "rgba(34,197,94,0.3)"
                    : isRunning ? "rgba(201,168,76,0.35)"
                    : isError ? "rgba(232,82,74,0.3)"
                    : "rgba(255,255,255,0.05)"
                  }`,
                  color: isDone ? "rgb(34,197,94)"
                    : isRunning ? "var(--accent-gold)"
                    : isError ? "rgb(232,82,74)"
                    : "var(--text-muted)",
                }}
              >
                {isDone && <Check className="h-2 w-2" />}
                {isRunning && <Loader2 className="h-2 w-2 animate-spin" />}
                {isError && <X className="h-2 w-2" />}
                {isPending && <span className="text-[6px] font-num">{i + 1}</span>}
              </div>

              <div className="flex items-center gap-1.5 py-[4px]">
                <span
                  className="text-[9px] tracking-wide transition-colors duration-300"
                  style={{
                    color: isRunning ? "var(--accent-gold)"
                      : isDone ? "rgba(34,197,94,0.6)"
                      : isError ? "rgb(232,82,74)"
                      : "var(--text-muted)",
                    fontFamily: "'Space Mono', monospace",
                    fontWeight: isRunning ? 600 : 400,
                  }}
                >
                  {label}
                </span>
                {isRunning && elapsed > 0 && (
                  <span className="font-num text-[7px] tabular-nums" style={{ color: "rgba(255,255,255,0.15)" }}>
                    {elapsed}s
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
