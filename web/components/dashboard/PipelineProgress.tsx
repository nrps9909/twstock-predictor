"use client";

import { useEffect, useState } from "react";
import { PIPELINE_STEPS, type PipelineStep } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Check, Loader2, X, SkipForward } from "lucide-react";

interface StepState {
  status: "pending" | "running" | "done" | "error" | "skipped";
  message: string;
}

interface PipelineProgressProps {
  steps: Record<PipelineStep, StepState>;
  progress: number;
}

export function PipelineProgress({ steps, progress }: PipelineProgressProps) {
  // Find the currently running step for the message display
  const runningStep = PIPELINE_STEPS.find(({ key }) => steps[key].status === "running");
  const runningMessage = runningStep ? steps[runningStep.key].message : "";
  const runningKey = runningStep?.key ?? null;

  // Elapsed timer — resets when the running step changes
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    setElapsed(0);
    if (!runningKey) return;

    const id = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [runningKey]);

  return (
    <div className="glass-card-static p-6">
      {/* Progress bar */}
      <div className="relative mb-6 h-[3px] rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.03)" }}>
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${progress}%`,
            background: "linear-gradient(90deg, #B8962F, #C9A84C, #E2C87A)",
            boxShadow: "0 0 12px rgba(201,168,76,0.4)",
          }}
        />
        {/* Sweep highlight */}
        <div
          className="absolute inset-y-0 w-24 rounded-full"
          style={{
            left: `${Math.max(0, progress - 8)}%`,
            background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
            opacity: progress > 0 && progress < 100 ? 1 : 0,
            transition: "left 0.7s ease-out",
          }}
        />
      </div>

      {/* Running message + elapsed timer */}
      {runningMessage && (
        <div className="text-center mb-5 flex items-center justify-center gap-2">
          <span className="font-num text-xs tracking-wider" style={{ color: "var(--accent-gold)" }}>
            {runningMessage}
          </span>
          {elapsed > 0 && (
            <span
              className="font-num text-[10px] tabular-nums"
              style={{ color: "var(--text-muted)" }}
            >
              {elapsed}s
            </span>
          )}
        </div>
      )}

      {/* Steps */}
      <div className="flex items-start justify-between">
        {PIPELINE_STEPS.map(({ key, label }, i) => {
          const step = steps[key];
          const isDone = step.status === "done";
          const isRunning = step.status === "running";
          const isError = step.status === "error";
          const isSkipped = step.status === "skipped";

          return (
            <div key={key} className="flex items-start flex-1">
              <div className="flex flex-col items-center flex-1">
                {/* Circle */}
                <div
                  className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-full text-[10px] font-medium transition-all duration-500",
                    isDone && "step-done",
                    isRunning && "step-running"
                  )}
                  style={{
                    background: isDone
                      ? "rgba(34,197,94,0.12)"
                      : isRunning
                      ? "rgba(201,168,76,0.12)"
                      : isError
                      ? "rgba(232,82,74,0.12)"
                      : "rgba(255,255,255,0.02)",
                    border: `1px solid ${
                      isDone
                        ? "rgba(34,197,94,0.25)"
                        : isRunning
                        ? "rgba(201,168,76,0.3)"
                        : isError
                        ? "rgba(232,82,74,0.25)"
                        : "rgba(255,255,255,0.04)"
                    }`,
                    color: isDone
                      ? "rgb(34,197,94)"
                      : isRunning
                      ? "var(--accent-gold)"
                      : isError
                      ? "rgb(232,82,74)"
                      : "var(--text-muted)",
                  }}
                >
                  {isDone && <Check className="h-3.5 w-3.5" />}
                  {isRunning && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                  {isError && <X className="h-3.5 w-3.5" />}
                  {isSkipped && <SkipForward className="h-3 w-3" />}
                  {step.status === "pending" && <span className="font-num">{i + 1}</span>}
                </div>

                {/* Label */}
                <span
                  className="mt-2 text-[9px] text-center whitespace-nowrap tracking-wider transition-colors duration-300"
                  style={{
                    color: isRunning
                      ? "var(--accent-gold)"
                      : isDone
                      ? "rgba(34,197,94,0.7)"
                      : "var(--text-muted)",
                    fontFamily: "'Space Mono', monospace",
                  }}
                >
                  {label}
                </span>
              </div>

              {/* Connector */}
              {i < PIPELINE_STEPS.length - 1 && (
                <div
                  className="flex-1 min-w-[8px] mt-4"
                  style={{
                    height: "1px",
                    background: isDone
                      ? "linear-gradient(90deg, rgba(34,197,94,0.3), rgba(34,197,94,0.1))"
                      : "rgba(255,255,255,0.03)",
                    transition: "background 0.5s ease",
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
