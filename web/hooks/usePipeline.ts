"use client";

import { useState, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type { PipelineEvent, AnalysisPhase, AnalysisResult } from "@/lib/types";

export type PipelineState = "idle" | "running" | "complete" | "error";

export interface PhaseState {
  status: "pending" | "running" | "done" | "error" | "skipped";
  message: string;
  data?: Record<string, unknown>;
}

const INITIAL_PHASES: Record<AnalysisPhase, PhaseState> = {
  data_collection: { status: "pending", message: "" },
  feature_extraction: { status: "pending", message: "" },
  scoring: { status: "pending", message: "" },
  narrative: { status: "pending", message: "" },
  risk_control: { status: "pending", message: "" },
  finalize: { status: "pending", message: "" },
};

export function usePipeline() {
  const [state, setState] = useState<PipelineState>("idle");
  const [progress, setProgress] = useState(0);
  const [phases, setPhases] = useState<Record<AnalysisPhase, PhaseState>>({ ...INITIAL_PHASES });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setState("idle");
    setProgress(0);
    setResult(null);
    setError(null);
    setPhases({ ...INITIAL_PHASES });
  }, []);

  const run = useCallback((stockId: string) => {
    // Abort previous run
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    reset();
    setState("running");

    const controller = api.runPipeline(
      stockId,
      (event: PipelineEvent) => {
        setProgress(event.progress);

        // Update phase state (skip "complete" pseudo-phase)
        if (event.phase !== "complete") {
          const phase = event.phase as AnalysisPhase;
          setPhases((prev) => ({
            ...prev,
            [phase]: {
              status: event.status,
              message: event.message,
              data: event.data,
            },
          }));
        }

        // Check for completion
        if (event.phase === "complete" && event.status === "done" && event.data) {
          setState("complete");
          setResult(event.data as unknown as AnalysisResult);
        }

        // Check for terminal error
        if (event.phase === "complete" && event.status === "error") {
          setState("error");
          setError(event.data?.error as string || event.message);
        }
      },
    );

    controllerRef.current = controller;
  }, [reset]);

  const abort = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    setState("idle");
  }, []);

  return {
    state,
    progress,
    phases,
    result,
    error,
    run,
    abort,
    reset,
  };
}
