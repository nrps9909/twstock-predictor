"use client";

import { useState, useCallback, useRef, useEffect } from "react";
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

// Module-level cache: survives page navigation without unmount losing state
let cachedState: PipelineState = "idle";
let cachedResult: AnalysisResult | null = null;
let cachedError: string | null = null;
let cachedProgress = 0;
let cachedPhases: Record<AnalysisPhase, PhaseState> = { ...INITIAL_PHASES };

export function usePipeline() {
  const [state, _setState] = useState<PipelineState>(cachedState);
  const [progress, _setProgress] = useState(cachedProgress);
  const [phases, _setPhases] = useState<Record<AnalysisPhase, PhaseState>>(cachedPhases);
  const [result, _setResult] = useState<AnalysisResult | null>(cachedResult);
  const [error, _setError] = useState<string | null>(cachedError);
  const controllerRef = useRef<AbortController | null>(null);

  // Wrap setters to also update module-level cache
  const setState = useCallback((v: PipelineState) => { cachedState = v; _setState(v); }, []);
  const setProgress = useCallback((v: number) => { cachedProgress = v; _setProgress(v); }, []);
  const setResult = useCallback((v: AnalysisResult | null) => { cachedResult = v; _setResult(v); }, []);
  const setError = useCallback((v: string | null) => { cachedError = v; _setError(v); }, []);
  const setPhases = useCallback((updater: (prev: Record<AnalysisPhase, PhaseState>) => Record<AnalysisPhase, PhaseState>) => {
    _setPhases((prev) => {
      const next = updater(prev);
      cachedPhases = next;
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    setState("idle");
    setProgress(0);
    setResult(null);
    setError(null);
    const fresh = { ...INITIAL_PHASES };
    cachedPhases = fresh;
    _setPhases(fresh);
  }, [setState, setProgress, setResult, setError]);

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
  }, [reset, setState, setProgress, setPhases, setResult, setError]);

  const abort = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.abort();
      controllerRef.current = null;
    }
    setState("idle");
  }, [setState]);

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
