"use client";

import { useState, useCallback, useRef } from "react";
import { api } from "@/lib/api";
import type { PipelineEvent, PipelineStep, PipelineResult, AgentSubEvent } from "@/lib/types";

export type PipelineState = "idle" | "running" | "complete" | "error";

interface StepState {
  status: "pending" | "running" | "done" | "error" | "skipped";
  message: string;
  data?: Record<string, unknown>;
}

export function usePipeline() {
  const [state, setState] = useState<PipelineState>("idle");
  const [progress, setProgress] = useState(0);
  const [steps, setSteps] = useState<Record<PipelineStep, StepState>>({
    check_data: { status: "pending", message: "" },
    fetch_data: { status: "pending", message: "" },
    technical: { status: "pending", message: "" },
    sentiment: { status: "pending", message: "" },
    check_model: { status: "pending", message: "" },
    train_model: { status: "pending", message: "" },
    predict: { status: "pending", message: "" },
    agent: { status: "pending", message: "" },
    synthesize: { status: "pending", message: "" },
  });
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [agentSubEvents, setAgentSubEvents] = useState<AgentSubEvent[]>([]);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setState("idle");
    setProgress(0);
    setResult(null);
    setError(null);
    setAgentSubEvents([]);
    setSteps({
      check_data: { status: "pending", message: "" },
      fetch_data: { status: "pending", message: "" },
      technical: { status: "pending", message: "" },
      sentiment: { status: "pending", message: "" },
      check_model: { status: "pending", message: "" },
      train_model: { status: "pending", message: "" },
      predict: { status: "pending", message: "" },
      agent: { status: "pending", message: "" },
      synthesize: { status: "pending", message: "" },
    });
  }, []);

  const run = useCallback((stockId: string, opts?: { forceRetrain?: boolean }) => {
    // Abort previous run
    if (controllerRef.current) {
      controllerRef.current.abort();
    }

    reset();
    setState("running");

    const controller = api.runPipeline(
      stockId,
      (event: PipelineEvent) => {
        const step = event.step as PipelineStep;
        setProgress(event.progress);

        setSteps((prev) => ({
          ...prev,
          [step]: {
            status: event.status,
            message: event.message,
            data: event.data,
          },
        }));

        // Accumulate agent sub-events
        if (step === "agent" && event.data?.substep) {
          setAgentSubEvents((prev) => [...prev, event.data as unknown as AgentSubEvent]);
        }

        // Check for completion
        if (step === "synthesize" && event.status === "done" && event.data) {
          setState("complete");
          setResult(event.data as unknown as PipelineResult);
        }

        // Check for terminal error
        if (step === "synthesize" && event.status === "error") {
          setState("error");
          setError(event.message);
        }
      },
      { forceRetrain: opts?.forceRetrain },
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
    steps,
    result,
    error,
    agentSubEvents,
    run,
    abort,
    reset,
  };
}
