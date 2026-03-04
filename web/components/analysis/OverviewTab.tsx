"use client";

import type { PipelineResult } from "@/lib/types";
import { PredictionSummary } from "@/components/dashboard/PredictionSummary";
import { TechnicalSnapshot } from "@/components/dashboard/TechnicalSnapshot";
import { SentimentSnapshot } from "@/components/dashboard/SentimentSnapshot";

interface OverviewTabProps {
  result: PipelineResult;
}

export function OverviewTab({ result }: OverviewTabProps) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <PredictionSummary result={result} />
        <TechnicalSnapshot
          signals={result.technical?.signals}
          indicators={result.technical?.indicators}
        />
        <SentimentSnapshot sentiment={result.sentiment} />
      </div>

      {/* Reasoning */}
      {result.reasoning && (
        <div
          className="glass-card p-5"
        >
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-3"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            ANALYSIS SUMMARY
          </div>
          <p className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
            {result.reasoning}
          </p>
        </div>
      )}
    </div>
  );
}
