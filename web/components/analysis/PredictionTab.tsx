"use client";

import type { PipelineResult } from "@/lib/types";
import { PredictionChart } from "@/components/charts/PredictionChart";
import { formatPrice } from "@/lib/utils";

interface PredictionTabProps {
  result: PipelineResult;
}

export function PredictionTab({ result }: PredictionTabProps) {
  const prediction = result.prediction;

  if (!prediction) {
    return (
      <div className="glass-card p-12 text-center">
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>
          ML 預測資料不可用
        </span>
      </div>
    );
  }

  const signal = prediction.signal || "hold";
  const signalColor = signal.includes("buy") ? "var(--signal-buy)" : signal.includes("sell") ? "var(--signal-sell)" : "var(--signal-hold)";

  return (
    <div className="space-y-6">
      {/* Prediction chart */}
      {prediction.predicted_prices?.length > 0 && (
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-[0.15em] font-semibold mb-4"
            style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
          >
            PRICE PREDICTION
          </div>
          <PredictionChart
            historicalPrices={[{ date: "today", close: result.current_price }]}
            predictedPrices={prediction.predicted_prices}
            confidenceLower={prediction.confidence_lower || []}
            confidenceUpper={prediction.confidence_upper || []}
            height={350}
          />
        </div>
      )}

      {/* Model info grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Signal strength */}
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-wider mb-2"
            style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
          >
            SIGNAL STRENGTH
          </div>
          <div className="font-num text-3xl font-bold" style={{ color: signalColor }}>
            {(prediction.signal_strength * 100).toFixed(0)}%
          </div>
        </div>

        {/* Model weights */}
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-wider mb-3"
            style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
          >
            MODEL WEIGHTS
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>LSTM</span>
              <span className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                {(prediction.lstm_weight * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex h-1 rounded-full overflow-hidden" style={{ background: "rgba(255,255,255,0.03)" }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${prediction.lstm_weight * 100}%`,
                  background: "#AB47BC",
                }}
              />
              <div
                className="h-full rounded-full"
                style={{
                  width: `${prediction.xgb_weight * 100}%`,
                  background: "#4FC3F7",
                }}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>XGBoost</span>
              <span className="font-num text-sm font-bold" style={{ color: "var(--text-primary)" }}>
                {(prediction.xgb_weight * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Market state (HMM) */}
        <div className="glass-card p-5">
          <div
            className="text-[9px] tracking-wider mb-2"
            style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
          >
            MARKET STATE
          </div>
          {prediction.market_state ? (
            <>
              <div className="font-num text-xl font-bold mb-2" style={{ color: "var(--accent-gold)" }}>
                {prediction.market_state.state_name}
              </div>
              {prediction.market_state.probabilities?.length > 0 && (
                <div className="space-y-1">
                  {["Bull", "Bear", "Sideways"].map((label, i) => {
                    const prob = prediction.market_state!.probabilities[i];
                    if (prob == null) return null;
                    return (
                      <div key={label} className="flex items-center justify-between">
                        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>{label}</span>
                        <span className="font-num text-[10px]" style={{ color: "var(--text-secondary)" }}>
                          {(prob * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          ) : (
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>N/A</span>
          )}
        </div>
      </div>
    </div>
  );
}
