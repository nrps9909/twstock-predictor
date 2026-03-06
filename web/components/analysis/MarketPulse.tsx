"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { SIGNAL_COLORS, SIGNAL_LABELS, STOCK_LIST } from "@/lib/constants";

interface MarketPulseProps {
  onAnalyze: (stockId: string) => void;
}

export function MarketPulse({ onAnalyze }: MarketPulseProps) {
  const { data } = useQuery({
    queryKey: ["market-recommendations"],
    queryFn: () => api.getMarketRecommendations(),
    staleTime: 5 * 60 * 1000,
  });

  if (!data) return null;

  const items = [
    ...((data.buy || []) as any[]).map((s: any) => ({ ...s, type: "buy" })),
    ...((data.sell || []) as any[]).map((s: any) => ({ ...s, type: "sell" })),
  ].slice(0, 8);

  if (items.length === 0) return null;

  return (
    <div className="card-reveal">
      <div className="flex items-center gap-2 mb-3">
        <div
          className="text-[9px] tracking-[0.15em] font-semibold"
          style={{ color: "var(--text-secondary)", fontFamily: "'Space Mono', monospace" }}
        >
          MARKET PULSE
        </div>
        {data.scan_date && (
          <span className="text-[9px] font-num" style={{ color: "var(--text-muted)" }}>
            {data.scan_date}
          </span>
        )}
      </div>
      <div className="flex gap-3 overflow-x-auto pb-2">
        {items.map((item: any) => {
          const signal = item.signal || item.type;
          const signalColor =
            SIGNAL_COLORS[signal as keyof typeof SIGNAL_COLORS] || "#9E9E9E";
          const stockId = item.stock_id || "";
          const score = item.total_score ?? item.score ?? null;
          const priceChange = item.price_change_pct ?? null;

          return (
            <button
              key={stockId}
              onClick={() => onAnalyze(stockId)}
              className="glass-card shrink-0 p-3 text-left transition-all duration-200 hover:scale-[1.02]"
              style={{ width: 160 }}
            >
              <div className="flex items-center justify-between mb-2">
                <span
                  className="font-num text-sm font-bold"
                  style={{ color: "var(--accent-gold)" }}
                >
                  {stockId}
                </span>
                <span
                  className="rounded-md px-1.5 py-0.5 text-[9px] font-bold"
                  style={{
                    background: `${signalColor}12`,
                    color: signalColor,
                    border: `1px solid ${signalColor}25`,
                  }}
                >
                  {SIGNAL_LABELS[signal] || signal}
                </span>
              </div>
              <div
                className="text-[10px] truncate mb-1.5"
                style={{ color: "var(--text-secondary)" }}
              >
                {item.stock_name || STOCK_LIST[stockId] || ""}
              </div>
              <div className="flex items-center justify-between">
                {priceChange !== null && (
                  <span
                    className="font-num text-[10px]"
                    style={{
                      color:
                        priceChange >= 0 ? "var(--signal-buy)" : "var(--signal-sell)",
                    }}
                  >
                    {priceChange >= 0 ? "+" : ""}
                    {Number(priceChange).toFixed(2)}%
                  </span>
                )}
                {score !== null && (
                  <div className="flex items-center gap-1">
                    <div
                      className="h-1 rounded-full"
                      style={{
                        width: 40,
                        background: "rgba(255,255,255,0.04)",
                      }}
                    >
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(0, Math.min(100, Number(score) * 100))}%`,
                          background: signalColor,
                          opacity: 0.7,
                        }}
                      />
                    </div>
                    <span
                      className="font-num text-[9px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {Number(score).toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
