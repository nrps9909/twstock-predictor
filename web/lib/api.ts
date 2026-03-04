// Typed API client

import type {
  StockInfo, StockStatus, TechnicalResult, SentimentSummary,
  PipelineEvent, PredictionRecord,
  MarketOverview, MarketIntel, MarketScanEvent, Alert,
  DeepPipelineResult, FactorICData,
} from "./types";

const BASE = "";  // Use Next.js rewrite proxy
const SSE_BASE = "http://127.0.0.1:8000";  // SSE direct to FastAPI (bypass proxy buffering)

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// Stocks
export const api = {
  getStocks: () => fetchJSON<StockInfo[]>("/api/stocks"),

  getStockStatus: (id: string) => fetchJSON<StockStatus>(`/api/stocks/${id}/status`),

  getPrices: (id: string, limit = 250) =>
    fetchJSON<Array<Record<string, unknown>>>(`/api/stocks/${id}/prices?limit=${limit}`),

  getTechnical: (id: string, days = 120) =>
    fetchJSON<TechnicalResult>(`/api/stocks/${id}/technical?days=${days}`),

  getSentiment: (id: string, days = 30) =>
    fetchJSON<SentimentSummary>(`/api/stocks/${id}/sentiment/summary?days=${days}`),

  // Prediction History
  getPredictionHistory: (stockId?: string, limit = 50) =>
    fetchJSON<PredictionRecord[]>(
      `/api/predictions/history?${stockId ? `stock_id=${stockId}&` : ""}limit=${limit}`
    ),

  getRecentPredictions: (limit = 20) =>
    fetchJSON<PredictionRecord[]>(`/api/predictions/recent?limit=${limit}`),

  // ── Market endpoints ──────────────────────────────
  getMarketOverview: () => fetchJSON<MarketOverview>("/api/market/overview"),

  getMarketRecommendations: () =>
    fetchJSON<{ scan_date: string | null; buy: unknown[]; sell: unknown[] }>("/api/market/recommendations"),

  getMarketIntel: () => fetchJSON<MarketIntel>("/api/market/intel"),

  getInstitutional: () =>
    fetchJSON<Record<string, unknown>>("/api/market/institutional"),

  // ── Alert endpoints ─────────────────────────────
  getAlerts: (limit = 50, unreadOnly = false) =>
    fetchJSON<Alert[]>(`/api/alerts?limit=${limit}&unread_only=${unreadOnly}`),

  getUnreadAlertCount: () =>
    fetchJSON<{ count: number }>("/api/alerts/unread-count"),

  markAlertRead: (id: number) =>
    fetchJSON<{ ok: boolean }>(`/api/alerts/${id}/read`, { method: "PATCH" }),

  markAllAlertsRead: () =>
    fetchJSON<{ ok: boolean }>("/api/alerts/read-all", { method: "PATCH" }),

  // ── Pipeline (deep analysis) endpoints ──────────
  getPipelineResult: (stockId: string) =>
    fetchJSON<DeepPipelineResult | { status: string }>(`/api/market/pipeline/${stockId}`),

  triggerPipeline: (stockId: string) =>
    fetchJSON<{ status: string; stock_id: string }>(`/api/market/pipeline/${stockId}`, { method: "POST" }),

  getFactorIC: (factor: string, window = 60) =>
    fetchJSON<FactorICData>(`/api/market/factor-ic?factor=${factor}&window=${window}`),

  runMarketScan: (
    onEvent: (event: MarketScanEvent) => void,
    opts?: { topN?: number },
  ): AbortController => {
    const controller = new AbortController();

    fetch(`${SSE_BASE}/api/market/scan?top_n=${opts?.topN ?? 40}`, {
      method: "POST",
      signal: controller.signal,
    }).then(async (res) => {
      if (!res.ok || !res.body) return;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6)) as MarketScanEvent;
              onEvent(event);
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    }).catch((err) => {
      if (err.name !== "AbortError") {
        console.error("Market scan SSE error:", err);
      }
    });

    return controller;
  },

  // SSE Pipeline
  runPipeline: (
    id: string,
    onEvent: (event: PipelineEvent) => void,
    opts?: { forceRetrain?: boolean; epochs?: number },
  ): AbortController => {
    const controller = new AbortController();

    fetch(`${SSE_BASE}/api/stocks/${id}/pipeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        force_retrain: opts?.forceRetrain ?? false,
        epochs: opts?.epochs ?? 50,
      }),
      signal: controller.signal,
    }).then(async (res) => {
      if (!res.ok || !res.body) return;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const event = JSON.parse(line.slice(6)) as PipelineEvent;
              onEvent(event);
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    }).catch((err) => {
      if (err.name !== "AbortError") {
        console.error("Pipeline SSE error:", err);
      }
    });

    return controller;
  },
};
