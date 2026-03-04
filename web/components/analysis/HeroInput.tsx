"use client";

import { useState, useRef, useEffect } from "react";
import { STOCK_LIST } from "@/lib/constants";
import { Search, Loader2, X } from "lucide-react";

interface HeroInputProps {
  onAnalyze: (stockId: string) => void;
  onAbort: () => void;
  isRunning: boolean;
  isCompact: boolean;
  currentStock?: string | null;
}

const QUICK_PICKS = ["2330", "2317", "2454", "2881", "2303", "2308"];

export function HeroInput({ onAnalyze, onAbort, isRunning, isCompact, currentStock }: HeroInputProps) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-focus on mount
  useEffect(() => {
    if (!isRunning && !isCompact) {
      inputRef.current?.focus();
    }
  }, [isRunning, isCompact]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || isRunning) return;
    if (!/^\d{4}$/.test(trimmed)) return;
    onAnalyze(trimmed);
  };

  const handleQuickPick = (stockId: string) => {
    if (isRunning) return;
    setInput(stockId);
    onAnalyze(stockId);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSubmit();
  };

  // Compact mode (after analysis complete)
  if (isCompact && currentStock) {
    return (
      <div className="flex items-center justify-center gap-4 py-4">
        <div className="flex items-center gap-2">
          <span className="font-num text-lg font-bold" style={{ color: "var(--accent-gold)" }}>
            {currentStock}
          </span>
          {STOCK_LIST[currentStock] && (
            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {STOCK_LIST[currentStock]}
            </span>
          )}
        </div>
        <button
          onClick={() => {
            setInput("");
            onAnalyze(currentStock);
          }}
          className="rounded-lg px-3 py-1.5 text-xs font-medium transition-all duration-200"
          style={{
            background: "rgba(201,168,76,0.08)",
            border: "1px solid rgba(201,168,76,0.15)",
            color: "var(--accent-gold)",
          }}
        >
          重新分析
        </button>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col items-center justify-center transition-all duration-500"
      style={{ minHeight: isRunning ? "auto" : "50vh" }}
    >
      {/* Title */}
      {!isRunning && (
        <div className="text-center mb-8">
          <h1
            className="font-display text-4xl md:text-5xl tracking-[0.15em] mb-3"
            style={{ color: "var(--accent-gold)" }}
          >
            AURUM
          </h1>
          <p className="text-sm" style={{ color: "var(--text-muted)" }}>
            AI 驅動的台股深度分析
          </p>
        </div>
      )}

      {/* Input row */}
      <div className="flex items-center gap-3 w-full max-w-lg">
        <div
          className="relative flex-1 group"
          style={{
            boxShadow: isRunning ? "none" : "0 0 30px rgba(201,168,76,0.08)",
          }}
        >
          <Search
            className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 transition-colors"
            style={{ color: isRunning ? "var(--text-muted)" : "var(--accent-gold)" }}
          />
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value.replace(/\D/g, "").slice(0, 4))}
            onKeyDown={handleKeyDown}
            placeholder="輸入股票代碼 (例: 2330)"
            disabled={isRunning}
            className="w-full rounded-xl pl-11 pr-4 py-3.5 text-sm font-num outline-none transition-all duration-200"
            style={{
              background: "rgba(255,255,255,0.03)",
              border: `1px solid ${isRunning ? "var(--border)" : "rgba(201,168,76,0.2)"}`,
              color: "var(--text-primary)",
              opacity: isRunning ? 0.5 : 1,
            }}
          />
        </div>

        {isRunning ? (
          <button
            onClick={onAbort}
            className="flex items-center gap-2 rounded-xl px-5 py-3.5 text-sm font-medium transition-all duration-200"
            style={{
              background: "rgba(232,82,74,0.1)",
              border: "1px solid rgba(232,82,74,0.2)",
              color: "#EF5350",
            }}
          >
            <X className="h-4 w-4" />
            取消
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!/^\d{4}$/.test(input.trim())}
            className="btn-gold flex items-center gap-2 rounded-xl px-5 py-3.5 text-sm font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-all duration-200"
          >
            <Search className="h-4 w-4" />
            開始分析
          </button>
        )}
      </div>

      {/* Quick picks */}
      {!isRunning && (
        <div className="flex items-center gap-2 mt-5 flex-wrap justify-center">
          <span className="text-[10px] tracking-wider mr-1" style={{ color: "var(--text-muted)" }}>
            快速選擇
          </span>
          {QUICK_PICKS.map((id) => (
            <button
              key={id}
              onClick={() => handleQuickPick(id)}
              className="rounded-lg px-3 py-1 text-xs font-num transition-all duration-200"
              style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.06)",
                color: "var(--text-secondary)",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "rgba(201,168,76,0.2)";
                (e.currentTarget as HTMLElement).style.color = "var(--accent-gold)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.06)";
                (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
              }}
            >
              {id} {STOCK_LIST[id] || ""}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
