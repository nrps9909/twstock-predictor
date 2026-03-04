"use client";

import { useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { cn } from "@/lib/utils";

interface ChartDataPoint {
  date: string;
  [key: string]: string | number | null;
}

interface TechnicalIndicatorsProps {
  data: ChartDataPoint[];
  height?: number;
}

const TABS = [
  { key: "kd", label: "KD" },
  { key: "rsi", label: "RSI" },
  { key: "macd", label: "MACD" },
  { key: "bb", label: "BB" },
];

export function TechnicalIndicators({ data, height = 200 }: TechnicalIndicatorsProps) {
  const [activeTab, setActiveTab] = useState("kd");

  return (
    <div>
      {/* Tabs */}
      <div className="flex gap-1 mb-3">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              "rounded-md px-3 py-1 text-xs font-medium transition-colors",
              activeTab === tab.key
                ? "bg-accent-gold/10 text-accent-gold"
                : "text-text-secondary hover:text-text-primary hover:bg-white/[0.04]"
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        {activeTab === "kd" ? (
          <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: "#1A1F2E", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", fontSize: "11px" }} />
            <ReferenceLine y={80} stroke="rgba(239,83,80,0.3)" strokeDasharray="3 3" />
            <ReferenceLine y={20} stroke="rgba(38,166,154,0.3)" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="kd_k" stroke="#FFD700" strokeWidth={1.5} dot={false} name="K" />
            <Line type="monotone" dataKey="kd_d" stroke="#4FC3F7" strokeWidth={1.5} dot={false} name="D" />
          </ComposedChart>
        ) : activeTab === "rsi" ? (
          <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <YAxis domain={[0, 100]} tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: "#1A1F2E", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", fontSize: "11px" }} />
            <ReferenceLine y={70} stroke="rgba(239,83,80,0.3)" strokeDasharray="3 3" />
            <ReferenceLine y={30} stroke="rgba(38,166,154,0.3)" strokeDasharray="3 3" />
            <Line type="monotone" dataKey="rsi_14" stroke="#AB47BC" strokeWidth={1.5} dot={false} name="RSI" />
          </ComposedChart>
        ) : activeTab === "macd" ? (
          <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <YAxis tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: "#1A1F2E", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", fontSize: "11px" }} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.1)" />
            <Bar dataKey="macd_hist" name="柱狀體" fill="rgba(239,83,80,0.5)" />
            <Line type="monotone" dataKey="macd" stroke="#FFD700" strokeWidth={1.5} dot={false} name="MACD" />
            <Line type="monotone" dataKey="macd_signal" stroke="#4FC3F7" strokeWidth={1.5} dot={false} name="Signal" />
          </ComposedChart>
        ) : (
          <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis dataKey="date" tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <YAxis tick={{ fill: "#8B90A0", fontSize: 9 }} tickLine={false} />
            <Tooltip contentStyle={{ backgroundColor: "#1A1F2E", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", fontSize: "11px" }} />
            <Line type="monotone" dataKey="bb_upper" stroke="rgba(255,255,255,0.2)" strokeWidth={1} dot={false} name="上軌" />
            <Line type="monotone" dataKey="bb_middle" stroke="#4FC3F7" strokeWidth={1} dot={false} name="中軌" />
            <Line type="monotone" dataKey="bb_lower" stroke="rgba(255,255,255,0.2)" strokeWidth={1} dot={false} name="下軌" />
            <Line type="monotone" dataKey="close" stroke="#E2E4E9" strokeWidth={1.5} dot={false} name="收盤" />
          </ComposedChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}
