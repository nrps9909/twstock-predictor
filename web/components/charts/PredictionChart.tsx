"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";

interface PredictionChartProps {
  historicalPrices: { date: string; close: number }[];
  predictedPrices: number[];
  confidenceLower: number[];
  confidenceUpper: number[];
  height?: number;
}

export function PredictionChart({
  historicalPrices,
  predictedPrices,
  confidenceLower,
  confidenceUpper,
  height = 300,
}: PredictionChartProps) {
  // Combine historical + predicted
  const lastHistorical = historicalPrices[historicalPrices.length - 1];
  const lastDate = lastHistorical?.date || "";
  const lastPrice = lastHistorical?.close || 0;

  // Create chart data
  const chartData = [
    // Last 30 days of historical
    ...historicalPrices.slice(-30).map((d) => ({
      date: d.date,
      historical: d.close,
      predicted: null as number | null,
      upper: null as number | null,
      lower: null as number | null,
    })),
    // Transition point (connect lines)
    {
      date: lastDate,
      historical: lastPrice,
      predicted: lastPrice,
      upper: lastPrice,
      lower: lastPrice,
    },
    // Predicted days
    ...predictedPrices.map((price, i) => ({
      date: `T+${i + 1}`,
      historical: null as number | null,
      predicted: price,
      upper: confidenceUpper[i],
      lower: confidenceLower[i],
    })),
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#8B90A0", fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
        />
        <YAxis
          tick={{ fill: "#8B90A0", fontSize: 10, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
          domain={["auto", "auto"]}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1A1F2E",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "8px",
            fontSize: "12px",
            fontFamily: "JetBrains Mono",
          }}
        />

        {/* Confidence interval area */}
        <Area
          type="monotone"
          dataKey="upper"
          stroke="none"
          fill="rgba(212, 160, 23, 0.1)"
          stackId="ci"
        />
        <Area
          type="monotone"
          dataKey="lower"
          stroke="none"
          fill="transparent"
          stackId="ci"
        />

        {/* Historical line */}
        <Line
          type="monotone"
          dataKey="historical"
          stroke="#E2E4E9"
          strokeWidth={2}
          dot={false}
          connectNulls={false}
        />

        {/* Predicted line */}
        <Line
          type="monotone"
          dataKey="predicted"
          stroke="#D4A017"
          strokeWidth={2}
          strokeDasharray="5 5"
          dot={{ fill: "#D4A017", r: 3 }}
          connectNulls={false}
        />

        {/* Divider */}
        <ReferenceLine
          x={lastDate}
          stroke="rgba(212, 160, 23, 0.3)"
          strokeDasharray="3 3"
          label={{ value: "今日", fill: "#8B90A0", fontSize: 10 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
