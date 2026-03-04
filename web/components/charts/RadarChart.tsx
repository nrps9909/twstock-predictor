"use client";

import {
  ResponsiveContainer,
  RadarChart as RechartsRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
} from "recharts";

interface RadarChartProps {
  data: { name: string; value: number; fullMark?: number }[];
  height?: number;
}

export function RadarChart({ data, height = 250 }: RadarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsRadar data={data} cx="50%" cy="50%" outerRadius="70%">
        <PolarGrid stroke="rgba(255,255,255,0.06)" />
        <PolarAngleAxis
          dataKey="name"
          tick={{ fill: "#8B90A0", fontSize: 11 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[0, 100]}
          tick={false}
          axisLine={false}
        />
        <Radar
          name="分析"
          dataKey="value"
          stroke="#D4A017"
          fill="rgba(212, 160, 23, 0.2)"
          fillOpacity={0.6}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1A1F2E",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "8px",
            fontSize: "12px",
          }}
        />
      </RechartsRadar>
    </ResponsiveContainer>
  );
}
