"use client";

import { useEffect, useRef } from "react";
import { createChart, type IChartApi, type ISeriesApi, ColorType } from "lightweight-charts";

interface CandleData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface CandlestickChartProps {
  data: CandleData[];
  height?: number;
  showVolume?: boolean;
  overlays?: {
    sma_5?: number[];
    sma_20?: number[];
    sma_60?: number[];
  };
}

export function CandlestickChart({
  data,
  height = 400,
  showVolume = true,
  overlays,
}: CandlestickChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8B90A0",
        fontSize: 11,
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: {
        vertLines: { color: "rgba(255, 255, 255, 0.03)" },
        horzLines: { color: "rgba(255, 255, 255, 0.03)" },
      },
      crosshair: {
        vertLine: { color: "rgba(212, 160, 23, 0.3)", width: 1 },
        horzLine: { color: "rgba(212, 160, 23, 0.3)", width: 1 },
      },
      rightPriceScale: {
        borderColor: "rgba(255, 255, 255, 0.06)",
      },
      timeScale: {
        borderColor: "rgba(255, 255, 255, 0.06)",
        timeVisible: false,
      },
    });

    chartRef.current = chart;

    // Candlestick series — Taiwan convention: red = up, green = down
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#EF5350",
      downColor: "#26A69A",
      borderUpColor: "#EF5350",
      borderDownColor: "#26A69A",
      wickUpColor: "#EF5350",
      wickDownColor: "#26A69A",
    });

    const candleData = data.map((d) => ({
      time: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));
    candleSeries.setData(candleData as any);

    // Volume
    if (showVolume) {
      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });

      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });

      const volumeData = data.map((d) => ({
        time: d.date,
        value: d.volume || 0,
        color: d.close >= d.open ? "rgba(239, 83, 80, 0.3)" : "rgba(38, 166, 154, 0.3)",
      }));
      volumeSeries.setData(volumeData as any);
    }

    // MA overlays
    if (overlays) {
      const maColors = { sma_5: "#FFD700", sma_20: "#4FC3F7", sma_60: "#AB47BC" };
      for (const [key, values] of Object.entries(overlays)) {
        if (!values || values.length === 0) continue;
        const lineSeries = chart.addLineSeries({
          color: maColors[key as keyof typeof maColors] || "#888",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        const lineData = values
          .map((v, i) => ({
            time: data[i]?.date,
            value: v,
          }))
          .filter((d) => d.time && d.value != null && !isNaN(d.value));
        lineSeries.setData(lineData as any);
      }
    }

    chart.timeScale().fitContent();

    // Resize handler
    const resizeObserver = new ResizeObserver(() => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [data, height, showVolume, overlays]);

  return <div ref={containerRef} className="w-full" />;
}
