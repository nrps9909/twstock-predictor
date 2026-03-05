"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { AnalysisResult, TechnicalResult } from "@/lib/types";
import { OverviewTab } from "./OverviewTab";
import { TechnicalTab } from "./TechnicalTab";
import { FactorTab } from "./FactorTab";
import { SentimentTab } from "./SentimentTab";

const TABS = [
  { key: "overview", label: "總覽" },
  { key: "technical", label: "技術分析" },
  { key: "factors", label: "因子分析" },
  { key: "sentiment", label: "情緒" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

interface ResultTabsProps {
  result: AnalysisResult;
  technicalData: TechnicalResult | null;
}

export function ResultTabs({ result, technicalData }: ResultTabsProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  return (
    <div>
      {/* Sticky tab bar */}
      <div
        className="sticky top-14 z-40 flex gap-1 px-4 py-2 -mx-6 mb-4"
        style={{
          background: "rgba(11,14,22,0.9)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              "relative px-4 py-2 rounded-lg text-xs font-medium transition-all duration-200",
              activeTab === tab.key
                ? "text-[#C9A84C]"
                : "text-[#4B5263] hover:text-[#8B90A0]"
            )}
          >
            {tab.label}
            {activeTab === tab.key && (
              <div
                className="absolute bottom-0 left-3 right-3 h-[2px] rounded-full"
                style={{ background: "var(--accent-gold)" }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === "overview" && <OverviewTab result={result} />}
        {activeTab === "technical" && <TechnicalTab technicalData={technicalData} />}
        {activeTab === "factors" && <FactorTab result={result} />}
        {activeTab === "sentiment" && <SentimentTab result={result} />}
      </div>
    </div>
  );
}
