"use client";

import type { AgentDecision, AnalystReport } from "@/lib/types";
import { SIGNAL_COLORS, ROLE_LABELS, SIGNAL_LABELS } from "@/lib/constants";
import { Shield, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface AgentDebatePanelProps {
  agent: AgentDecision;
}

export function AgentDebatePanel({ agent }: AgentDebatePanelProps) {
  const [expanded, setExpanded] = useState(false);

  const allReports = [
    ...agent.analyst_reports,
    ...(agent.researcher ? [agent.researcher] : []),
  ];

  const actionColor = SIGNAL_COLORS[agent.action as keyof typeof SIGNAL_COLORS] || SIGNAL_COLORS.hold;

  return (
    <div className="glass-card-static overflow-hidden">
      {/* Header bar */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-6 py-5 transition-colors duration-300"
        style={{ background: expanded ? "rgba(201,168,76,0.02)" : "transparent" }}
      >
        <div className="flex items-center gap-4">
          <div className="section-label" style={{ marginBottom: 0 }}>MULTI-AGENT ANALYSIS</div>

          {/* Action badge */}
          <div
            className="flex items-center gap-1.5 rounded-lg px-3 py-1 text-[10px] font-bold tracking-wider"
            style={{
              backgroundColor: `${actionColor}10`,
              color: actionColor,
              border: `1px solid ${actionColor}18`,
              fontFamily: "'Space Mono', monospace",
            }}
          >
            {SIGNAL_LABELS[agent.action] || agent.action}
          </div>

          <span
            className="font-num text-[11px]"
            style={{ color: "var(--text-muted)" }}
          >
            {Math.round(agent.confidence * 100)}%
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Approval badge */}
          <div
            className="flex items-center gap-1"
            style={{ color: agent.approved ? "rgba(34,197,94,0.7)" : "rgba(232,82,74,0.7)" }}
          >
            <Shield className="h-3 w-3" />
            <span
              className="text-[10px] font-medium tracking-wider"
              style={{ fontFamily: "'Space Mono', monospace" }}
            >
              {agent.approved ? "APPROVED" : "DENIED"}
            </span>
          </div>

          {expanded ? (
            <ChevronUp className="h-3.5 w-3.5" style={{ color: "var(--text-muted)" }} />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" style={{ color: "var(--text-muted)" }} />
          )}
        </div>
      </button>

      {/* Agent flow — always visible */}
      <div className="px-6 pb-5">
        <div className="flex items-center gap-1.5 overflow-x-auto py-1">
          {allReports.map((report, i) => {
            const sig = report.signal || "hold";
            const color = SIGNAL_COLORS[sig as keyof typeof SIGNAL_COLORS] || "rgba(139,144,160,0.5)";
            const isResearcher = report.role === "researcher";

            return (
              <div key={i} className="flex items-center gap-1.5">
                <div
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 whitespace-nowrap transition-all duration-200"
                  style={{
                    background: isResearcher ? "rgba(201,168,76,0.05)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${isResearcher ? "rgba(201,168,76,0.12)" : "rgba(255,255,255,0.04)"}`,
                  }}
                >
                  <div
                    className="h-[5px] w-[5px] rounded-full"
                    style={{ backgroundColor: color, boxShadow: `0 0 4px ${color}40` }}
                  />
                  <span
                    className="text-[9px] tracking-wider"
                    style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
                  >
                    {ROLE_LABELS[report.role] || report.role}
                  </span>
                  <span
                    className="text-[9px] font-medium tracking-wider"
                    style={{ color, fontFamily: "'Space Mono', monospace" }}
                  >
                    {sig === "buy" || sig === "strong_buy" ? "BUY" :
                     sig === "sell" || sig === "strong_sell" ? "SELL" : "HOLD"}
                  </span>
                </div>
                {i < allReports.length - 1 && (
                  <div
                    className="w-3 h-px"
                    style={{ background: "linear-gradient(90deg, rgba(201,168,76,0.15), rgba(201,168,76,0.05))" }}
                  />
                )}
              </div>
            );
          })}

          {/* Arrow to final decision */}
          <div
            className="w-4 h-px"
            style={{ background: "linear-gradient(90deg, rgba(201,168,76,0.1), rgba(201,168,76,0.2))" }}
          />

          {/* Final decision */}
          <div
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[9px] font-bold tracking-wider"
            style={{
              background: `${actionColor}08`,
              border: `1px solid ${actionColor}15`,
              color: actionColor,
              fontFamily: "'Space Mono', monospace",
            }}
          >
            <Shield className="h-3 w-3" />
            {agent.approved ? "PASS" : "DENY"}
          </div>
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div
          className="px-6 pb-6 space-y-3"
          style={{ borderTop: "1px solid rgba(201,168,76,0.06)" }}
        >
          <div className="pt-5 space-y-2.5">
            {allReports.map((report, i) => (
              <AgentReportCard key={i} report={report} />
            ))}
          </div>

          {/* Reasoning */}
          {agent.reasoning && (
            <div
              className="rounded-xl p-4"
              style={{
                background: "rgba(201,168,76,0.03)",
                border: "1px solid rgba(201,168,76,0.08)",
              }}
            >
              <div
                className="text-[9px] tracking-wider mb-2"
                style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
              >
                REASONING
              </div>
              <div className="text-[13px] leading-relaxed" style={{ color: "var(--text-primary)" }}>
                {agent.reasoning}
              </div>
            </div>
          )}

          {/* Risk notes */}
          {agent.risk_notes && (
            <div
              className="rounded-xl p-4"
              style={{
                background: "rgba(232,82,74,0.03)",
                border: "1px solid rgba(232,82,74,0.08)",
              }}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <Shield className="h-3 w-3" style={{ color: "rgba(232,82,74,0.6)" }} />
                <span
                  className="text-[9px] tracking-wider"
                  style={{ color: "rgba(232,82,74,0.6)", fontFamily: "'Space Mono', monospace" }}
                >
                  RISK NOTES
                </span>
              </div>
              <div className="text-[13px] leading-relaxed" style={{ color: "var(--text-primary)" }}>
                {agent.risk_notes}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AgentReportCard({ report }: { report: AnalystReport }) {
  const sig = report.signal || "hold";
  const color = SIGNAL_COLORS[sig as keyof typeof SIGNAL_COLORS] || "rgba(139,144,160,0.5)";

  return (
    <div
      className="rounded-xl p-4 transition-all duration-200"
      style={{
        background: "rgba(255,255,255,0.015)",
        border: "1px solid rgba(255,255,255,0.04)",
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <span
          className="text-[10px] tracking-wider font-medium"
          style={{ color: "var(--text-secondary)", fontFamily: "'Space Mono', monospace" }}
        >
          {ROLE_LABELS[report.role] || report.role}
        </span>
        <div className="flex items-center gap-3">
          <span
            className="text-[10px] font-bold tracking-wider"
            style={{ color, fontFamily: "'Space Mono', monospace" }}
          >
            {SIGNAL_LABELS[sig] || sig}
          </span>
          <span className="font-num text-[10px]" style={{ color: "var(--text-muted)" }}>
            {Math.round(report.confidence * 100)}%
          </span>
        </div>
      </div>
      {report.reasoning && (
        <p className="text-[12px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          {report.reasoning}
        </p>
      )}
    </div>
  );
}
