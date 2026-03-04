"use client";

import type { AgentSubEvent } from "@/lib/types";
import { ROLE_LABELS } from "@/lib/constants";
import { Check, Loader2, Circle, MessageSquare, Shield, Cpu } from "lucide-react";

interface AgentActivityFeedProps {
  events: AgentSubEvent[];
}

const ANALYST_ROLES = ["technical", "sentiment", "fundamental", "quant"] as const;

const SIGNAL_DISPLAY: Record<string, { label: string; color: string }> = {
  buy: { label: "BUY", color: "#EF5350" },
  strong_buy: { label: "BUY", color: "#EF5350" },
  sell: { label: "SELL", color: "#26A69A" },
  strong_sell: { label: "SELL", color: "#26A69A" },
  hold: { label: "HOLD", color: "#FFC107" },
  error: { label: "ERR", color: "rgba(232,82,74,0.6)" },
};

export function AgentActivityFeed({ events }: AgentActivityFeedProps) {
  // Derive state from events
  const analystsStarted = events.some((e) => e.substep === "analysts_start");
  const analystResults = new Map<string, AgentSubEvent>();
  const debateRounds: AgentSubEvent[] = [];
  let debateStarted = false;
  let synthesizing = false;
  let ruleEngineResult: AgentSubEvent | null = null;
  let riskCheckResult: AgentSubEvent | null = null;

  for (const evt of events) {
    switch (evt.substep) {
      case "analyst_done":
        if (evt.role) analystResults.set(evt.role, evt);
        break;
      case "debate_start":
        debateStarted = true;
        break;
      case "debate_round":
        debateRounds.push(evt);
        break;
      case "debate_synthesis":
        synthesizing = true;
        break;
      case "rule_engine":
        ruleEngineResult = evt;
        break;
      case "risk_check":
        riskCheckResult = evt;
        break;
    }
  }

  const allAnalystsDone = ANALYST_ROLES.every((r) => analystResults.has(r));
  const firstPendingRole = ANALYST_ROLES.find((r) => !analystResults.has(r));
  const decisionPhase = ruleEngineResult !== null || riskCheckResult !== null;

  return (
    <div
      className="glass-card-static overflow-hidden"
      style={{ borderColor: "rgba(201,168,76,0.1)" }}
    >
      <div className="px-5 py-4">
        <div
          className="text-[9px] tracking-[0.15em] font-semibold mb-4"
          style={{ color: "var(--accent-gold)", fontFamily: "'Space Mono', monospace" }}
        >
          AGENT ANALYSIS
        </div>

        <div className="space-y-4">
          {/* ── ANALYSTS ── */}
          {analystsStarted && (
            <Section title="ANALYSTS" icon={<Cpu className="h-3 w-3" />}>
              <div className="space-y-1.5">
                {ANALYST_ROLES.map((role, i) => {
                  const result = analystResults.get(role);
                  const isDone = !!result;
                  const isRunning = analystsStarted && !isDone && role === firstPendingRole;
                  const label = ROLE_LABELS[role] || role;

                  const sig = result?.signal || "";
                  const display = SIGNAL_DISPLAY[sig];
                  const conf = result?.confidence ?? 0;

                  return (
                    <div
                      key={role}
                      className="flex items-center gap-2.5 py-1 transition-all duration-300"
                      style={{
                        opacity: isDone ? 1 : isRunning ? 0.7 : 0.3,
                        animationDelay: `${i * 80}ms`,
                      }}
                    >
                      <StatusIcon done={isDone} running={isRunning} />
                      <span
                        className="text-[11px] flex-1"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {label}
                      </span>
                      {isDone && display && (
                        <div className="flex items-center gap-2">
                          <span
                            className="text-[10px] font-bold tracking-wider"
                            style={{ color: display.color, fontFamily: "'Space Mono', monospace" }}
                          >
                            {display.label}
                          </span>
                          <span
                            className="font-num text-[10px]"
                            style={{ color: "var(--text-muted)" }}
                          >
                            {Math.round(conf * 100)}%
                          </span>
                        </div>
                      )}
                      {isRunning && (
                        <span
                          className="text-[10px]"
                          style={{ color: "var(--text-muted)" }}
                        >
                          分析中...
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* ── DEBATE ── */}
          {debateStarted && (
            <Section title="DEBATE" icon={<MessageSquare className="h-3 w-3" />}>
              <div className="space-y-2">
                {debateRounds.map((round, i) => (
                  <div key={i} className="space-y-1">
                    <div className="flex items-center gap-2">
                      <StatusIcon done={true} running={false} />
                      <span
                        className="text-[11px] font-medium"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Round {round.round}
                      </span>
                    </div>
                    {/* Bull/Bear arguments */}
                    <div className="ml-6 space-y-1">
                      {round.bull_args && round.bull_args.length > 0 && (
                        <div
                          className="text-[10px] leading-relaxed"
                          style={{ color: "rgba(239,83,80,0.7)" }}
                        >
                          <span className="font-medium">看多:</span>{" "}
                          {round.bull_args.join(", ")}
                        </div>
                      )}
                      {round.bear_args && round.bear_args.length > 0 && (
                        <div
                          className="text-[10px] leading-relaxed"
                          style={{ color: "rgba(38,166,154,0.7)" }}
                        >
                          <span className="font-medium">看空:</span>{" "}
                          {round.bear_args.join(", ")}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                {/* Synthesizing indicator */}
                {synthesizing && !ruleEngineResult && (
                  <div className="flex items-center gap-2">
                    <StatusIcon done={false} running={true} />
                    <span
                      className="text-[11px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      綜合判斷中...
                    </span>
                  </div>
                )}
                {/* Still in debate (no rounds yet but started) */}
                {debateRounds.length === 0 && !synthesizing && (
                  <div className="flex items-center gap-2">
                    <StatusIcon done={false} running={true} />
                    <span
                      className="text-[11px]"
                      style={{ color: "var(--text-muted)" }}
                    >
                      辯論進行中...
                    </span>
                  </div>
                )}
              </div>
            </Section>
          )}

          {/* ── DECISION ── */}
          {(decisionPhase || (allAnalystsDone && debateStarted)) && (
            <Section title="DECISION" icon={<Shield className="h-3 w-3" />}>
              <div className="space-y-1.5">
                {/* Rule engine */}
                <div className="flex items-center gap-2.5 py-1">
                  <StatusIcon
                    done={!!ruleEngineResult}
                    running={!ruleEngineResult && synthesizing}
                  />
                  <span
                    className="text-[11px] flex-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    規則引擎
                  </span>
                  {ruleEngineResult && (
                    <span
                      className="text-[10px] font-bold tracking-wider"
                      style={{
                        color: SIGNAL_DISPLAY[ruleEngineResult.action || "hold"]?.color || "var(--text-muted)",
                        fontFamily: "'Space Mono', monospace",
                      }}
                    >
                      {{ buy: "買進", sell: "賣出", hold: "持有" }[ruleEngineResult.action || "hold"] || ruleEngineResult.action}
                    </span>
                  )}
                </div>

                {/* Risk check */}
                <div className="flex items-center gap-2.5 py-1">
                  <StatusIcon
                    done={!!riskCheckResult}
                    running={!!ruleEngineResult && !riskCheckResult}
                  />
                  <span
                    className="text-[11px] flex-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    風控審核
                  </span>
                  {riskCheckResult && (
                    <span
                      className="text-[10px] font-bold tracking-wider"
                      style={{
                        color: riskCheckResult.approved ? "rgba(34,197,94,0.8)" : "rgba(232,82,74,0.8)",
                        fontFamily: "'Space Mono', monospace",
                      }}
                    >
                      {riskCheckResult.approved ? "通過" : "否決"}
                    </span>
                  )}
                </div>
              </div>
            </Section>
          )}
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div
        className="flex items-center gap-1.5 mb-2"
        style={{ color: "var(--text-muted)" }}
      >
        {icon}
        <span
          className="text-[8px] tracking-[0.15em] font-semibold"
          style={{ fontFamily: "'Space Mono', monospace" }}
        >
          {title}
        </span>
      </div>
      {children}
    </div>
  );
}

function StatusIcon({ done, running }: { done: boolean; running: boolean }) {
  if (done) {
    return (
      <div
        className="flex h-4 w-4 items-center justify-center rounded-full"
        style={{ background: "rgba(34,197,94,0.12)", color: "rgb(34,197,94)" }}
      >
        <Check className="h-2.5 w-2.5" />
      </div>
    );
  }
  if (running) {
    return (
      <div
        className="flex h-4 w-4 items-center justify-center rounded-full"
        style={{ background: "rgba(201,168,76,0.12)", color: "var(--accent-gold)" }}
      >
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
      </div>
    );
  }
  return (
    <div
      className="flex h-4 w-4 items-center justify-center rounded-full"
      style={{ color: "rgba(255,255,255,0.1)" }}
    >
      <Circle className="h-2.5 w-2.5" />
    </div>
  );
}
