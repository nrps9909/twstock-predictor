"use client";

import type { PipelineResult, AgentSubEvent } from "@/lib/types";
import { AgentDebatePanel } from "@/components/dashboard/AgentDebatePanel";
import { AgentActivityFeed } from "@/components/dashboard/AgentActivityFeed";

interface AgentTabProps {
  result: PipelineResult;
  agentSubEvents: AgentSubEvent[];
}

export function AgentTab({ result, agentSubEvents }: AgentTabProps) {
  if (!result.agent) {
    return (
      <div className="glass-card p-12 text-center">
        <span className="text-sm" style={{ color: "var(--text-muted)" }}>
          Agent 分析資料不可用
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Debate panel (main) */}
      <AgentDebatePanel agent={result.agent} />

      {/* Activity feed (real-time events) */}
      {agentSubEvents.length > 0 && (
        <AgentActivityFeed events={agentSubEvents} />
      )}
    </div>
  );
}
