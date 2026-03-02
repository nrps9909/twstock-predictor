"""Multi-Agent 自動交易系統

架構（參考 TradingAgents arxiv:2412.20138）：

[技術面 Agent] ──┐
[情緒面 Agent] ──┤──→ [研究員 Agent] ──→ [交易員 Agent] ──→ [風控 Agent] ──→ 執行
[基本面 Agent] ──┤     (多空辯論)        (決策)            (核准/否決)
[量化 Agent]   ──┘
"""
