"""敘事生成 Agent — LLM 只負責兩件事

1. 情緒特徵萃取 (Haiku, Phase 2): 從新聞/PTT 萃取結構化情緒
2. 敘事生成 (Sonnet, Phase 4): 從 20 因子 + regime 生成分析報告

取代原有 4 Agent + 多輪辯論機制。
"""

import logging

import pandas as pd

from src.utils.llm_client import call_claude, parse_json_response

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# LLM Call #1: 情緒特徵萃取 (Haiku)
# ═══════════════════════════════════════════════════════

SENTIMENT_EXTRACT_PROMPT = """你是台股情緒分析引擎。從以下資料萃取結構化情緒特徵。

股票: {stock_id}

═══ 近期新聞/情緒資料 ═══
{sentiment_data}

═══ 法人動向摘要 ═══
{institutional_summary}

═══ 全球市場數據 ═══
{global_summary}

請分析並回傳 JSON:
{{
    "sentiment_score": 0.0到1.0之間的數字（0.5=中性, >0.5=偏多, <0.5=偏空）,
    "key_themes": ["主題1", "主題2"],
    "contrarian_flag": false,
    "geopolitical_risk": "low 或 medium 或 high",
    "sector_sentiment": "positive 或 neutral 或 negative",
    "catalyst_timeline": "near 或 medium 或 distant"
}}

規則:
- 法人與散戶情緒背離時，通常跟隨法人方向
- 極端情緒可能是反向指標 (contrarian_flag=true)
- 無資料時偏向中性 (0.5)
- 只回傳 JSON"""


async def extract_sentiment(
    stock_id: str,
    sentiment_df: pd.DataFrame | None = None,
    trust_info: dict | None = None,
    global_data: dict | None = None,
) -> dict:
    """LLM 情緒特徵萃取 (Phase 2, Haiku)

    Returns:
        dict with sentiment_score, key_themes, contrarian_flag, etc.
    """
    # Format sentiment data
    sentiment_text = "無情緒資料"
    if sentiment_df is not None and not sentiment_df.empty:
        lines = []
        for _, row in sentiment_df.head(15).iterrows():
            source = row.get("source", "")
            title = row.get("title", "")
            score = row.get("sentiment_score", "")
            lines.append(f"- [{source}] {title} (score={score})")
        sentiment_text = "\n".join(lines) if lines else "無情緒資料"

    # Format institutional summary
    inst_text = "無法人資料"
    if trust_info:
        parts = []
        foreign = trust_info.get("foreign_cumulative", 0)
        trust = trust_info.get("trust_cumulative", 0)
        foreign_days = trust_info.get("foreign_consecutive_days", 0)
        trust_days = trust_info.get("trust_consecutive_days", 0)
        if foreign != 0:
            direction = "買超" if foreign > 0 else "賣超"
            parts.append(f"外資{foreign_days}日{direction}{abs(foreign):,.0f}張")
        if trust != 0:
            direction = "買超" if trust > 0 else "賣超"
            parts.append(f"投信{trust_days}日{direction}{abs(trust):,.0f}張")
        inst_text = "；".join(parts) if parts else "無法人資料"

    # Format global data
    global_text = "無全球市場資料"
    if global_data:
        parts = []
        sox = global_data.get("sox_return", 0)
        tsm = global_data.get("tsm_return", 0)
        ewt_20d = global_data.get("ewt_return_20d", 0)
        if sox:
            parts.append(f"SOX 日報酬: {sox:+.2%}")
        if tsm:
            parts.append(f"TSM ADR 日報酬: {tsm:+.2%}")
        if ewt_20d:
            parts.append(f"EWT 20日報酬: {ewt_20d:+.2%}")
        global_text = "\n".join(f"- {p}" for p in parts) if parts else "無全球市場資料"

    prompt = SENTIMENT_EXTRACT_PROMPT.format(
        stock_id=stock_id,
        sentiment_data=sentiment_text,
        institutional_summary=inst_text,
        global_summary=global_text,
    )

    text = await call_claude(prompt, model="claude-haiku-4-5-20251001", timeout=60)
    result = parse_json_response(text)

    # Validate
    if "sentiment_score" not in result:
        result["sentiment_score"] = 0.5
    else:
        result["sentiment_score"] = max(0.0, min(1.0, float(result["sentiment_score"])))

    return result


# ═══════════════════════════════════════════════════════
# LLM Call #2: 敘事生成 (Sonnet)
# ═══════════════════════════════════════════════════════

NARRATIVE_PROMPT = """你是台股高級分析師。根據以下 20 因子分析結果，生成結構化投資分析報告。

═══ 股票資訊 ═══
代號: {stock_id} {stock_name}
現價: {current_price}
體制: {regime} (HMM 市場狀態)

═══ 評分結果 ═══
總分: {total_score:.2f} / 1.00
訊號: {signal}
信心度: {confidence:.2f}

═══ 20 因子明細 ═══
{factor_summary}

═══ 演算法推理 ═══
{reasoning}

═══ ML 預測 ═══
{ml_summary}

═══ 法人動向 ═══
{institutional_summary}

請用 JSON 格式回傳分析報告:
{{
    "outlook": "整體展望（1-2 句話）",
    "outlook_horizon": "信號有效期（如: 1-2 週）",
    "key_drivers": [
        "驅動因子 1（引用具體數據）",
        "驅動因子 2",
        "驅動因子 3"
    ],
    "risks": [
        "風險 1（引用具體數據）",
        "風險 2"
    ],
    "catalysts": [
        "潛在催化劑 1",
        "潛在催化劑 2"
    ],
    "key_levels": {{
        "support": [支撐1, 支撐2],
        "resistance": [壓力1, 壓力2]
    }},
    "position_suggestion": "部位建議（如: 核心持股可加碼 10%，設停損 XXX）"
}}

規則:
- 數據驅動: 每個觀點都引用具體因子分數或數據
- 平衡觀點: 即使偏多也要提風險，即使偏空也要提機會
- 具體明確: 避免模糊的「可能」「或許」
- 只回傳 JSON"""


async def generate_narrative(
    stock_id: str,
    stock_name: str,
    factor_details: dict,
    total_score: float,
    signal: str,
    confidence: float,
    regime: str,
    ml_scores: dict | None = None,
    trust_info: dict | None = None,
    current_price: float = 0,
    reasoning: list[str] | None = None,
) -> dict:
    """LLM 敘事生成 (Phase 4, Sonnet)

    Returns:
        dict with outlook, key_drivers, risks, catalysts, key_levels, position_suggestion
    """
    # Format factor summary
    factor_lines = []
    sorted_factors = sorted(
        factor_details.items(),
        key=lambda x: x[1].get("weight", 0),
        reverse=True,
    )
    for name, detail in sorted_factors:
        score = detail.get("score", 0.5)
        weight = detail.get("weight", 0)
        available = detail.get("available", False)
        if available:
            direction = "偏多" if score > 0.55 else ("偏空" if score < 0.45 else "中性")
            factor_lines.append(f"  {name}: {score:.2f} (權重 {weight:.1%}, {direction})")
        else:
            factor_lines.append(f"  {name}: N/A (無資料)")
    factor_summary = "\n".join(factor_lines)

    # ML summary
    ml_summary = "無 ML 預測"
    if ml_scores:
        ml_val = ml_scores.get(stock_id)
        if ml_val is not None:
            ml_dir = "看多" if ml_val > 0.6 else ("看空" if ml_val < 0.4 else "中性")
            ml_summary = f"ML 分數: {ml_val:.2f} ({ml_dir})"

    # Institutional summary
    inst_text = "無法人資料"
    if trust_info:
        parts = []
        foreign = trust_info.get("foreign_cumulative", 0)
        trust = trust_info.get("trust_cumulative", 0)
        foreign_days = trust_info.get("foreign_consecutive_days", 0)
        trust_days = trust_info.get("trust_consecutive_days", 0)
        if foreign != 0:
            direction = "買超" if foreign > 0 else "賣超"
            parts.append(f"外資{foreign_days}日{direction}{abs(foreign):,.0f}張")
        if trust != 0:
            direction = "買超" if trust > 0 else "賣超"
            parts.append(f"投信{trust_days}日{direction}{abs(trust):,.0f}張")
        if foreign > 0 and trust > 0:
            parts.append("外資+投信同步買超")
        inst_text = "；".join(parts) if parts else "無法人資料"

    # Reasoning
    reasoning_text = "；".join(reasoning) if reasoning else "無"

    prompt = NARRATIVE_PROMPT.format(
        stock_id=stock_id,
        stock_name=stock_name,
        current_price=current_price,
        regime=regime,
        total_score=total_score,
        signal=signal,
        confidence=confidence,
        factor_summary=factor_summary,
        reasoning=reasoning_text,
        ml_summary=ml_summary,
        institutional_summary=inst_text,
    )

    text = await call_claude(prompt, model="claude-sonnet-4-6", timeout=120)
    result = parse_json_response(text)

    # Validate required fields
    if "outlook" not in result:
        result["outlook"] = ""
    if "key_drivers" not in result:
        result["key_drivers"] = []
    if "risks" not in result:
        result["risks"] = []

    return result
