"""敘事生成 Agent — LLM 雙層架構

1. 情緒特徵萃取 (Haiku, Phase 2): 從新聞/PTT 萃取結構化情緒
2. 分析報告 + 投資結論 (Opus, Phase 4): 從 20 因子 + regime 生成完整報告與投資建議

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
            streak = f"連續{abs(foreign_days)}日" if foreign_days != 0 else ""
            parts.append(f"外資{streak}{direction}{abs(foreign):,.0f}張")
        if trust != 0:
            direction = "買超" if trust > 0 else "賣超"
            streak = f"連續{abs(trust_days)}日" if trust_days != 0 else ""
            parts.append(f"投信{streak}{direction}{abs(trust):,.0f}張")
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
# LLM Call #2: 分析報告 + 投資結論 (Opus, 合併)
# ═══════════════════════════════════════════════════════

NARRATIVE_PROMPT = """你是頂級台股投資顧問，客戶是散戶投資者。
根據以下 20 因子量化分析結果，撰寫完整的投資分析報告與白話投資結論。

═══ 股票資訊 ═══
代號: {stock_id} {stock_name}
現價: ${current_price}
體制: {regime} (HMM 市場狀態: bull=多頭, bear=空頭, sideways=盤整)

═══ 評分結果 ═══
總分: {total_score:.3f} / 1.00 (>0.6 偏多, <0.4 偏空)
訊號: {signal}
信心度: {confidence:.1%}

═══ 20 因子明細 ═══
{factor_summary}

═══ 演算法推理 ═══
{reasoning}

═══ ML 預測 ═══
{ml_summary}

═══ 法人動向 ═══
{institutional_summary}

═══ 技術面參考 ═══
{technical_levels}

═══ 風控決策 ═══
{risk_summary}

請用 JSON 格式回傳完整分析報告:
{{
    "outlook": "整體展望分析（2-3 句，引用具體因子數據）",
    "outlook_horizon": "信號有效期（如: 1-2 週）",
    "key_drivers": [
        "看多驅動因子 1（引用具體數據，白話解讀）",
        "看多驅動因子 2",
        "看多驅動因子 3"
    ],
    "risks": [
        "風險因素 1（引用具體數據，白話解讀）",
        "風險因素 2"
    ],
    "catalysts": [
        "潛在催化劑 1",
        "潛在催化劑 2"
    ],
    "key_levels": {{
        "support": 支撐價位（數字）,
        "resistance": 壓力價位（數字）
    }},
    "position_suggestion": "具體部位建議（如: 可配置 15% 資金分批買進，若跌破 XXX 元停損）",
    "verdict": "2-4 句投資結論，白話文寫給散戶看。包含：(1) 目前多空狀態判斷 (2) 具體操作建議（買/賣/觀望） (3) 關鍵注意事項。語氣像理財顧問面對面跟客戶說話。",
    "verdict_short": "一句話結論（15-25字），例如: 建議逢低分批佈局，基本面支撐穩健",
    "risk_warning": "一句重大風險提醒（有才填，否則空字串）",
    "confidence_comment": "對分析信心度的白話解讀（一句話，例如: 多數指標方向一致，分析可信度高）"
}}

規則:
- 散戶能懂: 不用 alpha/beta/ATR/Sharpe 等專業術語，全部翻譯成白話
- 數據驅動: 每個觀點都引用具體因子分數或價格數據
- 平衡觀點: 即使偏多也要提風險，即使偏空也要提機會
- 具體可執行: 告訴投資者該做什麼，不要含糊的「可能」「或許」
- key_levels 的 support 應參考 MA20/MA60/近期低點等技術支撐；resistance 應參考前高/壓力均線
- key_levels 必須是合理的價位數字，優先使用「技術面參考」段落提供的數據
- position_suggestion 必須反映風控決策中的部位大小和停損價位（風控已通過才建議部位）
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
    technical_data: dict | None = None,
    risk_context: dict | None = None,
) -> dict:
    """LLM 分析報告 + 投資結論 (Phase 5, Opus)

    Pipeline order: data → features → scoring → risk → narrative.
    risk_context contains the risk control decision so the narrative
    reflects risk-adjusted positions and stop levels.

    Returns:
        dict with all narrative + verdict fields
    """
    # Format factor summary (all available factors)
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
            factor_lines.append(
                f"  {name}: {score:.2f} (權重 {weight:.1%}, {direction})"
            )
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
            streak = f"連續{abs(foreign_days)}日" if foreign_days != 0 else ""
            parts.append(f"外資{streak}{direction}{abs(foreign):,.0f}張")
        if trust != 0:
            direction = "買超" if trust > 0 else "賣超"
            streak = f"連續{abs(trust_days)}日" if trust_days != 0 else ""
            parts.append(f"投信{streak}{direction}{abs(trust):,.0f}張")
        if foreign > 0 and trust > 0:
            parts.append("外資+投信同步買超")
        inst_text = "；".join(parts) if parts else "無法人資料"

    # Reasoning
    reasoning_text = "；".join(reasoning) if reasoning else "無"

    # Technical levels for key_levels guidance
    tech_lines = []
    if technical_data:
        if technical_data.get("ma20") is not None:
            tech_lines.append(f"MA20: {technical_data['ma20']:.1f}")
        if technical_data.get("ma60") is not None:
            tech_lines.append(f"MA60: {technical_data['ma60']:.1f}")
        if technical_data.get("low_20d") is not None:
            tech_lines.append(f"近20日最低: {technical_data['low_20d']:.1f}")
        if technical_data.get("atr14") is not None:
            tech_lines.append(f"ATR(14): {technical_data['atr14']:.1f}")
    technical_levels = "\n".join(tech_lines) if tech_lines else "無技術面數據"

    # Risk summary for LLM
    risk_lines = []
    if risk_context:
        risk_lines.append(f"動作: {risk_context.get('action', '?')}")
        risk_lines.append(f"部位大小: {risk_context.get('position_size_pct', 0):.1f}%")
        risk_lines.append(f"核准: {'是' if risk_context.get('approved') else '否'}")
        if risk_context.get("stop_loss"):
            risk_lines.append(f"停損: ${risk_context['stop_loss']:.1f}")
        if risk_context.get("take_profit"):
            risk_lines.append(f"停利: ${risk_context['take_profit']:.1f}")
        for note in risk_context.get("risk_notes", []):
            risk_lines.append(f"風控備註: {note}")
    risk_summary = "\n".join(risk_lines) if risk_lines else "無風控資料"

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
        technical_levels=technical_levels,
        risk_summary=risk_summary,
    )

    text = await call_claude(prompt, model="claude-opus-4-6", timeout=180)
    result = parse_json_response(text)

    # Validate narrative fields
    if "outlook" not in result:
        result["outlook"] = ""
    if "key_drivers" not in result:
        result["key_drivers"] = []
    if "risks" not in result:
        result["risks"] = []
    # Validate verdict fields
    if "verdict" not in result:
        result["verdict"] = ""
    if "verdict_short" not in result:
        result["verdict_short"] = ""
    if "risk_warning" not in result:
        result["risk_warning"] = ""
    if "confidence_comment" not in result:
        result["confidence_comment"] = ""

    return result
