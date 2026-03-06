# 20 因子評分系統分析報告

> 產出日期：2026-03-06
> 核心檔案：`api/services/market_service.py`

---

## 1. 系統概覽

### 1.1 架構

20 因子評分系統是 twstock-predictor 的核心決策引擎，負責將多維度市場資訊壓縮為單一 0–1 分數與交易信號。系統分三個時間維度：

| 維度 | 因子數 | 權重合計 | 設計意圖 |
|------|--------|----------|----------|
| 短期 (Short) | 6 | 39% | 捕捉籌碼動能與技術面即時變化 |
| 中期 (Mid) | 8 | 32% | 趨勢確認、營收轉折、法人共識 |
| 長期 (Long) | 6 | 29% | 基本面價值、宏觀環境、ML 預測 |

### 1.2 評分流程

```
資料收集 → 20 因子各自計算 FactorResult(score, available, freshness)
         → HMM 體制判斷 (bull/bear/sideways)
         → REGIME_MULTIPLIERS 調整 BASE_WEIGHTS
         → 缺資料因子權重重分配
         → 加權總分 total_score ∈ [0, 1]
         → 信號映射 (strong_buy / buy / hold / sell / strong_sell)
         → 信心度計算 (agreement + strength + coverage + freshness) × risk_discount
```

### 1.3 信號閾值

| 分數區間 | 信號 |
|----------|------|
| > 0.70 | strong_buy |
| > 0.60 | buy |
| 0.40–0.60 | hold |
| < 0.40 | sell |
| < 0.30 | strong_sell |

---

## 2. 20 因子逐一分析

---

### F01 — foreign_flow（外資籌碼流向）

**權重：11%（最高）｜分類：短期**

#### 理論基礎
外資為台股最大法人，持股比例與買賣超對股價有顯著領先性。研究顯示外資淨買超與未來 5–20 日報酬呈正相關，尤其在大型權值股。

#### 資料來源
- TWSE T86 API（`twse_scanner.py`）→ `trust_info["foreign_cumulative"]`（5 日累計）
- `trust_info["foreign_consecutive_days"]`（連續買超天數）
- `df["foreign_buy_sell"]`（歷史每日淨買賣，FinMind `TaiwanStockInstitutionalInvestorsBuySell`）
- `avg_vol_20d`（20 日均量，用於標準化）

#### 實作細節

4 個子因子加權合成：

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Net Normalized | 40% | `0.5 + (net_5d / avg_vol_20d) × 0.2` [0,1] | 買超量占成交量比例 |
| Consecutive Days | 20% | `min(0.5 + days × 0.1, 1.0)` | 連續買超天數的趨勢確認 |
| Acceleration | 20% | 近 5 日 vs 前 5 日 flow 的加速度 | 動能轉折偵測 |
| Anomaly | 20% | 60 日 Z-score：`0.5 + z × 0.15` [0,1] | 異常大買超偵測 |

#### 效果評估
- **優點**：外資資訊優勢明顯，IC 通常為正（0.05–0.15）；11% 最高權重反映其實證重要性
- **缺點**：大型股偏重（外資交易集中前 50 大）；中小型股外資覆蓋不足時信號雜訊高
- **IC 預期**：5 日 IC 約 0.05–0.12（依體制波動）

#### 可優化方向
1. 加入「外資期貨未平倉」作為輔助信號，提升方向性判斷
2. 區分「外資自營」vs「外資投信」買賣超
3. 針對不同市值區間使用不同標準化基準

---

### F02 — technical_signal（技術訊號聚合）

**權重：8%｜分類：短期**

#### 理論基礎
技術分析透過價量型態辨識趨勢與轉折。本因子聚合多個經典指標而非依賴單一信號，降低假訊號風險。

#### 資料來源
- `signals["summary"]["raw_score"]`（系統內部技術分析引擎，0–5 分制）
- `df_tech["adx"]`（ADX 趨勢強度，`ta` library）
- `df_tech["close", "sma_5", "sma_20", "sma_60"]`（均線系統）
- `df_tech["obv"]`（OBV 量能確認）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Signal Score | 40% | `(raw/max + 1) / 2` 映射至 [0,1] | 技術面綜合評分 |
| ADX Strength | 20% | >40: 0.9 / 25-40: 0.7 / <25: 0.4 | 趨勢明確度 |
| MA Alignment | 20% | 5 級：完美多頭排列=1.0 → 空頭排列=0.0 | 均線排列狀態 |
| OBV Divergence | 20% | 價漲量縮=0.3 / 價跌量增=0.7 | 背離偵測 |

**MA 排列等級**：
- `close > sma5 > sma20 > sma60` → 1.0（完美多頭）
- `close > sma5 > sma20` → 0.8
- `close > sma20` → 0.65
- `close < sma5 < sma20 < sma60` → 0.0（完美空頭）

#### 效果評估
- **優點**：多指標聚合降低單一指標失效風險；ADX 過濾盤整期假訊號
- **缺點**：均線系統天然滯後；在震盪市效果衰減
- **IC 預期**：趨勢行情 IC 0.08–0.15；盤整期 IC 接近 0

#### 可優化方向
1. 加入 RSI 背離作為第 5 個子因子
2. ADX 閾值可根據 IC 回測動態調整
3. 加入 MACD 柱狀圖動能方向

---

### F03 — short_momentum（短期動能）

**權重：7%｜分類：短期**

#### 理論基礎
動量效應（Momentum Effect）是金融學中最穩健的異常之一。短期（1–5 日）動量捕捉近期價格慣性，同時用均值偏離度衡量超買超賣。

#### 資料來源
- `df["close"]`（每日收盤價）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Multi-Timeframe Returns | 60% | 1d(30%) + 3d(35%) + 5d(35%) returns | 多週期動量合成 |
| SMA Bias | 40% | `0.5 + (bias3×0.5 + bias5×0.5) × 8.0` [0,1] | 均值偏離度 |

- Return → Score 映射：`0.5 + ret × scale`，scale 分別為 10.0 / 5.0 / 4.0
- SMA Bias：`bias_n = (close - sma_n) / sma_n`

#### 效果評估
- **優點**：計算簡單、資料需求低、在趨勢市表現穩定
- **缺點**：反轉點訊號錯誤率高；scale 硬編碼缺乏自適應
- **IC 預期**：bull 行情 0.05–0.10；反轉期可能為負

#### 可優化方向
1. 加入「動量衰減率」（momentum decay）偵測趨勢末端
2. Scale 參數可根據波動率自適應（vol-adjusted momentum）
3. 結合成交量加權動量（VWAP momentum）

---

### F04 — trust_flow（投信籌碼流向）

**權重：5%｜分類：短期**

#### 理論基礎
投信代表國內專業法人，其買超行為反映基本面研究結論。投信連續買超通常伴隨中期股價上漲，尤其在中小型股。

#### 資料來源
- TWSE T86 API → `trust_info["trust_cumulative"]`（5 日累計）
- `trust_info["trust_consecutive_days"]`
- `df["trust_buy_sell"]`（FinMind 歷史資料）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Net Normalized | 40% | `0.5 + (net_5d / avg_vol_20d) × 0.25` [0,1] | 買超占比 |
| Consecutive Days | 30% | `min(0.5 + days × 0.1, 1.0)` | 連續天數 |
| Acceleration | 30% | 近 5d vs 前 5d 加速度 | 力道轉折 |

#### 效果評估
- **優點**：中小型股預測力強；投信買超持續性高
- **缺點**：投信有季底作帳效應（Q4 尤其明顯），產生噪音
- **IC 預期**：中小型股 IC 0.08–0.12；大型股偏低

#### 可優化方向
1. 加入季底效應時間衰減（11-12 月降低權重）
2. 區分主動型 vs 被動型投信

---

### F05 — volume_anomaly（量能異常）

**權重：4%｜分類：短期**

#### 理論基礎
成交量為價格的先行指標。異常放量搭配上漲為強勢確認；放量下跌則為風險訊號。量價一致性是趨勢健康度的重要衡量。

#### 資料來源
- `df["volume", "close"]`（每日成交量與收盤價）
- `df_tech["obv"]`（On-Balance Volume）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Volume Expansion | 50% | `vol_5d/vol_20d` 方向調整 | 放量方向 |
| Volume-Price Consistency | 30% | 5 日內量價同向天數 | 量價一致性 |
| OBV Trend | 20% | obv_5d vs obv_20d (±2% threshold) | OBV 趨勢確認 |

- Volume Expansion：上漲放量加分 `0.5 + (ratio-1)×0.3`；下跌放量減分

#### 效果評估
- **優點**：量價分析直覺且有效；OBV 提供累積確認
- **缺點**：除權息日量能異常會干擾；ETF 大額申贖影響成交量
- **IC 預期**：0.03–0.08（輔助性因子，獨立預測力有限）

#### 可優化方向
1. 過濾除權息日的異常量
2. 加入大單/散單比例（需券商 level2 資料）
3. Volume profile（價量分布）分析

---

### F06 — margin_sentiment（融資融券情緒）

**權重：4%｜分類：短期**

#### 理論基礎
融資為散戶槓桿指標，高融資使用率通常為反向指標——散戶過度樂觀時風險升高。融券（空方）增加反而可能是軋空動力。

#### 資料來源
- `df["margin_balance"]`（融資餘額，FinMind `TaiwanStockMarginPurchaseShortSale`）
- `df["short_balance"]`（融券餘額）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Margin Trend | 50% | `0.5 - margin_change × 3.0` [0,1] | **反向**：融資增=減分 |
| Margin Utilization | 30% | 融資/20 日最高：>0.8→0.2 / <0.3→0.7 | 使用率風險 |
| Short Ratio | 20% | short/margin：>0.20→0.7 / <0.10→0.5 | 軋空潛力 |

#### 效果評估
- **優點**：散戶逆向指標在台股有效，融資暴增後回檔機率高
- **缺點**：融資數據隔日公布，有一天延遲；部分股票融資餘額波動小
- **IC 預期**：bear 行情 IC 0.05–0.10；bull 行情效果有限

#### 可優化方向
1. 加入「融資維持率」作為強制平倉風險指標
2. 結合融資買進金額（非餘額變化）更精確

---

### F07 — trend_momentum（中期趨勢動能）

**權重：7%｜分類：中期**

#### 理論基礎
中期（20–60 日）動量效應學術文獻豐富（Jegadeesh & Titman 1993）。結合均線排列與 ADX 確認趨勢品質，過濾無方向性行情。

#### 資料來源
- `df["close"]`（60+ 天歷史）
- `df_tech["adx"]`

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Mid Returns | 40% | 20d(60%) + 60d(40%) returns | 中期報酬率 |
| MA Alignment | 30% | 同 F02 的 5 級均線排列 | 趨勢確認 |
| ADX Confirm | 30% | >40: 0.85 / 25-40: 0.65 / <25: 0.40 | 趨勢強度 |

- 20d return → score: `0.5 + ret × 3.0`
- 60d return → score: `0.5 + ret × 1.5`

#### 效果評估
- **優點**：中期動量 IC 穩定（0.05–0.10），學術驗證充分
- **缺點**：趨勢反轉時訊號嚴重落後；ADX 本身也有滯後
- **IC 預期**：bull/bear 趨勢行情 IC 0.08–0.12；sideways 衰減至 0.03

#### 可優化方向
1. 加入「52 週高低點距離」作為趨勢位置判斷
2. 中期動量可改用 risk-adjusted return（Sharpe-like）

---

### F08 — revenue_momentum（月營收動能）

**權重：4%｜分類：中期**

#### 理論基礎
台灣上市公司每月公布營收，為全球獨特的高頻基本面數據。營收 YoY 成長加速通常領先股價表現 1–3 個月。

#### 資料來源
- FinMind `TaiwanStockMonthRevenue`（需 13+ 個月歷史）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| YoY Growth | 50% | 分級：>30%→0.85 / 15-30%→0.72 / ... / <-15%→0.22 | 年增率等級 |
| YoY Accel | 30% | mean(近3月YoY) - mean(前3月YoY) | 成長加速度 |
| MoM | 20% | `0.5 + mom × 3.0` [0,1] | 月增率 |

#### 效果評估
- **優點**：台股獨有優勢；營收動能是中長期最有效的 alpha 因子之一
- **缺點**：月營收延遲公布（10 日前）；季節性（農曆年、假期）干擾 MoM
- **IC 預期**：20 日 IC 0.08–0.15（品質最高的因子之一）

#### 可優化方向
1. 加入「營收驚喜度」（actual vs consensus estimate）
2. MoM 加入季節性調整（SARIMA 或同期比較）
3. 區分本業 vs 業外收入

---

### F09 — institutional_sync（法人同步性）

**權重：4%｜分類：中期**

#### 理論基礎
三大法人（外資、投信、自營商）同步買超為極強的方向確認信號。法人與散戶逆向（法人買、融資減）更具預測力。

#### 資料來源
- `trust_info`（foreign/trust/dealer cumulative）
- `df["margin_balance"]`

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Foreign-Trust Sync | 40% | 同買=0.85 / 同賣=0.15 / 混合=0.50 | 雙法人共識 |
| 3-Institution Dir | 30% | 3 正=0.90 / 2 正=0.65 / 1 正=0.35 / 0 正=0.10 | 全法人方向 |
| Inst vs Retail | 30% | 法人買+融資減=0.85 / 法人賣+融資增=0.15 | 聰明錢 vs 散戶 |

#### 效果評估
- **優點**：法人共識信號穩定性高，極端值（全買/全賣）歷史勝率佳
- **缺點**：自營商含避險部位（選擇權對沖），方向信號有雜訊
- **IC 預期**：0.05–0.10；在三法人極度一致時 IC 提升

#### 可優化方向
1. 自營商拆分「自行買賣」vs「避險」部位
2. 加入法人「持有比例變化」（非僅買賣超）
3. 加權強度而非僅看方向

---

### F10 — volatility_regime（波動率狀態）

**權重：4%｜分類：中期**

#### 理論基礎
低波動溢價（Low Volatility Anomaly）：低波動股票長期表現優於高波動股票（Baker et al. 2011）。波動率壓縮（Bollinger Squeeze）常預示突破行情。

#### 資料來源
- `df["close"]`（計算歷史波動率）
- `df_tech["bb_width"]`（Bollinger Band 寬度）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Low Vol Premium | 40% | ann_vol: <20%→1.0 / 20-30%→0.7 / ... / >55%→0.0 | 低波動溢價 |
| Vol Compress | 30% | vol_5d/vol_20d: <0.7→0.65 / >1.5→0.30 | 壓縮=機會 |
| BB Percentile | 30% | 60 日歷史分位數：`1.0 - percentile` | 布林帶位置 |

- 年化波動率：`std(daily_returns[-20:]) × √252`

#### 效果評估
- **優點**：低波動溢價在台股有效；波動壓縮為突破前兆
- **缺點**：低波動不代表安全（可能是流動性不足）；BB 百分位受極端值影響
- **IC 預期**：sideways 行情 IC 0.05–0.08；趨勢行情較低

#### 可優化方向
1. 使用 GARCH 或 EWMA 替代簡單標準差
2. 加入 implied volatility（選擇權隱含波動率）
3. 波動率壓縮方向結合均線判斷突破方向

---

### F11 — news_sentiment（新聞情緒）

**權重：3%｜分類：中期**

#### 理論基礎
新聞情緒對短期價格有催化作用。來源加權降低低品質來源噪音，情緒動量捕捉輿論轉向。

#### 資料來源
- PTT 股票板（HTML 解析 + Firecrawl fallback）
- 鉅亨網 cnyes API（JSON + HTML fallback）
- Google News RSS
- Yahoo Finance TW（HTML 解析）
- LLM（Claude Haiku）進行情緒標註

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Source-Weighted | 40% | cnyes(0.40) + yahoo(0.25) + google(0.20) + ptt(0.15) | 來源可信度加權 |
| Sentiment Momentum | 30% | recent_half vs early_half: `0.5 + delta × 1.5` | 情緒轉向 |
| Engagement Anomaly | 30% | 高互動 + 正面=0.7 / 高互動 + 負面=0.3 | 互動量放大效果 |

- Freshness：`max(0, 1 - days_old / 14)`（14 天衰減）

#### 效果評估
- **優點**：多來源交叉驗證降低假新聞影響；LLM 情緒標註品質優於關鍵字
- **缺點**：爬蟲穩定性風險（網站改版）；PTT 噪音高；情緒分析有延遲
- **IC 預期**：0.02–0.06（輔助因子，獨立 IC 偏低）

#### 可優化方向
1. 加入 Threads/X 等社群平台擴大覆蓋
2. 情緒分數增加「事件分類」（業績、法規、產品）
3. 新聞量異常（突然爆量）作為獨立信號

---

### F12 — global_context（國際市場連動）

**權重：3%｜分類：中期**

#### 理論基礎
台股高度外銷導向，與半導體供應鏈（SOX、TSM）及台灣 ETF（EWT）高度相關。產業別差異化權重提升精確度。

#### 資料來源
- yfinance：SOX（費城半導體指數）、TSM（台積電 ADR）、ASML、EWT（iShares MSCI Taiwan ETF）
- 每日快取

#### 實作細節

**產業別權重矩陣 SECTOR_GLOBAL_WEIGHTS**：

| 產業 | SOX | TSM | ASML | EWT |
|------|-----|-----|------|-----|
| semiconductor | 40% | 25% | 20% | 15% |
| electronics | 25% | 20% | 10% | 45% |
| finance | 10% | 10% | 5% | 75% |
| default | 30% | 20% | 15% | 35% |

- Return → Score：`0.5 + return × 17.5` [0.05, 0.95]
- EWT Relative：`0.5 + (ewt - sox) × 10.0`

#### 效果評估
- **優點**：半導體股產業連動捕捉精準；產業差異化設計合理
- **缺點**：只用 1 日報酬雜訊高；非半導體產業僅靠 EWT
- **IC 預期**：半導體股 IC 0.05–0.10；傳產/金融 IC 接近 0

#### 可優化方向
1. 加入多天期報酬（3d/5d）降低日間噪音
2. 金融股加入美國金融 ETF（XLF）
3. 航運股加入 BDI（波羅的海乾散貨指數）

---

### F13 — margin_quality（季報毛利率趨勢）

**權重：4%｜分類：中期**

#### 理論基礎
毛利率擴張反映企業定價能力提升或成本控制改善，是盈利品質最核心的指標之一。趨勢比絕對值更重要。

#### 資料來源
- yfinance `quarterly_income_stmt`（主要：Gross Profit, Total Revenue, Operating Income）
- yfinance `ticker.info` fallback（grossMargins, operatingMargins）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| GM Trend | 60% | `0.5 + (QoQ×8 + YoY×4)` [0,1] | 毛利率 QoQ + YoY 趨勢 |
| OM Level | 40% | >20%→0.85 / 10-20%→0.70 / <0%→0.25 | 營益率等級 |

- Freshness：0.6（季報更新頻率低）

#### 效果評估
- **優點**：盈利品質指標穩定，長期 IC 優
- **缺點**：季報延遲（公布後 1-2 個月的舊數據）；yfinance 資料有時缺失
- **IC 預期**：20 日 IC 0.05–0.10（中長期因子）

#### 可優化方向
1. 加入 ROE/ROA 趨勢作為輔助
2. 使用台灣本地財報 API（公開資訊觀測站）替代 yfinance
3. 加入「毛利率 vs 同業比較」的相對排名

---

### F14 — sector_rotation（產業資金輪動）

**權重：3%｜分類：中期**

#### 理論基礎
資金在產業間流動反映市場對景氣循環的預期。資金流入的產業通常未來表現優於流出產業（sector momentum）。

#### 資料來源
- `_compute_sector_aggregates()`（法人買賣超 + 報酬率）
- TWSE 產業指數（`twse_scanner.fetch_industry_indices()`）
- `STOCK_SECTOR` 映射（~80 檔股票 → 8 產業）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Sector Flow vs Market | 35% | `0.5 + flow_diff / |market_flow| × 0.3` | 資金流向相對強度 |
| Industry Index Momentum | 30% | `0.5 + (sector_idx - market_idx) × 12.0` | 產業指數動量 |
| Relative Return | 20% | `0.5 + ret_diff × 5.0` | 20 日相對報酬 |
| Breadth | 15% | positive_flow_stocks / total_stocks | 資金廣度 |

#### 效果評估
- **優點**：產業輪動在台股有效（半導體 vs 傳產 vs 金融輪替明顯）
- **缺點**：`STOCK_SECTOR` 僅覆蓋 ~80 檔，未映射的股票無此因子
- **IC 預期**：0.03–0.08（依產業週期波動大）

#### 可優化方向
1. 擴充 `STOCK_SECTOR` 到 200+ 檔覆蓋全市場
2. 使用 GICS 或 TWSE 官方分類替代手工映射
3. 加入產業 PE ratio 相對排名

---

### F15 — ml_ensemble（ML 集成預測）

**權重：7%｜分類：長期**

#### 理論基礎
機器學習模型能捕捉非線性交互作用，Stacking Ensemble（LSTM + XGBoost + Ridge meta-learner）整合多模型優勢。

#### 資料來源
- `ml_scores[stock_id]`（StackingEnsemble 預測輸出，已在 [0,1] 區間）
- 43 特徵輸入（技術指標 + 籌碼 + 宏觀）

#### 實作細節
- 直接使用模型輸出，無額外轉換
- 模型需通過 3 質量門檻：direction_acc > 52%, MSE < naive, PBO < 0.6
- 模型不可用時 `available=False`，權重重分配給其他因子

#### 效果評估
- **優點**：捕捉非線性交互；CPCV 交叉驗證控制過擬合
- **缺點**：模型需定期重訓練；黑盒不透明；依賴特徵品質
- **IC 預期**：0.05–0.12（取決於訓練品質與市場體制）

#### 可優化方向
1. 加入 online learning（增量更新）減少重訓頻率
2. 輸出校準（Platt scaling）確保機率解釋正確
3. 加入模型不確定性估計（ensemble disagreement）

---

### F16 — fundamental_value（基本面價值）

**權重：6%｜分類：長期**

#### 理論基礎
價值投資（Value Investing）：低 PE、低 PB、高 ROE、高殖利率的股票長期表現優於市場。

#### 資料來源
- FinMind `TaiwanStockPER`（PE, PBR, dividend_yield）— 優先
- yfinance `ticker.info`（trailingPE, returnOnEquity, dividendYield）— fallback

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| P/E Risk | 30% | >80→0.30 / 50-80→0.38 / <0→0.30 / else→0.50 | 本益比風險 |
| P/B Reversion | 20% | <1.0→0.75 / 1-1.5→0.60 / >3.0→0.25 | 股價淨值比 |
| ROE | 25% | >25%→0.85 / 15-25%→0.72 / <0→0.25 | 股東權益報酬 |
| Dividend Yield | 25% | >6%→0.80 / 4-6%→0.68 / 0→0.40 | 殖利率 |

#### 效果評估
- **優點**：長期 IC 穩定；value factor 在台股有效（尤其高殖利率策略）
- **缺點**：短期內價值陷阱（value trap）風險；PE 對虧損公司無意義
- **IC 預期**：20 日 IC 0.03–0.08；年度 IC 更高

#### 可優化方向
1. 加入「相對 PE」（vs 產業中位數）替代絕對 PE
2. 加入 EV/EBITDA 等企業價值指標
3. 結合 earnings quality（應計項目）過濾價值陷阱

---

### F17 — liquidity_quality（流動性品質）

**權重：4%｜分類：長期**

#### 理論基礎
流動性溢價（Liquidity Premium）：高流動性降低交易成本，減少被迫平倉風險。穩定的流動性品質比偶發大量更重要。

#### 資料來源
- `df["volume", "high", "low", "close"]`（20+ 天歷史）

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| Avg Volume | 50% | `avg_vol_20d / 5000` [0,1] | 日均量（5000 張=滿分） |
| Vol Stability | 25% | CV: <0.3→0.9 / 0.3-0.5→0.7 / >0.8→0.3 | 量能穩定性 |
| Spread Proxy | 25% | (H-L)/C: <1%→0.9 / 1-2%→0.7 / >4%→0.3 | 價差代理 |

#### 效果評估
- **優點**：流動性風險過濾有效；避免推薦無法執行的低量股
- **缺點**：日均量標準 5000 張對中小型股過嚴；高低價差不等於 bid-ask spread
- **IC 預期**：獨立 IC 低（0.01–0.04），但作為風險過濾器有價值

#### 可優化方向
1. 使用 Amihud illiquidity ratio 替代簡單均量
2. 加入逐筆成交資料計算真實買賣價差
3. 日均量基準依市值分級（大/中/小型股不同標準）

---

### F18 — macro_risk（宏觀風險環境）

**權重：4%｜分類：長期**

#### 理論基礎
宏觀因子（VIX、殖利率曲線、匯率）影響系統性風險。殖利率曲線倒掛歷史上領先衰退；VIX 為恐慌指標；銅價為景氣領先指標。

#### 資料來源
- yfinance：^VIX, ^TNX（10Y yield）, USDTWD=X, HG=F（銅期貨）
- 衍生計算：yield curve spread（10Y-5Y）、trend changes

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| VIX | 30% | <15→0.80 / 15-20→0.65 / 20-25→0.45 / >30→0.15 | 恐慌指數 |
| Yield Curve | 20% | 10Y-5Y spread + 方向調整 | 殖利率曲線 |
| USD/TWD | 20% | `0.5 - fx_trend × 15.0` [0.1, 0.9] | 匯率趨勢 |
| 10Y Change | 15% | `0.5 - tnx_change × 1.5` | 長債利率變化 |
| Copper | 15% | `0.5 + copper_ret × 4.0` | 景氣循環指標 |

#### 效果評估
- **優點**：系統性風險預警有效；多維度宏觀覆蓋
- **缺點**：所有股票共用同一宏觀分數（無個股差異化）；VIX 為美國指標
- **IC 預期**：截面 IC 接近 0（因為所有股票同分），但作為風險調節器有效

#### 可優化方向
1. 加入「宏觀 beta」讓個股對宏觀敏感度不同
2. 加入台灣本地指標：景氣燈號、PMI
3. 加入中國/日本市場指標（台股受中國需求影響大）

---

### F19 — export_momentum（台灣出口動能）

**權重：4%｜分類：長期**

#### 理論基礎
台灣 GDP 出口占比超過 60%，出口動能直接反映經濟基本面。EWT（台灣 ETF）整合海外投資人對台灣經濟的預期。

#### 資料來源
- yfinance：EWT（1d/20d/60d return）, 0050.TW（元大台灣 50）, SOX
- 每日快取

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| 20d Export | 40% | `0.5 + ret × 4.0` [0,1] | EWT 20 日動能 |
| 60d Export | 25% | `0.5 + ret × 2.0` [0,1] | EWT 60 日趨勢 |
| TW50 Momentum | 15% | `0.5 + tw50_ret × 4.0` [0,1] | 本地 ETF 動量 |
| EWT vs SOX | 20% | `0.5 + relative × 10.0` [0,1] | 台灣 vs 半導體相對強弱 |

#### 效果評估
- **優點**：台灣出口與股市高度相關；EWT 即時反映海外預期
- **缺點**：EWT 成分以大型股為主，對中小型股預測力弱；同 macro_risk 缺乏個股差異化
- **IC 預期**：截面 IC 低（同質性問題），但大盤方向預測力佳

#### 可優化方向
1. 使用台灣海關出口實際數據（月度）替代 ETF 代理
2. 加入產業別出口數據（半導體 vs 石化 vs 機械出口）
3. 個股化：依出口營收占比加權

---

### F20 — us_manufacturing（美國製造業景氣）

**權重：4%｜分類：長期**

#### 理論基礎
美國製造業活動（ISM PMI 代理）影響全球供應鏈需求。XLI/SPY 比率反映製造業相對景氣。

#### 資料來源
- yfinance：XLI（Industrial Select Sector ETF）、SPY（S&P 500 ETF）
- 計算：XLI 20d return, XLI vs SMA200, XLI/SPY ratio trend

#### 實作細節

| 子因子 | 權重 | 公式 | 說明 |
|--------|------|------|------|
| XLI 20d Return | 40% | `0.5 + ret × 4.0` [0,1] | 製造業動量 |
| XLI/SPY Ratio | 40% | `0.5 + trend × 8.0` [0,1] | 相對強弱趨勢 |
| XLI vs SMA200 | 20% | >5%→0.75 / 0-5%→0.60 / <-5%→0.25 | 長期趨勢位置 |

#### 效果評估
- **優點**：供應鏈傳導邏輯清晰；XLI/SPY ratio 有學術支持
- **缺點**：台灣→美國製造業的傳導有延遲（1-3 個月）；非製造業股票關聯弱
- **IC 預期**：截面 IC 低；作為體制判斷輔助

#### 可優化方向
1. 使用真正的 ISM PMI 數據替代 ETF 代理
2. 依產業差異化：半導體看 SOX 而非 XLI
3. 加入中國 PMI 作為亞洲需求指標

---

## 3. 體制自適應權重（REGIME_MULTIPLIERS）

### 3.1 設計邏輯

HMM（Hidden Markov Model）偵測 3 種市場體制，每種體制對 20 因子權重進行乘數調整：

### 3.2 完整乘數表

| 因子 | Bull | Bear | Sideways |
|------|------|------|----------|
| foreign_flow | 1.0 | **1.3** | 1.0 |
| technical_signal | **1.1** | 0.8 | **1.3** |
| short_momentum | **1.3** | **0.5** | 0.8 |
| trust_flow | 1.0 | **1.2** | 1.0 |
| volume_anomaly | 1.0 | 1.0 | **1.2** |
| margin_sentiment | 0.8 | **1.5** | 1.0 |
| trend_momentum | **1.3** | **0.5** | **0.7** |
| revenue_momentum | 1.0 | 1.0 | 1.0 |
| institutional_sync | 1.0 | 1.0 | 1.0 |
| volatility_regime | **0.7** | **1.5** | **1.2** |
| news_sentiment | 0.8 | 1.0 | **1.2** |
| global_context | 1.0 | **1.2** | 1.0 |
| margin_quality | 0.8 | **1.2** | 1.0 |
| sector_rotation | **1.2** | 0.8 | **1.3** |
| ml_ensemble | 1.0 | 0.8 | 1.0 |
| fundamental_value | 0.8 | **1.2** | 1.0 |
| liquidity_quality | 0.8 | **1.3** | 1.0 |
| macro_risk | 0.8 | **1.3** | 1.0 |
| export_momentum | 1.0 | **1.2** | 1.0 |
| us_manufacturing | 0.8 | **1.3** | 1.0 |

### 3.3 設計原則

- **Bull**：提升動能因子（momentum, trend）+ 輪動因子；壓抑防禦因子（volatility, fundamental）
- **Bear**：提升防禦因子（margin_sentiment, volatility, liquidity, macro）+ 外資流向；壓抑動能因子
- **Sideways**：提升技術/量能/輪動因子（尋找方向突破信號）；壓抑趨勢因子

### 3.4 可優化方向

1. 從固定乘數改為 IC-based 動態調整（用過去 60 日 IC 排名決定權重）
2. 加入體制轉換期的漸進調整（目前是硬切換）
3. 加入更多體制（如 crash、recovery）

---

## 4. 信心度計算

### 4.1 公式

```
raw_confidence = agreement × 0.30 + strength × 0.30 + coverage × 0.25 + freshness × 0.15
final_confidence = raw_confidence × risk_discount
```

### 4.2 組成

| 組成 | 權重 | 計算方式 | 說明 |
|------|------|---------|------|
| Agreement | 30% | 同方向因子佔比 | 因子共識度 |
| Strength | 30% | `abs(total_score - 0.5) × 2` | 信號強度 |
| Coverage | 25% | `sum(BASE_WEIGHT for available factors)` | 資料完整度 |
| Freshness | 15% | 加權平均 factor freshness | 資料時效性 |

### 4.3 風險折扣

| 條件 | 折扣 |
|------|------|
| 年化波動率 > 60% | ×0.70 |
| 年化波動率 40–60% | ×0.85 |
| 日均量 < 200 張 | ×0.60 |
| 日均量 200–500 張 | ×0.80 |
| 融資 5 日增 > 10% | ×0.85 |
| PE > 80 或 PE < 0 | ×0.75 |
| PE > 50 | ×0.85 |

- 多重折扣可疊加
- **下限（floor）：0.30**

### 4.4 可優化方向

1. Agreement 改用因子 IC 加權（高 IC 因子的同意更有價值）
2. 加入「模型預測一致性」（LSTM vs XGBoost 的分歧度）
3. 風險折扣參數可根據歷史數據校準

---

## 5. 缺資料重分配機制

### 5.1 觸發條件

當任何因子的 `FactorResult.available = False` 時，其 BASE_WEIGHT 不會「消失」，而是按比例重分配給其他可用因子。

### 5.2 演算法

```python
# _compute_weights(factors, regime)
1. 對所有 20 因子取 BASE_WEIGHT
2. 乘以 REGIME_MULTIPLIERS[regime]
3. 移除 available=False 的因子
4. 剩餘因子權重重新正規化至 sum=1.0
```

### 5.3 範例

假設 `ml_ensemble` (7%) 和 `margin_quality` (4%) 不可用：

- 可用因子原始權重和 = 1.0 - 0.07 - 0.04 = 0.89
- 每個可用因子新權重 = 原權重 / 0.89
- foreign_flow: 0.11 / 0.89 = 0.1236（12.4%）

### 5.4 設計考量

- **優點**：保持總權重恆等於 1.0；不浪費信號預算
- **缺點**：大量缺資料時，少數因子被過度放大
- **改進建議**：設定單因子最大權重上限（如 20%）避免過度集中

---

## 6. 因子 IC 追蹤系統

### 6.1 架構

```
Day 0 15:00  市場掃描 → save_factor_ic_records()
             [record_date, stock_id, factor_name, factor_score, forward_return=NULL]
             ↓
Day 8 16:00  backfill_forward_returns(5)
             → 查詢 StockPrice 計算 5 日遠期報酬
             [forward_return_5d = (future_close - base_close) / base_close]
             ↓
Day 28 16:00 → 同時回填 forward_return_20d
             ↓
API 查詢     get_factor_ic_rolling("foreign_flow", window=60)
             → Spearman rank correlation(score_vector, return_vector) per day
             → Rolling mean/std/ICIR
```

### 6.2 Spearman IC 計算

```python
# 每日截面
for date in sorted_dates:
    pairs = date_groups[date]  # [(factor_score, forward_return), ...]
    if len(pairs) >= 5:        # 最少 5 檔股票
        ic = spearmanr(scores, returns)  # 秩相關係數
```

- **IC > 0**：因子分數越高 → 未來報酬越高（因子有效）
- **ICIR > 1.0**：因子 IC 穩定且高效

### 6.3 資料模型

```
FactorICRecord:
  id (PK), record_date (Index), stock_id (Index),
  factor_name, factor_score, forward_return_5d, forward_return_20d, created_at
  UniqueConstraint: (record_date, stock_id, factor_name)
```

### 6.4 API 端點

```
GET /api/v1/market/factor-ic?factor={name}&window={days}
→ { factor, ic_mean, ic_std, icir, ic_series: [{date, ic}] }
```

### 6.5 排程

| 時間 | 任務 |
|------|------|
| 15:00 | 市場掃描 → save_factor_ic_records() |
| 15:30 | 深度分析 pipeline |
| 16:00 | backfill_forward_returns() |

### 6.6 可優化方向

1. 加入 IC decay 分析（5d vs 10d vs 20d IC 衰減速率）
2. 加入 turnover-adjusted IC（考慮因子換手率）
3. 基於滾動 IC 自動調整 BASE_WEIGHTS（adaptive weights）
4. 加入 IC 統計顯著性檢定（t-test）

---

## 7. 整體可優化方向

### 7.1 系統架構層級

| 改進項目 | 說明 | 優先級 | 難度 |
|----------|------|--------|------|
| **IC-Adaptive Weights** | 根據滾動 IC 動態調整 BASE_WEIGHTS | 高 | 中 |
| **因子正交化** | PCA 或 Gram-Schmidt 消除因子間共線性 | 高 | 中 |
| **非線性合成** | 從加權平均改為 GBM/NN 合成因子 | 中 | 高 |
| **產業差異化權重** | 半導體 vs 金融 vs 傳產使用不同 BASE_WEIGHTS | 中 | 低 |
| **因子衰減模型** | 不同因子用不同 half-life decay | 中 | 低 |

### 7.2 資料層級

| 改進項目 | 說明 | 優先級 | 難度 |
|----------|------|--------|------|
| **本地財報 API** | 用公開資訊觀測站替代 yfinance 季報 | 高 | 中 |
| **逐筆成交** | 接入 Shioaji 取得 Level2 委託簿 | 中 | 高 |
| **選擇權資料** | PC ratio + implied vol 作為新因子 | 中 | 中 |
| **產業擴充** | `STOCK_SECTOR` 從 80 檔擴充至全市場 | 高 | 低 |
| **中國指標** | 上證指數、中國 PMI 作為需求指標 | 低 | 低 |

### 7.3 風控層級

| 改進項目 | 說明 | 優先級 | 難度 |
|----------|------|--------|------|
| **單因子權重上限** | 缺資料重分配後設 max 20% cap | 高 | 低 |
| **因子 IC 監控告警** | IC 持續為負時自動降權或停用 | 高 | 中 |
| **信心度校準** | Calibration plot 確認信心度與實際準確率對齊 | 中 | 中 |
| **宏觀因子個股化** | macro_risk/export_momentum 依 beta 差異化 | 中 | 中 |

### 7.4 已知問題

1. **截面同質性**：F18/F19/F20 所有股票同分，截面 IC 接近 0，但佔 12% 權重
2. **季報延遲**：F13 freshness=0.6 但仍佔 4%，實際上已過時的資訊
3. **STOCK_SECTOR 覆蓋不足**：未映射的股票 F14 不可用
4. **硬編碼閾值**：所有子因子的分級閾值都是手動設定，缺乏數據驅動校準
5. **乘數未正規化**：REGIME_MULTIPLIERS 乘完後才正規化，極端情況可能讓單因子過重

---

## 附錄 A：完整權重表

| # | 因子 | BASE_WEIGHT | Bull | Bear | Sideways |
|---|------|-------------|------|------|----------|
| 1 | foreign_flow | 0.11 | 0.11 | 0.143 | 0.11 |
| 2 | technical_signal | 0.08 | 0.088 | 0.064 | 0.104 |
| 3 | short_momentum | 0.07 | 0.091 | 0.035 | 0.056 |
| 4 | trust_flow | 0.05 | 0.05 | 0.06 | 0.05 |
| 5 | volume_anomaly | 0.04 | 0.04 | 0.04 | 0.048 |
| 6 | margin_sentiment | 0.04 | 0.032 | 0.06 | 0.04 |
| 7 | trend_momentum | 0.07 | 0.091 | 0.035 | 0.049 |
| 8 | revenue_momentum | 0.04 | 0.04 | 0.04 | 0.04 |
| 9 | institutional_sync | 0.04 | 0.04 | 0.04 | 0.04 |
| 10 | volatility_regime | 0.04 | 0.028 | 0.06 | 0.048 |
| 11 | news_sentiment | 0.03 | 0.024 | 0.03 | 0.036 |
| 12 | global_context | 0.03 | 0.03 | 0.036 | 0.03 |
| 13 | margin_quality | 0.04 | 0.032 | 0.048 | 0.04 |
| 14 | sector_rotation | 0.03 | 0.036 | 0.024 | 0.039 |
| 15 | ml_ensemble | 0.07 | 0.07 | 0.056 | 0.07 |
| 16 | fundamental_value | 0.06 | 0.048 | 0.072 | 0.06 |
| 17 | liquidity_quality | 0.04 | 0.032 | 0.052 | 0.04 |
| 18 | macro_risk | 0.04 | 0.032 | 0.052 | 0.04 |
| 19 | export_momentum | 0.04 | 0.04 | 0.048 | 0.04 |
| 20 | us_manufacturing | 0.04 | 0.032 | 0.052 | 0.04 |

> 注：Bull/Bear/Sideways 欄為 BASE × MULTIPLIER（未正規化），實際使用時會正規化至 sum=1.0。

## 附錄 B：資料來源總覽

| 資料來源 | 因子 | 速率限制 | 重試 | 快取 |
|----------|------|----------|------|------|
| FinMind API | F01,F04,F08,F16 | 2 req/s | 3× | 無 |
| TWSE T86 | F01,F04,F09,F14 | 1 req/s | 2× | 5 min TTL |
| yfinance | F12,F13,F16,F18,F19,F20 | 無限制 | 3× | 每日 |
| Sentiment Crawler | F11 | 5 workers | 2× | 無 |
| TA Library | F02,F05,F07,F10 | N/A | N/A | N/A |
| Price DataFrame | F03,F05,F06,F17 | N/A | N/A | N/A |
| ML Model | F15 | N/A | N/A | N/A |
