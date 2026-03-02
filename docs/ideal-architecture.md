# 台股 AI 量化交易系統 — 理想專案架構書

> 結合 2025-2026 最新論文、業界共識、與實戰經驗的終極設計藍圖

**版本**：v1.0 | **日期**：2026-02-26

---

## 目錄

1. [設計哲學](#一設計哲學)
2. [殘酷的真相：研究告訴我們什麼](#二殘酷的真相研究告訴我們什麼)
3. [理想架構全貌](#三理想架構全貌)
4. [Layer 1：資料層](#四layer-1資料層)
5. [Layer 2：標籤工程](#五layer-2標籤工程)
6. [Layer 3：特徵工程](#六layer-3特徵工程)
7. [Layer 4：ML 模型層](#七layer-4ml-模型層)
8. [Layer 5：市場狀態偵測](#八layer-5市場狀態偵測)
9. [Layer 6：信號合成與規則引擎](#九layer-6信號合成與規則引擎)
10. [Layer 7：LLM Agent 層（顧問角色）](#十layer-7llm-agent-層顧問角色)
11. [Layer 8：硬性風控](#十一layer-8硬性風控)
12. [Layer 9：執行與監控](#十二layer-9執行與監控)
13. [Layer 10：回饋與記憶](#十三layer-10回饋與記憶)
14. [驗證框架](#十四驗證框架)
15. [台股特殊考量](#十五台股特殊考量)
16. [績效期望值](#十六績效期望值)
17. [技術棧建議](#十七技術棧建議)
18. [開發路線圖](#十八開發路線圖)
19. [附錄：論文索引](#十九附錄論文索引)

---

## 一、設計哲學

### 三條鐵律

```
1. LLM 是研究助手，不是交易員
2. 每增加一個自由度，就多一條過擬合的路
3. 不信任任何單次回測結果
```

### 核心原則

| 原則 | 來源 | 說明 |
|------|------|------|
| **ML 做預測，規則做風控** | FINSABER, Two Sigma | LLM 不擅長發明 alpha，但擅長加速研究 |
| **何時不交易比交易什麼重要** | HMM 研究, Renaissance | 過濾壞時機比選好標的更有效 |
| **簡單打敗複雜** | LLM Quant 年度回顧 | 三行相似的程式碼比一個過早的抽象好 |
| **寧願少賺也不要多虧** | 1/4 Kelly | 倉位管理是唯一能控制的變數 |
| **對自己的數字誠實** | 2602.14233 偏誤審查 | 實盤通常比回測差 30-50% |

### 來自業界的智慧

> "LLMs are NOT great at inventing alpha — they remove friction around the work that surrounds it."
> — Ilya Navogitsyn, Dataconomy 2026

> "Each added degree of freedom becomes another way to overfit."
> — LLM Quant 2025 年度回顧

> "Each of us must remain a watchful supervisor."
> — Two Sigma 2026 AI Outlook

> Renaissance Technologies 的 Medallion Fund 只對了 **50.75%** 的時候，但靠著每年執行數百萬筆不相關的小賭注，達到了 66.1% 的年化毛報酬率。

---

## 二、殘酷的真相：研究告訴我們什麼

### LLM 交易的壞消息（佔主導地位）

| 論文 | 年份 | 關鍵發現 |
|------|------|---------|
| **FINSABER** (KDD 2026) | 2025 | 20 年回測、100+ 股票：LLM 策略的優勢在更廣股票和更長期限下**完全消失**。FinMem 產生**負 alpha** (-1.34%)。 |
| **StockBench** | 2025 | 最好的 LLM (Kimi-K2) 僅 +1.9%，GPT-5 僅 +0.3%，幾乎等同 buy-and-hold (+0.4%)。Sortino ratio 全面偏低。 |
| **TradeTrap** | 2025 | LLM agent 記憶攻擊成功率極高：狀態竄改導致 **61% 總虧損**、91.97% 最大回撤。小擾動會在決策迴路中級聯放大。 |
| **偏誤審查** | 2026 | 164 篇論文中沒有任何一種偏誤被超過 28% 的論文討論。74% 研究者認為評估工具「稀缺」或「不存在」。 |

**FINSABER 的具體數據：**

| 股票 | 策略 | Sharpe | 年化報酬 | 最大回撤 |
|------|------|--------|---------|---------|
| TSLA | FinMem | 0.641 | 42.15% | -34.23% |
| TSLA | Buy & Hold | **0.630** | 37.77% | -50.84% |
| NFLX | FinMem | 0.293 | 12.57% | -27.72% |
| NFLX | Buy & Hold | **0.622** | 23.92% | -48.12% |

> 結論：FinMem 在 TSLA 上的 Sharpe 僅微幅勝出 0.011，在 NFLX 上慘輸。
> 當擴大到 63-91 檔股票的複合評估時，Buy & Hold 的 0.703 Sharpe 完勝 FinAgent 的 0.241。

**StockBench 完整排行榜（2025 年 3-6 月）：**

| 排名 | 模型 | 報酬率 | Sortino |
|------|------|--------|---------|
| 1 | Kimi-K2 | +1.9% | 0.0420 |
| 7 | Claude-4-Sonnet | +2.2% | 0.0245 |
| 9 | GPT-5 | +0.3% | 0.0132 |
| 12 | **Buy & Hold** | **+0.4%** | **0.0155** |

### 好消息（少但重要）

| 方法 | 來源 | 成果 |
|------|------|------|
| **混合系統** | ComSIA 2026 | 技術+ML+情緒+狀態偵測+ATR 倉位：**Sharpe 1.68**、135% 回報、-15.6% 最大回撤 |
| **HMM 狀態切換** | Multi-Model Ensemble-HMM (2025) | 作為交易過濾器使用，消除不利行情的虧損交易 |
| **情緒分析** | 多項研究 | LLM 最被驗證的金融應用：FinBERT 81-90% 情緒準確率 |
| **TFT** | 多項研究 (2025) | MAE 降低 40-50%，MAPE < 2%，優於 LSTM |
| **Triple Barrier + Meta-Labeling** | AFML (2018) + 後續 | 結合信心度的倉位管理，提升精確率 |

### 業界共識

```
2026 不是「LLM 做交易」的時代，
而是「AI 成為量化研究作業系統」的時代。

— Two Sigma 2026 展望
```

---

## 三、理想架構全貌

```
╔══════════════════════════════════════════════════════════════════╗
║  Layer 1: 資料層                                                  ║
║  FinMind API → OHLCV、法人、融資融券                                ║
║  Firecrawl   → PTT、鉅亨網新聞                                     ║
║  Claude Haiku → 結構化情緒提取                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 2: 標籤工程                                                  ║
║  Triple Barrier (ATR 動態障礙) + Meta-Labeling 信心度               ║
║  取代 naive pct_change — 這是一切可信度的基礎                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 3: 特徵工程                                                  ║
║  43 維原始 → SHAP/MI 篩選至 15-20 維                                ║
║  共線移除 (r > 0.95) + 行情相依特徵重要性監控                         ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 4: ML 模型層                                                 ║
║  ┌─────────┐  ┌──────────┐  ┌─────┐                              ║
║  │  LSTM   │  │ XGBoost  │  │ TFT │   ← 三模型各有所長             ║
║  │Attention│  │(樣本權重) │  │     │                               ║
║  └────┬────┘  └────┬─────┘  └──┬──┘                              ║
║       └────────────┼───────────┘                                  ║
║                    ▼                                               ║
║         Stacking Ensemble (Ridge meta-learner)                    ║
║         + Purged Walk-Forward CV (purging + embargo)              ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 5: 市場狀態偵測                                              ║
║  HMM 3-State (Bull / Sideways / Bear)                             ║
║  觀測 = [daily_return, realized_vol]                              ║
║  → 動態權重 + 交易過濾器（核心價值：何時不交易）                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 6: 信號合成 + 規則引擎                                       ║
║  ML 信號 (70%) + Agent 建議 (30%)                                  ║
║  + HMM 狀態縮放 + 閾值決策                                         ║
║  + Meta-Label 信心度 → 倉位大小                                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 7: LLM Agent 層（顧問角色，非決策者）                         ║
║  4 分析師 → 研究員辯論 → 建議                                       ║
║  唯一可信用途：情緒提取、研究加速、模式辨識                            ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 8: 硬性風控（LLM 無法覆蓋）                                  ║
║  1/4 Kelly │ ATR Trailing Stop │ 15% 回撤熔斷                     ║
║  20% 單一倉位上限 │ 8% 停損 │ 2:1 風險報酬比                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 9: 執行與監控                                                ║
║  APScheduler 排程 │ LINE/Telegram 通知 │ 台股成本模型               ║
╠══════════════════════════════════════════════════════════════════╣
║  Layer 10: 回饋與記憶                                               ║
║  三層記憶 (FinMem) + 層級轉移                                       ║
║  交易結果 → 驗證 → 升級/降級模式                                    ║
║  模型定期再訓練 + 特徵重要性漂移偵測                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 四、Layer 1：資料層

### 資料來源矩陣

| 來源 | 資料類型 | 頻率 | 延遲 | API |
|------|---------|------|------|-----|
| FinMind | OHLCV、三大法人買賣超、融資融券、營收 | 日 | T+0 盤後 | REST |
| PTT 股票板 | 社群貼文（標題 + 內文） | 即時 | 分鐘級 | 爬蟲 |
| 鉅亨網 | 財經新聞 | 即時 | 分鐘級 | 爬蟲 |
| 公開資訊觀測站 | 財報、重大訊息 | 季/即時 | T+0 | 爬蟲 |
| 台灣期交所 | 期貨未平倉、選擇權 Put/Call Ratio | 日 | T+0 | REST |

### 資料品質管線

```python
Raw Data → 缺值偵測 → 異常值過濾 → 時間對齊 → 存入 DB
                                          │
                                          ├── 存活者偏誤修正：
                                          │   保留已下市/被併購公司的歷史資料
                                          │   (2602.14233 偏誤審查要求)
                                          │
                                          └── 時間戳記驗證：
                                              確保資料有 "as-of" 時間戳
                                              防止前視偏誤
```

### 關鍵設計原則

1. **存活者偏誤修正**：維護完整的歷史股票宇宙，包含已下市/被併購的公司
2. **時間戳記嚴格性**：所有資料都有 "as-of" 時間戳，防止無意間使用未來資訊
3. **速率限制 + 重試**：指數退避重試（1s → 2s → 4s），Token Bucket 速率限制
4. **資料完整性校驗**：抓取後自動比對筆數、日期連續性、OHLC 合理性

---

## 五、Layer 2：標籤工程

### 為什麼標籤方法至關重要

> naive `pct_change(5).shift(-5)` 的致命問題：
> 1. 固定持有期 — 不反映真實交易的止盈/停損行為
> 2. 不考慮路徑 — 無法區分「穩定上漲 5%」和「先跌 10% 再反彈 5%」
> 3. 所有樣本等權 — 忽略標籤之間的時間重疊

### Triple Barrier Method

```
                    ┌──── 上障礙 (止盈) = entry + ATR × upper_mult
                    │
entry price ────────┤     持有期內，
                    │     先觸及哪個障礙就取哪個標籤
                    │
                    └──── 下障礙 (停損) = entry - ATR × lower_mult

              ────────────────────────────────→ 時間障礙 (max_holding 天)
```

**參數建議（台股）：**

| 參數 | 建議值 | 理由 |
|------|-------|------|
| `upper_multiplier` | 2.0 × ATR | 台股日均波動約 1-2%，2 ATR ≈ 合理止盈 |
| `lower_multiplier` | 2.0 × ATR | 對稱設計，讓模型學習方向而非偏誤 |
| `max_holding` | 10 天 | 台股 T+2 結算，10 天 ≈ 兩週，適合短線 |
| `atr_window` | 14 天 | 標準 ATR 計算視窗 |

### Meta-Labeling（進階：二階段方法）

```
Stage 1: 主模型決定方向（多/空/中性）
         │
         ▼
Stage 2: 次模型輸出信心機率 p ∈ [0, 1]
         │
         ├── p > 0.6 → 正常倉位
         ├── 0.4 < p < 0.6 → 減半倉位
         └── p < 0.4 → 不交易
```

**來源**：Marcos López de Prado, *Advances in Financial Machine Learning*, Ch. 3 & 50

**價值**：
- 主模型負責 recall（不漏掉機會）
- 次模型負責 precision（過濾假信號）
- 信心度直接映射到倉位大小 → 解決「多少錢」的問題

### 樣本唯一性權重 (Average Uniqueness)

```python
weight[i] = mean(1 / concurrency[t]) for t in holding_period[i]
```

- 當多個標籤的持有期重疊時，重疊區域的唯一性下降
- 高度重疊的樣本給予低權重 → 防止 XGBoost 過度學習冗餘信號
- 傳入 `sample_weight` 參數

---

## 六、Layer 3：特徵工程

### 特徵分類矩陣

| 類別 | 特徵 | 數量 | 行情依賴性 |
|------|------|------|-----------|
| 價格 | close, open, high, low, volume, return_1d/5d/20d | 8 | 低 |
| 趨勢 | SMA(5/20/60), EMA(12/26) | 5 | 趨勢市有效 |
| 動量 | RSI(14), KD(K/D), MACD/Signal/Hist | 6 | 震盪市有效 |
| 波動 | BB_upper/lower/width, ATR, Parkinson Vol, Realized Vol | 6 | 全市場 |
| 乖離 | BIAS(5/10/20) | 3 | 均值回歸 |
| 量能 | OBV, volume_ratio_5d | 2 | 全市場 |
| 趨勢強度 | ADX, DI+/DI- | 3 | 趨勢確認 |
| 情緒 | sentiment_score, ma5, change, post_volume, bullish_ratio | 5 | 全市場 |
| 籌碼 | foreign/trust/dealer_buy_sell, margin/short_balance | 5 | 全市場 |
| 微結構 | spread_proxy | 1 | 流動性 |
| 日曆 | day_of_week, month, is_settlement | 3 | 季節性 |

**總計：47 維 → 篩選後 15-20 維**

### 三階段特徵工程流程

```
Stage 1: 原始特徵計算（47 維）
    │
    ▼
Stage 2: 篩選
    ├── Mutual Information（預設，無需額外模型）
    ├── SHAP（基於輕量 XGBoost，更準確但更慢）
    └── 共線移除（Pearson r > 0.95 的配對中移除排名較低者）
    │
    ▼
Stage 3: 行情相依監控
    └── 每月重新計算特徵重要性排名
        若排名大幅變動 → 觸發模型再訓練
```

### 特徵篩選的研究支撐

- **SHAP 處理特徵相關性更好**：考慮所有可能的交互作用（Springer 2024 比較研究）
- **ADX 始終被識別為重要**：在多項研究中穩定出現在 top features
- **行情相依特徵重要性**：一目均衡表和 EMA 在趨勢市有效，但盤整市失效（需要 HMM 配合）
- **法人籌碼比散戶情緒更重要**：台股尤其如此（法人佔成交量 60%+）

### 避免的陷阱

- **不要加更多特徵** — 43 個對有限樣本量已經過多，要減少
- **不要用 PCA 取代特徵篩選** — PCA 失去可解釋性，且假設線性結構
- **不要對日曆特徵做 one-hot** — 會爆炸維度，用原始數值即可

---

## 七、Layer 4：ML 模型層

### 模型選擇矩陣

| 模型 | 擅長 | 弱點 | 角色 |
|------|------|------|------|
| **LSTM + Attention** | 時間序列長期依賴、捕捉動量 | 訓練慢、需要較多資料 | 趨勢捕捉 |
| **XGBoost** | 表格特徵、缺失值穩健、快 | 不擅長序列模式 | 特徵交互 |
| **TFT** | 多步預測、可解釋性、自動特徵選擇 | 複雜、需要 GPU | 主力預測 |

### TFT 為何是 2026 的首選

根據最新研究（2025）：
- MAE 降低 40-50%（相比 LSTM 和 BiLSTM）
- MAPE < 2%
- **內建可解釋性**：Variable Selection Network 自動顯示哪些特徵重要
- **Attention 層**：暴露哪些時間步最有影響力

**TFT 關鍵超參數：**

| 參數 | 建議值 | 說明 |
|------|-------|------|
| hidden_size | 64-128 | 主架構超參數 |
| lstm_layers | 1-2 | 更多層不一定更好 |
| attention_heads | 4 | 多頭注意力 |
| dropout | 0.1-0.3 | 正則化 |
| lookback | 60 天 | 約 3 個月歷史 |
| forecast_horizon | 5-10 天 | 短線預測 |

**進階變體：**
- **TFT-GNN**：結合圖神經網路捕捉股票間關聯性（2025）
- **TFT-ASRO**：直接優化 Sharpe Ratio 而非 MSE（2025）

### 集成策略

```
                 LSTM 預測
                    │
                    ├──→ ┌──────────────────────┐
XGBoost 預測 ──────┼──→ │  Stacking Ensemble   │ ──→ 最終預測
                    ├──→ │  (Ridge meta-learner) │
   TFT 預測 ──────┘     └──────────────────────┘
                              │
                    用 validation set 訓練
                    test set 完全不碰
```

**為什麼 Ridge 做 meta-learner？**
- 學習「趨勢行情信 LSTM/TFT，盤整行情信 XGBoost」的非線性互補
- L2 正則化防止 meta-learner 本身過擬合
- 比簡單加權平均更好，但不會像深度網路那樣增加過擬合風險

### Purged Walk-Forward CV

```
Fold 1: [Train────────] [purge][embargo] [Test─────]
Fold 2: [Train───────────────] [purge][embargo] [Test─────]
Fold 3: [Train──────────────────────] [purge][embargo] [Test─────]
                                        10d    5d

purge = max_holding_period (10 天)：
  移除訓練集末尾與測試集標籤有時間重疊的樣本

embargo = 額外安全間隔 (5 天)：
  防止自相關特徵洩漏
```

**進階：Combinatorial Purged CV (CPCV)**

```python
from skfolio.model_selection import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    n_folds=10,
    n_test_folds=2,
    purge_threshold=10,
    embargo_threshold=5,
)
# 產生多條回測路徑的分佈
# 計算 Probability of Backtest Overfitting (PBO)
# 計算 Deflated Sharpe Ratio (DSR)
```

**CPCV 的價值**（比單路徑 Walk-Forward 更好）：
- 產生**統計分佈**而非單一績效路徑
- 直接計算過擬合機率（PBO）
- 如果 PBO > 50%，策略幾乎確定是過擬合

---

## 八、Layer 5：市場狀態偵測

### 為什麼 HMM 是目前證據最充分的技術

| 研究 | 方法 | 結果 |
|------|------|------|
| Multi-Model Ensemble-HMM (2025) | XGBoost + RF + HMM 投票 | 在 Russell 3000 / S&P 500 上驗證有效 |
| NIFTY 50 研究 (2024) | 3-state HMM | Sharpe 從 0.67 → 1.05 |
| Renaissance Technologies | HMM (Baum-Welch) | 30+ 年使用的核心方法之一 |
| ComSIA 2026 | 20 日均線行情分類 | 搭配 XGBoost + ATR 達到 Sharpe 1.68 |

### 3-State Gaussian HMM 設計

```
觀測值 X = [daily_return, realized_volatility]

State 0: Bull  ── 正報酬、低波動 ── 信號 × 1.0
State 1: Bear  ── 負報酬、高波動 ── 信號 × 0.3（核心：何時不交易）
State 2: Side  ── 低報酬、中波動 ── 信號 × 0.5
```

**狀態自動分配**：根據學到的各狀態平均報酬率排序，自動對應 bull/bear/sideways。

### Dual Pipeline 架構（來自 2025 Multi-Model Ensemble-HMM）

```
Pipeline 1: 短期技術信號
    XGBoost + Random Forest
    特徵：MACD, RSI, BB, 量能
    → 產出短期方向預測

Pipeline 2: 長期行情判斷
    HMM (3 states)
    特徵：20 日均線斜率、波動率、Put/Call Ratio
    → 產出行情狀態

Voting Classifier:
    若 Pipeline 2 = Bear → 不管 Pipeline 1 說什麼，不做多
    若 Pipeline 2 = Bull + Pipeline 1 = Buy → 執行
    若 Pipeline 2 = Sideways + Pipeline 1 = Buy → 減半倉位
```

### 行情切換的應對策略

| 從\到 | Bull | Sideways | Bear |
|-------|------|----------|------|
| **Bull** | 持續 | 減倉 50% | 清倉 + 暫停 |
| **Sideways** | 加倉 | 持續 | 清倉 + 暫停 |
| **Bear** | 試探性小倉位 | 小倉位 | 持續暫停 |

**熊市中的核心行為：**
- 所有 buy 信號降級為 hold
- 已有持倉觸發 trailing stop 後禁止加倉
- 等到 HMM 狀態切換至 sideways 以上才重新進場

---

## 九、Layer 6：信號合成與規則引擎

### 規則引擎設計

```python
class RuleEngine:
    ML_WEIGHT = 0.70    # ML 信號佔 70%
    AGENT_WEIGHT = 0.30  # Agent 建議佔 30%

    def decide(ml_signal, ml_conf, agent_signal, agent_conf, market_state):
        # 1. 加權分數
        ml_score    = signal_to_score(ml_signal) × ml_conf
        agent_score = signal_to_score(agent_signal) × agent_conf
        combined    = ML_WEIGHT × ml_score + AGENT_WEIGHT × agent_score

        # 2. HMM 狀態縮放
        state_scale = {"bull": 1.0, "sideways": 0.7, "bear": 0.5}
        adjusted = combined × state_scale[market_state]

        # 3. Meta-Label 信心度 → 倉位
        if meta_label_prob < 0.4:
            return "hold"  # 信心不足，不交易

        # 4. 閾值決策
        if adjusted > +0.25: action = "buy"
        elif adjusted < -0.25: action = "sell"
        else: action = "hold"

        # 5. 倉位 = Kelly × meta_label_prob × state_scale
        position = kelly_fraction × meta_label_prob × state_scale

        return action, position
```

### 為什麼 70/30 比例？

- **ML 信號的優勢**：可量化、可回測、一致性高
- **Agent 建議的價值**：捕捉 ML 無法處理的定性資訊（新聞事件、政策變化）
- **30% 的上限**：防止 Agent 的不可預測性主導決策
- 根據 FINSABER 的結論，Agent 佔比不應超過 30%

### 信號衝突處理

| ML 信號 | Agent 信號 | 市場狀態 | 決策 |
|---------|-----------|---------|------|
| buy | buy | bull | **buy**（全共識） |
| buy | sell | bull | **hold**（衝突，觀望） |
| buy | buy | bear | **hold**（HMM 否決） |
| sell | sell | bear | **sell**（全共識） |
| hold | buy | bull | **buy**（弱信號） |
| buy | hold | sideways | **hold**（信號不足） |

---

## 十、Layer 7：LLM Agent 層（顧問角色）

### Agent 的正確用途

| 用途 | 可信度 | 說明 |
|------|-------|------|
| 情緒提取 | **高** | LLM 最被驗證的金融應用（81-90% 準確率） |
| 新聞摘要 | **高** | 結構化非結構化資訊 |
| 研究加速 | **高** | Two Sigma 2026：讓研究員效率提升 10x |
| 模式辨識 | **中** | 輔助人類發現潛在模式 |
| 交易決策 | **低** | FINSABER 證明不可靠 |
| 股價預測 | **極低** | 僅 ~54% 準確率（幾乎等同擲硬幣） |

### 多 Agent DAG 設計

```
Phase 1（並行，4 個分析師）：
    ┌── 技術面 Agent (Claude Haiku)  → RSI、MACD 解讀
    ├── 情緒面 Agent (Claude Sonnet) → PTT/新聞風向
    ├── 基本面 Agent (Claude Haiku)  → 法人動向解讀
    └── 量化面 Agent (無 LLM)        → ML 預測彙整

Phase 2（研究員辯論）：
    Bull Researcher ←→ Bear Researcher
    多輪 message passing（非單次 prompt）
    → 產出：consensus, signal, confidence, key_risks

Phase 3（輸出建議，非決策）：
    researcher_signal → 傳入規則引擎的 agent_signal
    researcher_confidence → 傳入規則引擎的 agent_confidence

    ⚠ Agent 不直接決定 action / position_size / stop_loss
    ⚠ 這些由規則引擎 + 風控系統決定
```

### TradeTrap 防禦措施

根據 TradeTrap (arXiv 2512.02261) 的攻擊向量：

| 攻擊類型 | 防禦 |
|---------|------|
| 記憶竄改 | Agent 記憶只讀，不可寫回交易狀態 |
| Prompt 注入 | 輸入清洗 + 固定 system prompt |
| 資料偽造 | Agent 不直接接觸原始資料源 |
| 工具劫持 | Agent 無執行權限，只有建議權限 |
| 狀態竄改 | Portfolio 狀態由獨立模組管理，Agent 不可修改 |

**核心防禦原則：LLM 絕不擁有直接交易執行權。**

### 三層記憶系統 (FinMem)

| 層 | 視窗 | 內容 | 衰減 |
|----|------|------|------|
| Shallow (短期) | 14 天 | 日新聞摘要、近期決策 | α = 0.9 |
| Intermediate (中期) | 90 天 | 季度模式、法人趨勢 | α = 0.967 |
| Deep (長期) | 365 天 | 年度模式、結構性變化 | α = 0.988 |

**排名公式**：recency × relevancy × importance，各層有獨立穩定性參數

**層級轉移機制**：
- 勝率 ≥ 60%（3+ 筆交易）→ 短期模式升級到長期
- 連續虧損 3 次 → 相關模式降級、標記不可靠

---

## 十一、Layer 8：硬性風控

### 硬性規則（LLM 無法覆蓋）

```
┌─────────────────────────────────────────────┐
│           硬性風控 — 最後防線                   │
│                                               │
│  這些規則無論 LLM / Agent / ML 怎麼說          │
│  都不可違反。違反 = 自動拒絕交易。              │
│                                               │
│  1. 最大單一倉位: 20%                          │
│  2. 最大同時持倉: 5 檔                         │
│  3. 停損幅度: ≤ 8%                             │
│  4. 風險報酬比: ≥ 2:1                          │
│  5. Kelly 倉位: 1/4 Kelly                     │
│  6. 最大回撤熔斷: 15%                          │
│  7. Trailing Stop 觸發後禁止加倉                │
│  8. HMM Bear 狀態禁止新建多頭                   │
│  9. 單日虧損 > 3% → 暫停交易                   │
│  10. 單週虧損 > 5% → 暫停交易                  │
│  11. 單月虧損 > 10% → 暫停交易                 │
└─────────────────────────────────────────────┘
```

### ATR Trailing Stop 設計

```
進場 ──→ 初始停損 = entry - 2.5 × ATR

每日更新：
    if today_high > highest_since_entry:
        highest = today_high
        new_stop = highest - 2.5 × ATR
        if new_stop > current_stop:
            current_stop = new_stop  ← 只升不降（棘輪機制）

    if today_low ≤ current_stop:
        → 觸發出場
        → 標記 stock_id 禁止加倉
```

**為什麼 2.5x 而非 2.0x？**
- 靜態停損用 2.0x（初始進場）
- Trailing Stop 用 2.5x（稍寬以避免正常波動觸發）
- 台股日均波動約 1-2%，2.5 ATR ≈ 2.5-5%，合理的追蹤距離

### 倉位管理公式

```python
# ComSIA 2026 的倉位公式（驗證有效）
position = min(
    floor(0.01 × cash / ATR),     # 風險基礎
    floor(0.10 × cash / price),    # 最大倉位
    kelly_fraction × meta_prob,    # Kelly × 信心度
)
```

### 最大回撤熔斷

```
equity_curve → calculate_drawdown()

if max_drawdown > 15%:
    circuit_breaker = ON
    → 禁止一切買入
    → 只允許平倉
    → 需要人類手動重置

    重置條件建議：
    1. 回撤收窄至 10% 以下
    2. HMM 狀態切換至 bull/sideways
    3. 至少觀望 5 個交易日
```

---

## 十二、Layer 9：執行與監控

### APScheduler 排程

| 時間 | 任務 | 詳細 |
|------|------|------|
| 08:30 盤前 | 資料更新 + 全分析 | 抓取最新資料 → ML 預測 → HMM 狀態 → Agent 觀點 → 規則引擎 → 推播 |
| 09:00-13:30 盤中 | Trailing Stop 監控 | 每 30 分鐘檢查所有持倉的 trailing stop |
| 14:00 盤後 | 結算 + 回饋 | 更新權益曲線 → 熔斷檢查 → 記憶層級轉移 → 交易日誌 |
| 20:00 晚間 | 情緒 + 再訓練 | 爬蟲更新 → 特徵重要性檢查 → 條件式模型再訓練 |
| 週末 | 週報 + 深度分析 | 週績效報告 → HMM 狀態歷史 → 模式驗證 |

### 台股成本模型

```python
# 買進
buy_cost = price × shares × (1 + 0.001425 × 0.28)  # 手續費 0.1425% × 2.8折

# 賣出
sell_cost = price × shares × (1 - 0.001425 × 0.28 - 0.003)  # + 證交稅 0.3%

# 來回成本 ≈ 0.38%
# 每月交易 4 次 → 年成本 ≈ 18%
# → 每月交易不應超過 2-3 次
```

### 通知層級

| 等級 | 條件 | 推播方式 |
|------|------|---------|
| INFO | 每日分析完成 | Telegram |
| SIGNAL | 規則引擎產出 buy/sell | LINE + Telegram |
| WARNING | 回撤接近上限 (> 10%) | LINE + Telegram + 音效 |
| CRITICAL | 熔斷觸發 / 系統異常 | 全通道 + 電話（如果可能） |

---

## 十三、Layer 10：回饋與記憶

### 閉環回饋系統

```
交易結果
    │
    ├──→ 記憶層級轉移
    │    ├── 勝率 ≥ 60% → 模式升級
    │    └── 連續虧損 3 次 → 模式降級
    │
    ├──→ 模型再訓練觸發
    │    ├── 特徵重要性排名變動 > 30% → 觸發
    │    ├── 近 20 筆交易方向準確率 < 50% → 觸發
    │    └── HMM 狀態持續 bear > 20 天 → 觸發
    │
    ├──→ 策略參數調整
    │    ├── Trailing Stop multiplier 微調
    │    ├── 規則引擎閾值微調
    │    └── 永遠 A/B 測試，不盲目調整
    │
    └──→ 人類審查
         ├── 週報：績效 + HMM 狀態 + 特徵漂移
         ├── 月報：完整回測重跑 + PBO 檢查
         └── 季報：架構層級審查
```

### 模型再訓練策略

```python
# 不要每天重新訓練 — 過擬合風險
# 不要每季才訓練 — 市場已經變了

觸發條件（滿足任一即觸發）：
1. 滾動 20 筆交易方向準確率 < 50%
2. 特徵重要性 top-10 排名變動 > 3 個位置
3. HMM 偵測到行情切換（bull ↔ bear）
4. 距離上次訓練 > 60 個交易日

再訓練流程：
1. 用最新資料重新建立特徵
2. 重新執行特徵篩選（可能篩出不同特徵）
3. Purged Walk-Forward 驗證
4. 若新模型 > 舊模型 → 替換
5. 若新模型 ≤ 舊模型 → 保留舊模型 + 記錄
```

---

## 十四、驗證框架

### 結構性效度框架

來自 2602.14233 偏誤審查的五大組件：

| 組件 | 具體做法 |
|------|---------|
| **時間清洗** | 所有資料有 as-of 時間戳；模型訓練截止日明確標示 |
| **動態宇宙構建** | 保留已下市/被併購股票的歷史資料；報告非存活者佔比 |
| **推理穩健性** | Agent 的解釋視為可測試對象；做實體替換測試 |
| **認知校準** | 輸出預測分佈（非點估計）；包含明確的「不交易」選項 |
| **現實約束** | 報告扣除成本後的淨績效；包含交易延遲和滑價 |

### 每次改進必做的 A/B 驗證

```
                    改進前模型
                        │
        ┌───────────────┼───────────────┐
        │                               │
    Purged WF CV                   Purged WF CV
    (相同資料切分)                  (相同資料切分)
        │                               │
    改進前績效                       改進後績效
        │                               │
        └───────── 比較 ────────────────┘
                    │
            t-test / bootstrap CI
            判斷改進是否顯著
```

### 過擬合偵測清單

- [ ] Sharpe Ratio > 2.0？**幾乎確定過擬合**
- [ ] 回測與實盤差距 > 30%？**可能過擬合**
- [ ] CPCV 的 PBO > 50%？**確定過擬合**
- [ ] 只在特定行情（如牛市）表現好？**行情過擬合**
- [ ] 特徵超過 20 個？**可能特徵過擬合**
- [ ] 訓練集和測試集方向準確率差距 > 10%？**可能過擬合**

---

## 十五、台股特殊考量

### 台股 vs 美股的關鍵差異

| 面向 | 台股 | 美股 |
|------|------|------|
| 交易時間 | 09:00-13:30（4.5 小時） | 09:30-16:00（6.5 小時） |
| 漲跌幅限制 | ±10% | 無限制 |
| 交易單位 | 1 張 = 1000 股（零股另計） | 1 股 |
| 結算 | T+2 | T+1 (2024 起) |
| 稅 | 證交稅 0.3%（賣方） | 資本利得稅 |
| 法人結構 | 外資 + 投信 + 自營商 | 更多元 |
| AI 研究 | 極少（= 機會） | 飽和 |
| 期貨結算 | 每月第三個星期三 | 每月第三個星期五 |

### 台股獨特的 alpha 來源

1. **法人籌碼**：三大法人買賣超是台股最有效的前瞻指標之一
2. **融資融券變化**：融資大增常見於散戶追高，是反向指標
3. **期貨結算日效應**：每月第三個星期三前後波動率增加
4. **營收月報效應**：台股公司每月公布營收，美股只有季報
5. **外資對台幣匯率敏感**：台幣升值 → 外資傾向加碼

### 台股的 API 生態

| API | 提供者 | 用途 |
|-----|-------|------|
| FinMind | FinMind | 歷史資料（免費方案有速率限制） |
| Shioaji | 永豐金證券 | 即時報價 + 下單 |
| XQ | 嘉實資訊 | 技術分析平台 |
| TWSE OpenAPI | 交易所 | 官方資料（延遲） |

---

## 十六、績效期望值

### 誠實的績效預期

| 指標 | 回測 | 模擬盤 | 實盤 |
|------|------|-------|------|
| Sharpe Ratio | 1.5-2.5 | 1.0-1.8 | **0.8-1.5** |
| 年化報酬 | 20-40% | 12-25% | **8-20%** |
| 最大回撤 | -8% ~ -12% | -10% ~ -15% | **-12% ~ -18%** |
| 勝率 | 55-65% | 50-60% | **48-58%** |
| 月交易次數 | N/A | N/A | **2-4 次** |

### 為什麼實盤會降？

| 因素 | 影響 |
|------|------|
| 滑價 | 回測假設完美成交，實際有 0.1-0.5% 滑價 |
| 情緒干擾 | 人類會在虧損時加大倉位（報復性交易） |
| 資料延遲 | 回測用收盤價，實盤用即時價 |
| 交易成本 | 台股來回約 0.38%，月交易 4 次 = 年成本 18% |
| 市場衝擊 | 大額委託會影響價格（小型股尤其明顯） |
| 行情變化 | 訓練期和實盤期的市場結構可能不同 |

### 里程碑設定

```
Month 1-3: 模擬盤
    目標：Sharpe > 0.5, MaxDD < 20%
    通過 → 進入 Month 4

Month 4-6: 小額實盤（總資金 10%）
    目標：Sharpe > 0.6, MaxDD < 18%
    通過 → 進入 Month 7

Month 7-12: 正常倉位實盤
    目標：Sharpe > 0.8, MaxDD < 15%
    每月檢視，連續 2 月不達標 → 回到模擬盤
```

---

## 十七、技術棧建議

### 核心依賴

| 類別 | 技術 | 版本 | 理由 |
|------|------|------|------|
| Python | Python | ≥ 3.12 | 型別提示、match 語法 |
| ML | PyTorch | ≥ 2.2 | LSTM + TFT |
| ML | XGBoost | ≥ 2.0 | 表格模型 + 樣本權重 |
| ML | scikit-learn | ≥ 1.4 | 特徵篩選、Ridge |
| TFT | pytorch-forecasting | ≥ 1.0 | TFT 實作 |
| HMM | hmmlearn | ≥ 0.3 | 市場狀態偵測 |
| 波動率 | arch | ≥ 7.0 | GARCH |
| 特徵 | shap | ≥ 0.45 | SHAP 特徵重要性（選用） |
| 驗證 | skfolio | ≥ 0.5 | CPCV（選用） |
| LLM | anthropic | ≥ 0.42 | Claude API |
| 編排 | langgraph | ≥ 0.2 | Agent DAG |
| 資料 | FinMind | ≥ 1.6 | 台股資料 |
| 資料 | pandas | ≥ 2.2 | 資料處理 |
| DB | sqlalchemy | ≥ 2.0 | ORM |
| 前端 | streamlit | ≥ 1.35 | UI |
| 圖表 | plotly | ≥ 5.22 | 互動圖表 |
| 排程 | apscheduler | ≥ 3.10 | 定時任務 |
| 最佳化 | cvxpy | ≥ 1.5 | 組合最佳化 |

### 資料庫 Schema（6 張核心表）

```sql
-- 市場資料
StockPrice:     stock_id, date, O, H, L, C, V, foreign/trust/dealer, margin/short
SentimentRecord: stock_id, date, source, title, sentiment_label, score, keywords

-- ML 結果
Prediction:     stock_id, date, model, predicted_return, confidence, signal

-- 回測
BacktestResult: strategy, fold, sharpe, max_drawdown, direction_accuracy, PBO

-- Agent 記憶
AgentMemory:    stock_id, memory_type, category, content, embedding, relevance, access_count
TradeJournal:   stock_id, date, action, price, exit_price, pnl, reasoning, review_notes
```

---

## 十八、開發路線圖

### Phase 1：修復根基（1-2 週）

- [x] Triple Barrier 標籤
- [x] Purged Walk-Forward CV（purging + embargo）
- [x] 特徵篩選（43 → 15-20 維）
- [x] 樣本唯一性權重
- [ ] 整合 CPCV（skfolio）計算 PBO

### Phase 2：核心升級（2-4 週）

- [x] HMM 3-State 市場狀態偵測
- [x] LLM Agent 降級為顧問
- [x] 規則引擎（ML 70% + Agent 30%）
- [x] ATR Trailing Stop
- [x] 最大回撤熔斷
- [ ] Meta-Labeling 二階段（主模型方向 + 次模型信心）
- [ ] 多輪 Agent 辯論（message passing）

### Phase 3：驗證與調優（2-4 週）

- [ ] 完整 A/B 回測（每個改進前後對比）
- [ ] 台股 10 檔權值股全量驗證
- [ ] 存活者偏誤修正（加入已下市股票）
- [ ] 成本模型校準（實際滑價測量）
- [ ] PBO 計算與過擬合風險評估

### Phase 4：模擬盤（3 個月）

- [ ] Shioaji API 串接（模擬下單）
- [ ] 即時 Trailing Stop 監控
- [ ] 每日自動分析 + 推播
- [ ] 每週績效報告 + HMM 狀態歷史
- [ ] 每月模型再訓練觸發檢查

### Phase 5：小額實盤（3 個月）

- [ ] 總資金 10% 實盤測試
- [ ] 人類審查每筆交易
- [ ] 收集實盤 vs 回測差距資料
- [ ] 調整成本模型（實際滑價 vs 假設滑價）

### 不建議做的事

| 別做 | 理由 |
|------|------|
| 追求更大的 LLM | FINSABER 證明瓶頸不在模型大小 |
| 用 LLM 直接預測股價 | 僅 ~54% 準確率 |
| 信任漂亮的回測數字 | 實盤通常降 30-50% |
| 加更多特徵 | 反而要減少 |
| 每天重新訓練模型 | 過擬合風險 |
| 用 VAE 取代 PCA | 帶來的增益可能不值得複雜度 |
| 追求 Sharpe > 2.0 | 幾乎確定是過擬合 |

---

## 十九、附錄：論文索引

### 核心論文

| 簡稱 | 全名 | 來源 | 年份 |
|------|------|------|------|
| **FINSABER** | Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? | KDD 2026 / arXiv 2505.07078 | 2025 |
| **StockBench** | StockBench: LLM Agent Benchmark | arXiv 2510.02209 | 2025 |
| **TradeTrap** | TradeTrap: LLM Agent Vulnerability | arXiv 2512.02261 | 2025 |
| **偏誤審查** | Evaluating LLMs in Finance Requires Explicit Bias Consideration | arXiv 2602.14233 | 2026 |
| **ComSIA** | Generating Alpha: Hybrid AI-Driven Trading System | ComSIA 2026 / arXiv 2601.19504 | 2026 |
| **AFML** | Advances in Financial Machine Learning | Cambridge University Press | 2018 |
| **FinMem** | FinMem: A Performance-Enhanced LLM Trading Agent | AAAI 2024 / arXiv 2311.13743 | 2024 |
| **TradingAgents** | TradingAgents: Multi-Agents LLM Financial Trading Framework | arXiv 2412.20138 | 2024 |

### 方法論論文

| 主題 | 來源 | 年份 |
|------|------|------|
| Multi-Model Ensemble-HMM Voting | AIMS Mathematics (2025) | 2025 |
| TFT-GNN Hybrid | MDPI Informatics (2025) | 2025 |
| TFT-ASRO Adaptive Sharpe | MDPI Sensors (2025) | 2025 |
| Combinatorial Purged CV | López de Prado (2018) / skfolio | 2018 |
| Meta-Labeling | López de Prado (2018) | 2018 |
| SHAP Feature Selection | Springer JBDA (2024) | 2024 |
| SHAP Financial ML | arXiv 2511.21588 | 2025 |

### 業界觀點

| 來源 | 內容 | 年份 |
|------|------|------|
| Two Sigma | AI in Investment Management: 2026 Outlook | 2026 |
| Renaissance Technologies | Medallion Fund 方法論分析 | 持續 |
| LLM Quant | 年度回顧 | 2025 |
| Dataconomy | LLM 在金融中的角色 | 2026 |

---

*本文件將隨著新研究的發表持續更新。最後更新：2026-02-26。*
