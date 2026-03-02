# **twstock-predictor：端到端人工智慧量化與多智能體台股交易系統架構與檔案配置深度解析報告**

## **系統架構哲學與 2026 量化交易典範轉移**

隨著 2026 年大型語言模型（LLM）與深度學習架構的指數型進化，傳統量化交易依賴單一時間序列模型或固定特徵工程的範式已遭遇嚴重的瓶頸。現代金融市場的微結構噪音增加、非平穩性（Non-stationarity）加劇，使得傳統統計套利與固定因子投資策略的夏普值（Sharpe Ratio）急遽衰退。本報告針對「twstock-predictor：台股智慧預測與自動交易系統」的底層設計邏輯與實體檔案架構進行深度解剖，探討其如何整合 Marcos López de Prado 的前沿金融機器學習理論 1、最新的多智能體（Multi-Agent）協同決策框架 3，以及動態隱馬爾可夫模型（HMM）預測集成技術 6，打造出一套具備自我反思、動態適應與嚴格風控的端到端台股系統。

本系統的設計哲學揚棄了「單一完美模型」的迷思，轉而擁抱「模組化異質決策」與「動態市場狀態適應」。從資料抓取的去噪、特徵工程的三重障礙標籤（Triple Barrier Method）2、機器學習時間序列交叉驗證的淨化與禁運（Purging and Embargoing）2，乃至於引入 Anthropic 最新發布的 Claude 4.6 Opus 進行深度邏輯推理與多智能體辯論 8，每一層架構皆精確針對台股特定的微結構限制與市場特性進行了深度最佳化。系統整體程式碼規模達 9,600 行，透過嚴謹的 src/ 目錄結構進行模組化隔離，確保了從研究環境到實盤部署的無縫接軌 10。

## **一、 系統目錄結構與工程設計規範 (System Infrastructure & Directory Layout)**

專業的量化交易系統必須具備高度的可重現性（Reproducibility）與擴展性。本系統揚棄了早期資料科學專案常見的扁平化腳本結構，採用了業界標準的 src/ 驅動架構 10。這種架構確保了核心商業邏輯、資料管線與外部依賴的嚴格解耦。

系統的實體檔案架構被精心劃分為十個核心命名空間，每個命名空間對應一個特定的量化處理階段：

| 目錄/模組層級 | 核心實體檔案清單 | 架構定位與設計目標 |
| :---- | :---- | :---- |
| **src/data/** | stock\_fetcher.py, sentiment\_crawler.py, news\_crawler.py | 負責所有外部 API 介接與非同步資料抓取，處理網路層級的重試邏輯與速率限制，確保原始資料的完整性與連續性 12。 |
| **src/analysis/** | features.py, llm\_features.py, labels.py | 將原始時間序列轉換為 43 維靜態與動態特徵，執行 VAE 降維，並標註三重障礙法標籤。 |
| **src/models/** | lstm\_attn.py, xgb\_model.py, tft\_model.py, ensemble.py | 封裝 PyTorch 與 XGBoost 演算法，實作隱馬爾可夫模型（HMM）動態權重分配與嚴格的淨化交叉驗證 7。 |
| **src/agents/** | analysts.py, researcher.py, trader.py, memory.py | 基於 LangGraph 構建非同步多智能體有向無環圖（DAG），實作角色分工、群組辯論與分層記憶體機制 13。 |
| **src/risk/** | manager.py, portfolio.py | 執行交易前的最後一道防線，實作 1/4 凱利準則資金分配、動態 ATR 停損與資產組合部位追蹤 15。 |
| **src/backtest/** | engine.py, strategy.py, report.py | 事件驅動（Event-driven）回測引擎，精確模擬台股交易稅制與流動性滑價，產出機構級績效指標。 |
| **src/portfolio/** | optimizer.py, metrics.py | 基於馬可維茲均值-變異數理論（Mean-Variance Optimization）進行多檔資產的權重最佳化。 |
| **src/pipeline/** | scheduler.py, tasks.py | 端到端即時排程管線，利用 APScheduler 管理盤前、盤中與盤後的自動化任務。 |
| **src/monitoring/** | alerts.py, logger.py | 負責 LINE Notify 與 Telegram Bot 的推播通知，以及系統層級的異常日誌記錄。 |
| **app/** | main.py, components/ | 基於 Streamlit 與 Plotly 構建的互動式前端儀表板，提供預測視覺化與多智能體辯論過程的透明度展示。 |

在上述的程式碼規模中，原有基礎框架佔約 3,700 行，針對 2026 年最新論文技術棧的優化（包含 Agent 架構、VAE 降維、動態集成等）新增與修改了約 5,900 行，總計 9,600 行的程式碼庫奠定了系統的工程穩健性。

## **二、 資料抓取層與台股微結構整合 (Data Layer & Microstructure Integration)**

資料層（src/data/）是量化系統的地基。金融數據往往充斥著缺失值、前復權錯誤與極端微結構噪音，因此資料抓取模組的設計必須具備極高的容錯能力與在地化市場適應性。

### **容錯機制與非同步排程架構**

在 stock\_fetcher.py 中，系統主要透過 FinMind API 獲取日 K 線（OHLCV）、三大法人買賣超與融資融券數據。為了應對高頻抓取時常見的網路延遲與 API 限制，系統實作了基於協程（Coroutine）的非同步抓取機制，並導入指數退避重試（Exponential Backoff，1s → 2s → 4s）與 Token Bucket 速率限制器 12。這種設計不僅能有效區分暫時性伺服器錯誤（如 HTTP 429 過多請求或 503 服務不可用）與永久性錯誤（如 404），更能確保在每日盤後更新時段，不會因瞬間併發請求過高而觸發資料提供商的封鎖機制。

### **台股微結構特徵與法規適應性**

系統的底層資料處理嚴格遵循台股特有的交易機制與微觀結構。台灣立法院於 2024 年底三讀通過，將現股當沖證券交易稅減半（0.15%）的優惠政策延長三年至 2027 年 12 月 31 日 17。針對這項法規延展，系統的資料層與 src/backtest/engine.py 模組已進行動態參數綁定，確保回測與實盤的摩擦成本（Friction Costs）計算具備時間一致性，避免因稅制變更導致淨報酬率估計失真。

此外，台灣證券交易所（TWSE）設有盤中瞬間價格穩定措施（Intraday Volatility Interruption）。當系統檢測到潛在成交價超出前五分鐘滾動加權平均價的 ![][image1] 時，將暫停連續撮合 2 分鐘，轉為集合競價機制，期間僅接受限價單（ROD）的新增、取消與修改，不接受市價單與 IOC/FOK 委託 21。stock\_fetcher.py 在清洗日內高頻數據時，會自動標記並平滑這些因流動性枯竭（Liquidity Freeze）而產生的微觀結構異常點，避免機器學習模型在訓練時將其誤判為真實的宏觀價格跳躍（Price Jumps）。

### **結構化情緒提取與自然語言處理**

在非結構化資料方面，sentiment\_crawler.py 與 news\_crawler.py 利用 Firecrawl 技術爬取 PTT 股票板與鉅亨網、工商時報等新聞文本。有別於傳統使用基於詞典（Lexicon-based）的樸素情緒分析，系統首創採用 Anthropic 的 Claude 4.5 Haiku 模型進行第一道結構化情緒提取 8。Claude 4.5 Haiku 以其極低的延遲、高吞吐量與優異的性價比，能精準提取新聞文本中的核心實體，並直接輸出結構化的 JSON 格式，包含新聞標題摘要、量化的多空判斷分數以及社群互動預測數。若 API 發生異常，系統則會自動降級（Fallback）使用預先編譯的正則表達式（Regex）進行基礎關鍵字匹配，確保資料管線不會中斷。

## **三、 特徵工程與標籤生成法 (Feature Engineering & Labeling)**

傳統量化模型最大的痛點在於特徵共線性與標籤設定的武斷性。本系統在 src/analysis/ 中構建了高度擴展的 43 維特徵矩陣，並將 Marcos López de Prado 倡導的金融機器學習最高標準予以工程化實作。

### **43 維特徵矩陣的學理設計**

特徵空間被嚴格劃分為六大子集，旨在捕捉市場的不同微觀與宏觀維度，徹底消除噪音並保留價格序列的記憶性。

| 特徵群組 | 維度數 | 核心指標涵義與學理依據 |
| :---- | :---- | :---- |
| **價格與趨勢** | 8 | 包含開高低收等基礎數據。為維持時間序列的平穩性（Stationarity），所有絕對價格皆轉換為對數收益率（Log Returns）或相對於移動平均線的乖離率。 |
| **技術指標** | 17 | 涵蓋相對強弱指標（RSI）、指數平滑異同移動平均線（MACD）、布林通道（Bollinger Bands）、隨機指標（Stochastic Oscillator）、平均真實區間（ATR）與商品通道指標（CCI）等標準 17 項技術指標 25。這些指標用於捕捉短期動能與均值回歸現象。 |
| **籌碼面** | 5 | 結合外資、投信、自營商三大法人買賣超與融資融券餘額變化，反映機構大資金流向與散戶槓桿情緒的博弈狀態。 |
| **情緒與文本** | 5 | 透過 VAE 降維後的新聞與社群情緒分數，捕捉非理性繁榮或市場恐慌的極端群體心理現象。 |
| **波動率與微結構 (Tier 2\)** | 5 | 包含 5 日與 20 日已實現波動率（Realized Volatility）、Parkinson 高低點波動率估計，以及買賣價差代理（Spread Proxy）與 5 日量能變化比率（Volume Ratio）28。這些微結構特徵能精確量化流動性風險與市場不確定性。 |
| **日曆效應 (Tier 2\)** | 3 | 星期效應（Day of Week）、月份效應（Month）以及期貨結算日（Is Settlement）虛擬變數。此設計專門捕捉台股特有的台指期結算日外資壓低或拉高結算現象。 |

### **變分自編碼器 (VAE) 文本特徵降維**

在 llm\_features.py 中，系統進一步處理經由大型語言模型提取的情緒文本。原始文本通常會被轉換為高達 768 維的密集向量（Dense Embeddings）。然而，直接將高維度向量輸入 XGBoost 或 LSTM 會導致嚴重的維度災難（Curse of Dimensionality），使得模型在訓練集上過擬合而在測試集上失效。

為此，系統導入了變分自編碼器（Variational Autoencoder, VAE）進行非線性降維 29。與傳統的主成分分析（PCA）或 t-SNE 不同，VAE 具備編碼器與解碼器結構，並基於貝氏推論框架運作。VAE 的損失函數由重建誤差（Reconstruction Loss）與 Kullback-Leibler 散度（KL Divergence）組成，迫使編碼器不僅壓縮資料，更要學習資料潛在的機率分佈 29。透過 VAE，系統成功將 768 維的文本向量壓縮至連續且具有生成性質的 10 維潛在空間（Latent Space）30。這種作法不僅保留了金融文本中隱含的非線性語義關係，更能產生平滑的數值特徵，無縫接軌至後續的機器學習模型，顯著減少了開發時間並提升了下游模型的預測穩定性 32。

### **三重障礙法 (The Triple Barrier Method)**

傳統的「固定時間窗口」標籤法（例如預測未來 5 天的絕對收益率大於 2% 標為 1，否則標為 0）存在致命缺陷。這種方法忽略了價格變化的路徑依賴性（Path Dependency），容易導致模型在實盤中先觸發了停損，但因為第五天的收盤價反彈，而在訓練時仍被錯誤標記為獲利。

為徹底解決此標籤洩漏問題，系統在 labels.py 中全面導入 López de Prado 提出的「三重障礙法」2。該方法為每一筆觀察值動態設定三道邊界：

1. **上方障礙（停利線）**：根據歷史動態波動率（例如過去 20 日的 ATR）向上設定。若價格路徑首先觸碰此線，則標記為 1。  
2. **下方障礙（停損線）**：同樣基於波動率向下設定。若價格路徑首先觸碰此線，即使未來反彈，仍嚴格標記為 \-1。  
3. **時間障礙（到期線）**：設定一個最大的持倉時間邊界（例如 5 個交易日）。若在該窗口內價格均未觸碰上下障礙，則根據到期時的收益符號標記為 0 或 sign(return) 2。

三重障礙法的核心價值在於，它使得機器學習模型的預測目標與真實交易環境中的風控邏輯完全一致，大幅降低了理論回測勝率與實盤實際勝率之間的巨大落差。

### **樣本唯一性與權重分配 (Sample Weights by Uniqueness)**

在金融時間序列中，當使用滑動窗口計算特徵與標籤時，相鄰交易日的觀察窗口會產生高度重疊，導致樣本並非獨立同分布（Non-IID）。這種現象會讓模型對波動率較高或趨勢延續的特定市場時期產生過度擬合 2。

系統實作了基於「唯一性」（Uniqueness）的樣本權重分配演算法 35。該演算法會掃描所有訓練樣本，計算每個樣本的標籤評估窗口與其他樣本重疊的程度。若某一天的樣本與眾多其他樣本共享了大量的價格路徑，該樣本的「唯一性」分數將下降，並在訓練時被賦予較低的權重（Sample Weight）；反之，若樣本代表獨立且罕見的市場事件，則賦予較高權重 34。這項技術在 xgb\_model.py 的 XGBoost 訓練階段能顯著提升樹模型的泛化能力與樣本外（Out-of-sample）表現 37。

## **四、 機器學習預測與動態集成層 (ML Prediction & Dynamic Stacking)**

單一預測模型無法適應所有的市場狀態。系統的 src/models/ 目錄實作了三個異質的底層預測模型，並透過馬可維茲轉移機制進行動態權重分配，徹底改變了傳統靜態集成的侷限性。

### **異質模型架構設計與特長分析**

系統部署了三種架構截然不同的深度學習與機器學習模型，以確保對市場特徵的全方位捕捉：

1. **LSTM with Attention 機制 (lstm\_attn.py)**：長短期記憶網路（Long Short-Term Memory）擅長捕捉時間序列的長期依賴性與序列特徵。系統在雙層 LSTM 之上疊加了 Attention 機制，使其能自動學習並聚焦於歷史序列中對當前預測最具影響力的特定時點（例如財報發布日或除權息日），隨後透過全連接層（FC）輸出預測結果 7。  
2. **XGBoost (xgb\_model.py)**：作為基於樹的極端梯度提升演算法，XGBoost 對於表格型特徵的非線性映射、特徵共線性與缺失值具有極強的穩健性。系統將超參數設定為 500 棵樹與適當的深度限制（depth=6），並結合 L1 與 L2 正則化（Regularization）以防止在雜訊極大的台股數據上產生過擬合 7。  
3. **TFT (Temporal Fusion Transformer, tft\_model.py)**：基於 pytorch-forecasting 套件實作。TFT 融合了自迴歸模型與多頭注意力機制（Multi-head Attention）的優勢，特別擅長處理包含已知未來輸入（如日曆效應、期貨結算日）與靜態元數據（如產業別）的多步時間序列預測。TFT 的變數選擇網路（Variable Selection Network）還能自動篩選出當下最重要的特徵。

### **HMM 動態集成與市場狀態切換 (Regime Switching)**

在傳統的機器學習中，Stacking Ensemble 通常依賴 Ridge 迴歸或簡單平均法賦予基模型固定的靜態權重。然而，金融市場存在明顯的「狀態切換」（Regime Shifts）特性。例如在低波動、流動性充沛的多頭市場中，趨勢追蹤模型表現較佳；而在高波動、充滿跳空缺口的震盪市中，均值回歸與樹狀模型更具優勢。

為了解決此問題，ensemble.py 引入了隱馬爾可夫模型（Hidden Markov Model, HMM）來即時偵測市場的隱藏狀態（如牛市、熊市、盤整）6。透過觀察總體經濟指標、大盤指數與短期實現波動率，HMM 能夠以非監督學習的方式計算出當前市場所處狀態的機率分佈 38。基於這些動態輸出的狀態機率，Meta-learner 能夠適應性地調整 LSTM、XGBoost 與 TFT 的輸出權重 7。例如，當 HMM 判斷市場進入高波動熊市，系統會自動提高對極端值較為敏銳且抗雜訊能力強的 XGBoost 的依賴權重，同時降低 LSTM 的權重，從而平滑整體投資組合的報酬率曲線並提升夏普值 43。

### **嚴格的時間序列交叉驗證：淨化與禁運 (Purging and Embargoing)**

在模型評估與超參數調優階段，傳統的 K-Fold 交叉驗證會隨機打亂數據，這在時間序列分析中是致命的錯誤，會將未來資料洩漏至過去（Lookahead Bias），導致回測績效嚴重高估 2。系統的訓練管線全面實作了「淨化（Purging）」與「禁運（Embargoing）」機制 1：

* **淨化 (Purging)**：在每次分割訓練集與測試集時，系統會檢驗特徵與標籤的時間戳記。若訓練集樣本的特徵評估時間，與測試集的三重障礙標籤生成窗口存在任何重疊，該訓練集樣本將被強制作廢剃除，確保模型在訓練時絕對無法窺探測試集的任何資訊 2。  
* **禁運 (Embargoing)**：由於金融市場價格具有強烈的序列自相關性與動能溢出效應，即便實施了淨化，測試集結束後的市場狀態仍會對緊接其後的訓練集產生資訊洩漏。因此，系統會在測試集結束後，強制加入一段「禁運期」（例如 5 個交易日），該期間內的資料不參與任何訓練 2。

結合 Expanding Window（擴展窗口）的前向推進多折驗證（Walk-forward Validation），以及將 DataLoader 的 shuffle 參數設為 False，此機制確保了模型保有時間序列的因果性，使其在實盤中的表現能高度貼合回測結果。

### **對數空間的信心區間 (Log-Space Confidence Intervals)**

針對機器學習模型輸出的點預測（Point Estimate），單一數值隱藏了極大的不確定性。系統會計算其預測信心區間（Prediction Intervals）以量化該次預測的風險 44。由於股票價格的幾何複利特性（Geometric Compounding）使其自然呈現對數常態分佈（Log-normal Distribution）47，系統揚棄了傳統在算術空間中使用 cumsum 計算對稱信心區間的錯誤做法。

所有的誤差分佈與信心區間皆在對數收益率空間（Log-space）中利用 ![][image2] 進行計算，隨後再透過幾何複利公式 np.cumprod(1+r) 或指數函數轉換回算術空間 45。這項工程改進確保了在面臨大幅震盪時，資產價格的下界預測不會出現不合理的負值，且信心區間的上下界能夠精確反映真實市場中下跌有限（最多歸零）但上漲無限的非對稱偏態風險。

## **五、 多智能體決策引擎 (Multi-Agent Decision Engine)**

傳統的量化系統在取得模型信號後，往往依賴剛性的規則系統進行下單，缺乏對突發性新聞與複雜市場語境的應變能力。本系統在 src/agents/ 目錄下，參考了最新的 TradingAgents 3、FinMem 4 與 FinPos 5 等前沿學術框架，利用 LangGraph 構建了一個由 8 個核心節點組成的非同步 DAG（有向無環圖）決策網路。

### **第一階段：多維度分析師兵分多路 (Role Specialization)**

受到大型投資銀行專業分工的啟發 3，系統實例化了四個具備不同專業 Prompt 與工具調用能力（Tool Use）的分析師 Agent。透過並行處理（Parallel Execution），大幅縮短了決策延遲：

1. **技術面 Agent (Claude 4.5 Haiku)**：專注解讀 K 線型態、RSI、MACD 與均線排列等技術面特徵。  
2. **情緒面 Agent (Claude 4.5 Haiku)**：分析 PTT 股版輿論熱度與鉅亨網新聞風向。  
3. **基本面 Agent (Claude 4.5 Haiku)**：追蹤三大法人籌碼動向、融資融券增減與月營收數據。  
4. **量化面 Agent (無 LLM)**：基於規則的 Python 節點，負責彙整並解讀 ML 模型層傳遞過來的預測信號、CI 區間與 HMM 市場狀態。

### **第二階段：研究員的分組辯論與反思 (Group-based Debate & Reflection)**

這四位分析師的分析報告將匯集至「研究員 Agent」。該節點由 Anthropic 目前推理能力最強的 Claude 4.6 Opus 擔任 8。Claude 4.6 Opus 具備長達 1M Token 的上下文窗口，在複雜的多學科推理與 Agentic 工作流中領先所有前沿模型 9。

在這裡，系統實作了 TradingAgents 論文中提出的「分組辯論機制」（Group-based Debate）13。研究員會迫使觀點相左的分析師（例如：技術面認為均線多頭排列應買入，但籌碼面發現外資連續大賣）進行內部對話與邏輯攻防。透過這種結構化的通訊協定（Structured Communication Protocol），不僅能逼迫模型進行深度的思維鏈（Chain of Thought）推導，大幅減少 LLM 常見的「幻覺」（Hallucinations），還能降低高達 46.9% 不必要的 Token 消耗 53。結合自我反思機制（Self-Reflection），研究員最終會修復邏輯盲區，輸出一份綜合研究報告。

### **第三階段：雙重決策交易員 (Dual Decision Traders)**

綜合報告隨後交由「交易員 Agent」（同樣由 Claude 4.6 Opus 驅動）處理。有別於多數框架僅做單一步驟的買賣判斷，本系統借鑒了 FinPos 論文的「位置感知」（Position-Aware）與「雙重決策」（Dual Trading Decision）架構 5。

在真實市場中，交易不僅僅是預測方向，更是部位的管理 51。交易決策被解耦為兩個獨立的子流程：

1. **方向決策 (The Direction Decision)**：判斷未來行情的絕對方向（買入、賣出或空倉），並結合長期資訊進行過濾 5。  
2. **數量與風險決策 (Quantity and Risk Decision Agent)**：結合當前投資組合的既有部位（Position Exposure）、市場波動度與剩餘購買力，給出具體的建倉比例與停損/停利邊界 5。此設計確保了系統不會在單一資產上過度曝險。

### **第四與第五階段：風控攔截與分層記憶體架構 (Risk Check & Layered Memory)**

決策產出後，必須經過基於硬性規則的「風控 Agent」審核。若決策違反了最大倉位或風險報酬比限制，風控節點將直接否決或調整該筆交易。

為了解決 LLM 缺乏狀態持久性與災難性遺忘的問題，系統在決策循環的最後（Phase 5）實作了 FinMem 論文提出的「人類對齊分層記憶體機制」（Human-aligned Layered Memory）4。此模組位於 src/agents/memory.py，將記憶體劃分為多個具備不同衰減率的層級：

* **短期工作記憶 (Working Memory)**：基於記憶體內雙向佇列（In-memory Deque），存放最近 5 天的短期市場動態、即時新聞與未平倉部位狀態，供 Agent 進行即時推理 14。  
* **淺層與中層處理記憶 (Shallow/Intermediate Memory)**：透過 SQLite 關聯式資料庫與 Embedding 向量檢索，儲存近期的交易決策與季報等具備中期影響力的事件 14。  
* **深層處理記憶 (Deep Processing Layer)**：存放被驗證過具有長期預測價值的「歷史模式」。當短期記憶中的特定事件產生巨大影響（如成功預測某次股災）時，系統會自動將其「過渡」（Transit）至深層記憶，使其成為 Agent 永久的交易智慧 14。

在未來的檢索過程中，系統會結合「相似度（Similarity）」、「近期性（Recency）」與「重要性（Importance）」三個指標來動態喚醒相關記憶 14，賦予系統強大的終身學習能力。

## **六、 風險管理與投資組合最佳化 (Risk Management & Optimization)**

再強大的預測模型與推理引擎，若無嚴謹的資金控管，終將在金融市場的長尾事件中面臨破產風險（Risk of Ruin）。src/risk/ 與 src/portfolio/ 模組構建了由下而上的立體風控網。

### **分數凱利準則 (Fractional Kelly Criterion)**

src/risk/manager.py 負責將交易員 Agent 建議的部位數量轉化為實際下單股數。系統採用了「凱利準則」（Kelly Criterion）來最大化投資組合的長期幾何增長率 16。 然而，在真實金融市場中，機器學習模型的勝率與賠率估計往往存在嚴重的「估計風險」（Estimation Risk）。若盲目採用全凱利（Full Kelly）下單，一旦模型參數估計過於樂觀，將面臨巨大的資產回撤與波動，甚至觸發破產邊界 57。

因此，系統實作了更為保守且被專業量化對沖基金廣泛採用的「四分之一凱利」（1/4 Kelly 或 Quarter Kelly）15。實證研究證明，1/4 Kelly 雖然放棄了部分極端利潤，但能提供全凱利約 50% 的複合增長率，同時將波動率與破產風險大幅壓縮至僅剩 25% 15。這種非對稱的風險報酬比，完美契合台股容易受地緣政治與國際股市牽連而產生跳空缺口的特性。

### **平均變異數最佳化與動態停損 (Mean-Variance & Dynamic Stops)**

在投資組合層級（Portfolio Level），src/portfolio/optimizer.py 整合了現代投資組合學派的馬可維茲（Markowitz）「均值-變異數最佳化」（Mean-Variance Optimization）。利用 ML 模型輸出的預期報酬率作為期望值向量，並根據歷史價格計算共變異數矩陣（Covariance Matrix）來衡量資產間的關聯風險，系統自動求解最大夏普值（Max Sharpe Ratio）與最小變異數（Minimum Variance）的資產配置權重。

此外，為防範個股黑天鵝事件，系統實施了絕對的剛性規則：

* 單一個股最大倉位不得超過 20%。  
* 投資組合最多持有 5 檔股票。  
* 單一產業（如半導體板塊）的曝險不得超過總資金的 40%。  
  停損方面，除了傳統的固定百分比硬停損（≤ \-8%），系統亦導入基於真實波動幅度（Average True Range, ATR）的動態追蹤停損（Trailing Stop）。這項機制確保了在波動率擴大的市場中給予部位適當的震盪呼吸空間，而在趨勢確立反轉時能迅速鎖定利潤。

## **七、 回測引擎、管線排程與前端展示 (Backtest, Pipeline & Frontend)**

為了驗證策略的有效性並實現全自動化交易，系統的後端基礎設施涵蓋了回測、排程與監控模組。

### **事件驅動回測引擎 (Event-Driven Backtesting)**

位於 src/backtest/ 的回測引擎採用事件驅動架構構建。與傳統向量化回測（Vectorized Backtesting）容易產生前瞻偏誤不同，事件驅動引擎透過模擬真實市場的報價推送（Tick/Bar Events）與訂單執行（Order Events）來驗證策略。engine.py 內建了精確的台股摩擦成本計算，包含券商手續費（0.1425% 乘以 2.8 折優惠）以及前述支援動態切換的 0.15% 當沖證交稅。回測結束後，report.py 會產出包含年化報酬率、夏普值、最大回撤（Maximum Drawdown）、勝率與月度報酬熱力圖的機構級績效報告。

### **即時管線與監控通知 (Pipeline & Monitoring)**

實盤運作依賴 src/pipeline/ 中的 APScheduler 進行定時排程。管線被精密設定為：

* **08:30**：執行盤前資料更新，觸發 Agent 完整分析並推播盤前策略。  
* **09:00–13:30**：盤中每 30 分鐘執行一次微批次監控，檢查價格是否觸發動態 ATR 停損/止盈。  
* **14:00**：盤後結算，整理投資組合摘要。  
* **20:00**：啟動晚間情緒爬蟲更新，並利用當日新數據對 ML 模型進行線上再訓練（Online Retraining）。  
  為確保系統運行安全，src/monitoring/ 模組整合了 LINE Notify 與 Telegram Bot API，即時推播交易信號產生、回撤警報與系統執行時的異常例外（Exceptions）。

### **前端互動式儀表板 (Frontend Dashboard)**

為了解決深度學習模型與 LLM 黑盒子缺乏可解釋性的問題，系統在 app/ 目錄下基於 Streamlit 與 Plotly 構建了三頁式互動儀表板：

1. **預測可視化**：展示結合對數空間信心區間（CI）的未來走勢預測圖，以及反映模型內部特徵重要性的雷達圖。  
2. **回測驗證**：視覺化展示 Walk-forward 多折驗證的結果，包含各折的均方誤差（MSE）與方向預測準確率的熱力圖。  
3. **多智能體觀測站**：提供一鍵執行 Multi-Agent 分析的功能，實時顯示各分析師的觀點文本、研究員的分組辯論思維鏈（Chain of Reasoning）以及最終的風控核准結果，賦予交易系統極高的透明度。

## **結論**

「twstock-predictor」系統架構代表了 2026 年量化交易演進的一個歷史性里程碑。本系統成功打破了傳統計量金融與現代生成式人工智慧之間的壁壘。一方面，透過引入三重障礙法（Triple Barrier）、淨化交叉驗證（Purged Cross-Validation）與四分之一凱利準則（1/4 Kelly Criterion），系統堅守了最嚴謹的統計與風險控制底線，徹底消除了時間序列過擬合與資料洩漏的致命傷。

另一方面，透過深度整合最新的 Claude 4.6 Opus 大型語言模型、TradingAgents 的分組辯論機制、VAE 的高維降維能力與 FinMem 的分層記憶網路，系統具備了過去剛性量化模型所完全缺乏的「邏輯反思能力」與「市場語義理解力」。這套融合了 HMM 狀態切換動態集成與雙重決策智能體的端到端系統，不僅精準適應了台灣股市特有的微結構限制與稅法環境，更為未來的完全自動化、高自我演化能力之金融交易系統樹立了全新的工程標竿與理論典範。

#### **Works cited**

1. Praise for Advances in Financial Machine Learning, accessed February 26, 2026, [https://papers.ssrn.com/sol3/Delivery.cfm/SSRN\_ID3104847\_code434076.pdf?abstractid=3104847\&mirid=1](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3104847_code434076.pdf?abstractid=3104847&mirid=1)  
2. Advances in Financial Machine Learning – Marcos Lopez de Prado \- Reasonable Deviations, accessed February 26, 2026, [https://reasonabledeviations.com/notes/adv\_fin\_ml/](https://reasonabledeviations.com/notes/adv_fin_ml/)  
3. TradingAgents: Multi-Agents LLM Financial Trading Framework, accessed February 26, 2026, [https://tradingagents-ai.github.io/](https://tradingagents-ai.github.io/)  
4. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design | Proceedings of the AAAI Symposium Series, accessed February 26, 2026, [https://ojs.aaai.org/index.php/AAAI-SS/article/view/31290](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31290)  
5. FinPos: A Position-Aware Trading Agent System for Real Financial Markets \- arXiv, accessed February 26, 2026, [https://arxiv.org/html/2510.27251v1](https://arxiv.org/html/2510.27251v1)  
6. A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading \- AIMS Press, accessed February 26, 2026, [https://www.aimspress.com/article/id/69045d2fba35de34708adb5d](https://www.aimspress.com/article/id/69045d2fba35de34708adb5d)  
7. Classifying and Predicting Stock Market States Using HMM and XGBoost \- Medium, accessed February 26, 2026, [https://medium.com/@xf600/classifying-and-predicting-stock-market-states-using-hmm-and-xgboost-c23bd4af68ed](https://medium.com/@xf600/classifying-and-predicting-stock-market-states-using-hmm-and-xgboost-c23bd4af68ed)  
8. Introducing Claude Opus 4.6 \- Anthropic, accessed February 26, 2026, [https://www.anthropic.com/news/claude-opus-4-6](https://www.anthropic.com/news/claude-opus-4-6)  
9. How to Use Claude Opus 4.6: Beginner to Advanced Guide (2026) \- SSNTPL, accessed February 26, 2026, [https://ssntpl.com/how-to-use-claude-opus-4-6-guide/](https://ssntpl.com/how-to-use-claude-opus-4-6-guide/)  
10. Structuring Your Project \- The Hitchhiker's Guide to Python, accessed February 26, 2026, [https://docs.python-guide.org/writing/structure/](https://docs.python-guide.org/writing/structure/)  
11. Quant Trading Project Structure, accessed February 26, 2026, [https://parrondo.github.io/quant-trading-project-structure/](https://parrondo.github.io/quant-trading-project-structure/)  
12. Quant Trading Framework in Python (part 2\) — how to invest like professionals \- Medium, accessed February 26, 2026, [https://medium.com/@jpolec\_72972/quant-trading-framework-in-python-part-2-how-to-invest-like-professionals-f0801af1a6c1](https://medium.com/@jpolec_72972/quant-trading-framework-in-python-part-2-how-to-invest-like-professionals-f0801af1a6c1)  
13. TradingAgents: Multi-Agents LLM Financial Trading Framework \- GitHub, accessed February 26, 2026, [https://github.com/TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)  
14. FinMem: A Performance-Enhanced LLM Trading ... \- Jordan Suchow, accessed February 26, 2026, [https://suchow.io/assets/docs/yu2024finmem.pdf](https://suchow.io/assets/docs/yu2024finmem.pdf)  
15. accessed February 26, 2026, [https://medium.com/@tmapendembe\_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705\#:\~:text=The%20researchers%20found%20that%20Quarter,you're%20definitely%20getting%20there.](https://medium.com/@tmapendembe_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705#:~:text=The%20researchers%20found%20that%20Quarter,you're%20definitely%20getting%20there.)  
16. Money Management via the Kelly Criterion \- QuantStart, accessed February 26, 2026, [https://www.quantstart.com/articles/Money-Management-via-the-Kelly-Criterion/](https://www.quantstart.com/articles/Money-Management-via-the-Kelly-Criterion/)  
17. Taiwan: Update – Financial Services News 2025 for Foreign Institutional Investors, accessed February 26, 2026, [https://wts.com/global/publishing-article/20250929\_taiwan\_financial\_services\_news\_foreign\_investor\~publishing-article](https://wts.com/global/publishing-article/20250929_taiwan_financial_services_news_foreign_investor~publishing-article)  
18. Legislative Yuan extends stock day-trading tax cut \- Taipei Times, accessed February 26, 2026, [https://www.taipeitimes.com/News/biz/archives/2025/01/01/2003829438](https://www.taipeitimes.com/News/biz/archives/2025/01/01/2003829438)  
19. Legislature approves extension of stock day-trading tax cut until 2027 \- Focus Taiwan, accessed February 26, 2026, [https://focustaiwan.tw/business/202412310013](https://focustaiwan.tw/business/202412310013)  
20. What is Day Trading ? TEJ Guides You Through The Basics From The Ground Up, accessed February 26, 2026, [https://www.tejwin.com/en/insight/what-is-day-trading/](https://www.tejwin.com/en/insight/what-is-day-trading/)  
21. Service \- Taiwan Stock Exchange Corporation, accessed February 26, 2026, [https://www.twse.com.tw/en/about/company/service.html](https://www.twse.com.tw/en/about/company/service.html)  
22. 2025 Guide to Investing in Taiwan \- Taiwan Stock Exchange Corporation, accessed February 26, 2026, [https://www.twse.com.tw/en/about/company/guide.html](https://www.twse.com.tw/en/about/company/guide.html)  
23. Trading Mechanism Introduction \- Taiwan Stock Exchange Corporation, accessed February 26, 2026, [https://www.twse.com.tw/en/products/system/trading.html](https://www.twse.com.tw/en/products/system/trading.html)  
24. Continuous Trading \- Taipei Exchange, accessed February 26, 2026, [https://www.tpex.org.tw/en-us/mainboard/trading/rules/continuous.html](https://www.tpex.org.tw/en-us/mainboard/trading/rules/continuous.html)  
25. Best Trading Indicators: Most Popular Technical Indicators / Axi, accessed February 26, 2026, [https://www.axi.com/int/blog/education/trading-indicators](https://www.axi.com/int/blog/education/trading-indicators)  
26. Assessing the Impact of Technical Indicators on Machine Learning Models for Stock Price Prediction \- arXiv.org, accessed February 26, 2026, [https://arxiv.org/html/2412.15448v1](https://arxiv.org/html/2412.15448v1)  
27. Technical Indicators | Use for Trading | List | Trading Strategy \- QuantInsti Blog, accessed February 26, 2026, [https://blog.quantinsti.com/technical-indicators-trading/](https://blog.quantinsti.com/technical-indicators-trading/)  
28. Key technical indicators for stock market prediction \- ResearchGate, accessed February 26, 2026, [https://www.researchgate.net/publication/392289141\_Key\_technical\_indicators\_for\_stock\_market\_prediction](https://www.researchgate.net/publication/392289141_Key_technical_indicators_for_stock_market_prediction)  
29. Similarity-assisted variational autoencoder for nonlinear dimension reduction with application to single-cell RNA sequencing data \- PMC, accessed February 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10647110/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10647110/)  
30. Interpretation for Variational Autoencoder Used to Generate Financial Synthetic Tabular Data \- MDPI, accessed February 26, 2026, [https://www.mdpi.com/1999-4893/16/2/121](https://www.mdpi.com/1999-4893/16/2/121)  
31. VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering \- bioRxiv.org, accessed February 26, 2026, [https://www.biorxiv.org/content/10.1101/2020.07.17.207993.full](https://www.biorxiv.org/content/10.1101/2020.07.17.207993.full)  
32. Autoencoder-based General Purpose Representation Learning for Customer Embedding, accessed February 26, 2026, [https://arxiv.org/html/2402.18164v1](https://arxiv.org/html/2402.18164v1)  
33. Function to compute the triple barrier method from Advances in Financial Machine Learning (2018) by Marcos Lopez de Prado : r/algotrading \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/algotrading/comments/g08o22/function\_to\_compute\_the\_triple\_barrier\_method/](https://www.reddit.com/r/algotrading/comments/g08o22/function_to_compute_the_triple_barrier_method/)  
34. ML for Algotrading Pt. 3: Sample Weights and Label Uniqueness \- YouTube, accessed February 26, 2026, [https://www.youtube.com/watch?v=g\_C42VewM10](https://www.youtube.com/watch?v=g_C42VewM10)  
35. Financial-Machine-Learning/USDJPY\_Notebook.ipynb at master \- GitHub, accessed February 26, 2026, [https://github.com/JackBrady/Financial-Machine-Learning/blob/master/USDJPY\_Notebook.ipynb](https://github.com/JackBrady/Financial-Machine-Learning/blob/master/USDJPY_Notebook.ipynb)  
36. The reasons most ML quant funds fail (human-generated summary of Marcos Lopez de Prado lecture) \- Andrejs, accessed February 26, 2026, [https://fluentnumbers.medium.com/the-reasons-most-ml-quant-funds-fail-human-generated-summary-of-marcos-lopez-de-prado-lecture-e7d6bd95ef50](https://fluentnumbers.medium.com/the-reasons-most-ml-quant-funds-fail-human-generated-summary-of-marcos-lopez-de-prado-lecture-e7d6bd95ef50)  
37. Sample uniqueness and sample weight in AFML book \- Quantitative Finance Stack Exchange, accessed February 26, 2026, [https://quant.stackexchange.com/questions/49424/sample-uniqueness-and-sample-weight-in-afml-book](https://quant.stackexchange.com/questions/49424/sample-uniqueness-and-sample-weight-in-afml-book)  
38. Improving S\&P 500 Volatility Forecasting through Regime-Switching Methods \- arXiv, accessed February 26, 2026, [https://arxiv.org/html/2510.03236v1](https://arxiv.org/html/2510.03236v1)  
39. Market Regime using Hidden Markov Model \- QuantInsti Blog, accessed February 26, 2026, [https://blog.quantinsti.com/regime-adaptive-trading-python/](https://blog.quantinsti.com/regime-adaptive-trading-python/)  
40. A forest of opinions: A multi-model ensemble-HMM voting framework for market regime shift detection and trading \- ResearchGate, accessed February 26, 2026, [https://www.researchgate.net/publication/397111020\_A\_forest\_of\_opinions\_A\_multi-model\_ensemble-HMM\_voting\_framework\_for\_market\_regime\_shift\_detection\_and\_trading](https://www.researchgate.net/publication/397111020_A_forest_of_opinions_A_multi-model_ensemble-HMM_voting_framework_for_market_regime_shift_detection_and_trading)  
41. Regime-Switching Factor Investing with Hidden Markov Models \- MDPI, accessed February 26, 2026, [https://www.mdpi.com/1911-8074/13/12/311](https://www.mdpi.com/1911-8074/13/12/311)  
42. A Hidden Markov Ensemble Algorithm Design for Time Series Analysis \- PMC, accessed February 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9025861/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9025861/)  
43. Ensembling Hidden Markov & Bayesian Models to Classify Regimes in Equity Markets, accessed February 26, 2026, [https://andrew-hyde.medium.com/the-ensemble-of-hidden-markov-bayesian-models-for-regime-switching-in-equity-markets-a2a7dc109a39](https://andrew-hyde.medium.com/the-ensemble-of-hidden-markov-bayesian-models-for-regime-switching-in-equity-markets-a2a7dc109a39)  
44. Prediction Intervals for Machine Learning \- MachineLearningMastery.com, accessed February 26, 2026, [https://machinelearningmastery.com/prediction-intervals-for-machine-learning/](https://machinelearningmastery.com/prediction-intervals-for-machine-learning/)  
45. Construction of Confidence Interval for a Univariate Stock Price Signal Predicted Through Long Short Term Memory Network \- PMC, accessed February 26, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7373837/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7373837/)  
46. How to add confidence to model's prediction? \- Data Science Stack Exchange, accessed February 26, 2026, [https://datascience.stackexchange.com/questions/36861/how-to-add-confidence-to-models-prediction](https://datascience.stackexchange.com/questions/36861/how-to-add-confidence-to-models-prediction)  
47. Log-normal distribution \- Wikipedia, accessed February 26, 2026, [https://en.wikipedia.org/wiki/Log-normal\_distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)  
48. Small Sample Confidence Intervals in Log Space Back \- AFIT Scholar, accessed February 26, 2026, [https://scholar.afit.edu/cgi/viewcontent.cgi?article=4348\&context=etd](https://scholar.afit.edu/cgi/viewcontent.cgi?article=4348&context=etd)  
49. How would I create a 95% confidence interval with log-transformed data? \- Cross Validated, accessed February 26, 2026, [https://stats.stackexchange.com/questions/148961/how-would-i-create-a-95-confidence-interval-with-log-transformed-data](https://stats.stackexchange.com/questions/148961/how-would-i-create-a-95-confidence-interval-with-log-transformed-data)  
50. \[PDF\] TradingAgents: Multi-Agents LLM Financial Trading Framework | Semantic Scholar, accessed February 26, 2026, [https://www.semanticscholar.org/paper/TradingAgents%3A-Multi-Agents-LLM-Financial-Trading-Xiao-Sun/e3dd4964c07c914a0ccca2e2f3ed6410f8a86a6a](https://www.semanticscholar.org/paper/TradingAgents%3A-Multi-Agents-LLM-Financial-Trading-Xiao-Sun/e3dd4964c07c914a0ccca2e2f3ed6410f8a86a6a)  
51. FinPos: A Position-Aware Trading Agent System for Real Financial Markets \- arXiv, accessed February 26, 2026, [https://arxiv.org/html/2510.27251v2](https://arxiv.org/html/2510.27251v2)  
52. Claude Opus 4.6: Features, Benchmarks, Hands-On Tests, and More \- DataCamp, accessed February 26, 2026, [https://www.datacamp.com/blog/claude-opus-4-6](https://www.datacamp.com/blog/claude-opus-4-6)  
53. TradingAgents: Multi-Agents LLM Financial Trading Framework \- OpenReview, accessed February 26, 2026, [https://openreview.net/pdf/bf4d31f6b4162b5b1618ab5db04a32aec0bcbc25.pdf](https://openreview.net/pdf/bf4d31f6b4162b5b1618ab5db04a32aec0bcbc25.pdf)  
54. FinPos: A Position-Aware Trading Agent System for Real Financial, accessed February 26, 2026, [https://papers.cool/arxiv/2510.27251](https://papers.cool/arxiv/2510.27251)  
55. FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design, accessed February 26, 2026, [https://ojs.aaai.org/index.php/AAAI-SS/article/download/31290/33450/35346](https://ojs.aaai.org/index.php/AAAI-SS/article/download/31290/33450/35346)  
56. The Kelly Criterion \- Quantitative Trading \- Nick Yoder, accessed February 26, 2026, [https://nickyoder.com/kelly-criterion/](https://nickyoder.com/kelly-criterion/)  
57. The Dangers of Full Kelly Criterion: Why Most Traders Should Use Fractional Kelly Criterion Instead | by Risk Management & Lot Sizing \- Medium, accessed February 26, 2026, [https://medium.com/@tmapendembe\_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705](https://medium.com/@tmapendembe_28659/the-dangers-of-full-kelly-criterion-why-most-traders-should-use-fractional-kelly-criterion-instead-0338e3bcc705)  
58. Kelly Criterion, do you use it? : r/options \- Reddit, accessed February 26, 2026, [https://www.reddit.com/r/options/comments/1pudkp6/kelly\_criterion\_do\_you\_use\_it/](https://www.reddit.com/r/options/comments/1pudkp6/kelly_criterion_do_you_use_it/)  
59. Practical Implementation of the Kelly Criterion: Optimal Growth Rate, Number of Trades, and Rebalancing Frequency for Equity Portfolios \- Frontiers, accessed February 26, 2026, [https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2020.577050/full](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2020.577050/full)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAXCAYAAABefIz9AAADbUlEQVR4XrVWTYhOURh+T0NRSpgGRYb8LCQLYWNJWWAhRZmFspkmGynTWM0kCztJKWUxCytrFkiflAXSEJKfmrEkC4X8pPE83znn3nPee8937sU89XznO+/vOec9P1ekIUzLvkZOb5GySsnTaO+RQSpgSl6PnHVOPweIU1YHUJX8Df5PlDmBHprta2lPLAJfg5vAe+BkrO6CAY+BO0NBhEQltofSHlgNngIvgvtLcSWNxglwMOj3gVvABYFsGDzi/m8DvyLuuJR+88Fb4KykE6bkMur/JCwoHgM3BLIhscmGk14lPom1dTS/0Z4O9CvBGXBVIDsX/Pd4J0H1SmTzm2KCCfQjxAu0z8EBF5CD4YAfsZNJMQ1+ADmxN2InFMJVLJLrMS0Gx0w2VRfeprB1wZK+3B6XwZtizwot16GZdRMvUIlsMSV2EiksEbtQW12f7tdKtewQWz0dNw1lqFarUZgDYis4ruQ1MFOm9wSZ8BK42/XXgs9KtdwRe0QSqIy3IhitSOrBS4Hn8KHYyS2N1Um8BJ+Cx8FDYn2vl+oi+1vwi9gJrXAyVo+UmnFX8N7SuLYgg2oZudH5eXCFKf8JPgb78im7mAGvBP0Lxl423PoheLsOuJbgubsNGpeHZ3QCHAEX1k14uVgjzbM1MtIn0kDwbqIHEt98CsUA9khc7V3gL3BvIIvgPLktWU2Pj+ANx3DBsshdMin4qz8LFdk/Cx1xl1YNWD0+Jd6Vu4fvr8dRcF7Qj6ES6itZg5XkquuK+gm6B1svULfPgfJ5OBkokhN0EdiwevT1OCPxOHlpcUc2Qm6CHbETmVTyugquF/useHAgtOEt6cEvp2+SvoE74j4EgiVj7nCcfKa4UPVoWUEOjoM8GMi48pT9CGQ8a6iW4Zvptw+qYPipFqZkn76bA5k4E/7w3HWrZ526v7qCXKT+oN+FX/EkTdQ3/uElGJBb7S5439kclnjga8An4JAXunYE7Xe0V4190D+D+wK9ByfVCQWBngvKx35QbB4+O82hEqXA7TYh9rAnb8AElom9+ehbbGGVlxXiB3UBpeen4itIp9Gep0DpG06jBs08m1mFaOqRsrPylLYF4hA6oO6nkLfLW8So2HtBRdEKpXfviefR3qMldALdnyvohc7l1QuZsy+QM8zp/w1to6ft/wDcw5TRkUbcuAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAAAZCAYAAACb1MhvAAADMklEQVR4Xu2XO4gUQRCGq1FBUXzjcVxwqIegCIqikYGBgSYmBuYm5gbKGYlgYCqCYGIkGpgdZoILJsIFJpoIwiGiIByCcInio/6p6d2empqZ7t5ZFw4/+Jmd6q6a6pp+zBL9E5w2dJDavz96erIOo+9j6fLrak8nL2KEV0SXBPqK5lSk/Lj5nhPFp6XT0/d1WnpsYM1y+yyuhmZavSMZP8L4YDBbqqbktE6y1lh/RK68VrSWEXfamAljsCiagdnf4jnrF8ksC7lOUqwHrE0wNC3d6CeNS1MCCbQULJovrEX8UBmgiChYUaz1glGwpMKj8y3lcZr1jfWWNT8yJ8UtSPewOcQ6po2ZcMHcsGAZCW5mHVS29yQzKyjWdEEyL4Lh4ccj1s3QEEmlYDbR0XZQkVexFC+ptgT888w97jDVD5RSxWFzd9RV2Fg23Atse1nvWOcCm2Y766Ohr6zPhh1KxD0kSX6MYrVynCSvRScvZw/rMesJFXWxOcNaJam05xpJolsDm4kxV4I9zGiNA47+RETRAjPdHt038prE94JuUKDPQNkOkHzSYByaIgEUZ8DaFgzwGUmwHIxNPxnMKCxDFCs8ETHz3wT3TcQUDDMKfa4q+ynWqqtOoCGYdihOuBzBB2otWOvM6aNgOBHxfAwqwJ2nuILhEMOY2j4/cMh9L69DnBRwmbXLGieW40+qTz8k+4od5py8rRTGKdg8yYkILRgJY9Zd1EajXw3pUemH/XlAxcoa4v9hLAS2CjdIioNp6EHS/hBAci+DthL957hCUTC73bYG4ETEs/HdFYIv/iusFdZctSmHIo/9rGVXzKQhSyQvpTHTFZIEf5Ocbtj8L5N8IMLxKZVvwAiwm8S3Va5630ARHftNzV9JzS4jqzTwyYCxf2L9YB3xDU2RsRzxN2Qn6wSN1jze6L7ymkDTY7oY+eVGaCIiHibE0fLaCjZ8vLU7umH9US+bt9RbqoTtMyRLxthEuwNNGzs/bfV7rbbncZZ1n+RLXzVputonT2oGTf2b7L2hH6Dvhbq1bonD+7X7t7eOiO3XRXacbMf/9M2kX4XEn/RToplUIv3FzY3kC/0X57eROMOalrgAAAAASUVORK5CYII=>