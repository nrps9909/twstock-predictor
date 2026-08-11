# 貢獻指南

感謝你願意協助改善 twstock-predictor。為了讓維護者能快速檢查與合併變更，請讓每個 Pull Request 聚焦在單一問題，並附上可重現的驗證方式。

## 開始之前

- 先搜尋既有 Issue 與 Pull Request，避免重複工作。
- 修正錯誤時，請說明重現步驟、預期結果與實際結果；能加入回歸測試時，請一併加入。
- 大型重構、模型策略或評分權重調整，請先開 Issue 討論範圍與驗收方式。
- 安全性問題請依照 [SECURITY.md](SECURITY.md) 私下回報，不要在公開 Issue 揭露細節。

## 開發環境

後端需要 Python 3.12 以上版本，建議使用 [uv](https://docs.astral.sh/uv/) 管理環境與鎖定依賴。

```bash
git clone https://github.com/nrps9909/twstock-predictor.git
cd twstock-predictor
uv sync --locked --extra dev
```

完整測試指令如下：

```bash
uv run --no-sync python -m pytest tests -q
```

開發時也可以只執行相關測試檔：

```bash
uv run --no-sync python -m pytest tests/test_technical.py -q
```

測試必須能在沒有 API key、既有資料庫或網路連線的乾淨環境執行。FinMind、TWSE、yfinance、LLM 等外部服務應在測試邊界使用 mock 或 fixture 取代。

若修改 `pyproject.toml` 的依賴，請執行 `uv lock`，並將更新後的 `uv.lock` 一起提交。

前端變更另需 Node.js 18 以上版本：

```bash
cd web
npm ci
npm run build
```

## Pull Request 檢查清單

- 變更範圍單一且說明清楚。
- 新行為有測試，錯誤修正有回歸測試。
- `uv sync --locked --extra dev` 與完整測試均通過。
- 前端變更已通過 `npm run build`。
- 未提交 API key、`.env`、本機資料庫、下載資料或模型產物。
- 使用者可見的行為或設定變更已同步更新文件。

本專案提供研究與技術示範，不構成投資建議。請勿以通過測試取代對資料品質、回測偏誤與風險限制的審查。
