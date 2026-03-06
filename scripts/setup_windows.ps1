# ============================================================
# twstock-predictor Windows 環境建置腳本
# 用法: 在 PowerShell 中執行
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#   .\scripts\setup_windows.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$TARGET = "D:\twstock-predictor"

Write-Host "=== twstock-predictor Windows Setup ===" -ForegroundColor Cyan

# ── 1. 檢查 Python 3.12+ ───────────────────────────────
$pyCmd = $null
foreach ($cmd in @("python3", "python")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "3\.(1[2-9]|[2-9]\d)") {
            $pyCmd = $cmd
            Write-Host "[OK] 找到 $ver" -ForegroundColor Green
            break
        }
    } catch {}
}
if (-not $pyCmd) {
    Write-Host "[ERROR] 需要 Python >= 3.12" -ForegroundColor Red
    Write-Host "  請從 https://www.python.org/downloads/ 安裝" -ForegroundColor Yellow
    exit 1
}

# ── 2. 檢查 uv（若無則安裝）──────────────────────────────
$uvExists = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvExists) {
    Write-Host "[INFO] 安裝 uv 套件管理器..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    # 重新載入 PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
}
Write-Host "[OK] uv: $(uv --version)" -ForegroundColor Green

# ── 3. 複製專案到 D 槽 ──────────────────────────────────
if (Test-Path $TARGET) {
    Write-Host "[WARN] $TARGET 已存在，跳過複製" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] 從 WSL 複製專案到 $TARGET ..." -ForegroundColor Yellow

    # 方法 A: 如果 git 可用，直接 clone（最乾淨）
    $wslPath = "\\wsl$\Ubuntu\home\ainsley\project\twstock-predictor"
    if (Test-Path $wslPath) {
        Copy-Item -Path $wslPath -Destination $TARGET -Recurse -Force
        # 移除 WSL 的 .venv（Windows 需要重建）
        $oldVenv = Join-Path $TARGET ".venv"
        if (Test-Path $oldVenv) {
            Remove-Item $oldVenv -Recurse -Force
        }
        Write-Host "[OK] 已複製專案" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] 找不到 WSL 路徑: $wslPath" -ForegroundColor Red
        Write-Host "  請手動複製，或修改腳本中的 WSL 發行版名稱" -ForegroundColor Yellow
        Write-Host "  例如: \\wsl`$\Ubuntu-22.04\home\ainsley\project\twstock-predictor" -ForegroundColor Yellow
        exit 1
    }
}

Set-Location $TARGET

# ── 4. 建立 venv + 安裝依賴 ──────────────────────────────
Write-Host "[INFO] 建立虛擬環境 + 安裝依賴..." -ForegroundColor Yellow
uv venv --python 3.12
uv pip install -e ".[dev,analysis]"
Write-Host "[OK] 依賴安裝完成" -ForegroundColor Green

# ── 5. 建立 .env（如果不存在）─────────────────────────────
$envFile = Join-Path $TARGET ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $TARGET ".env.example") $envFile
    Write-Host "[WARN] 已建立 .env，請填入 API Keys" -ForegroundColor Yellow
}

# ── 6. 建立 data 目錄 ───────────────────────────────────
$dataDir = Join-Path $TARGET "data"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
}

# ── 7. 驗證 ─────────────────────────────────────────────
Write-Host ""
Write-Host "=== 驗證安裝 ===" -ForegroundColor Cyan
& (Join-Path $TARGET ".venv\Scripts\python.exe") -c "import torch; import xgboost; print(f'torch={torch.__version__}, xgboost={xgboost.__version__}')"

Write-Host ""
Write-Host "=== 設定完成！===" -ForegroundColor Green
Write-Host ""
Write-Host "後續步驟:" -ForegroundColor Cyan
Write-Host "  1. 編輯 .env 填入 API Keys:" -ForegroundColor White
Write-Host "     notepad $envFile" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. 啟動前端 (Next.js):" -ForegroundColor White
Write-Host "     cd $TARGET\web" -ForegroundColor Gray
Write-Host "     npm install" -ForegroundColor Gray
Write-Host "     npm run dev" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. 執行測試:" -ForegroundColor White
Write-Host "     .venv\Scripts\python -m pytest tests\ -v" -ForegroundColor Gray
Write-Host ""
