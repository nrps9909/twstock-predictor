"""共用 LLM client — Anthropic SDK 優先，claude -p CLI 作為 fallback

優先級：
1. Anthropic Python SDK（需要有效的 ANTHROPIC_API_KEY）
2. claude -p CLI（使用 Claude Code 訂閱認證）

Server 啟動時（api/main.py）會移除 CLAUDECODE 環境變數，
確保 claude -p 不會被巢狀 session 偵測擋住。
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess

logger = logging.getLogger(__name__)

# ── 嘗試初始化 Anthropic SDK ───────────────────────────
_sync_client = None
_async_client = None
_use_sdk = False

try:
    import anthropic
    from src.utils.config import settings

    _api_key = settings.ANTHROPIC_API_KEY
    if _api_key:
        _sync_client = anthropic.Anthropic(api_key=_api_key)
        _async_client = anthropic.AsyncAnthropic(api_key=_api_key)
        _use_sdk = True
        logger.info("llm_client: Anthropic SDK 已初始化 (key=%s...)", _api_key[:12])
except ImportError:
    logger.info("llm_client: anthropic SDK 未安裝")

# ── Claude CLI ─────────────────────────────────────────
_CLAUDE_PATH: str = shutil.which("claude") or "claude"


def _clean_env() -> dict[str, str]:
    """清理環境變數供 claude CLI 使用"""
    env = os.environ.copy()
    for key in list(env):
        if "CLAUDECODE" in key or "CLAUDE_CODE" in key:
            del env[key]
    env.pop("ANTHROPIC_API_KEY", None)
    env["BROWSER"] = "0"
    # Windows 關鍵系統變數
    userprofile = env.get("USERPROFILE", r"C:\Users\Default")
    for k, v in {
        "SYSTEMROOT": r"C:\WINDOWS",
        "USERPROFILE": userprofile,
        "APPDATA": os.path.join(userprofile, r"AppData\Roaming"),
        "LOCALAPPDATA": os.path.join(userprofile, r"AppData\Local"),
    }.items():
        if k not in env:
            env[k] = v
    local = env.get("LOCALAPPDATA", os.path.join(userprofile, r"AppData\Local"))
    for k in ("TEMP", "TMP"):
        if k not in env:
            env[k] = os.path.join(local, "Temp")
    return env


def _call_cli_sync(
    prompt: str,
    model: str,
    timeout: float,
    system_prompt: str,
) -> str:
    """透過 claude -p CLI 呼叫（每次呼叫直接嘗試，不預設閘門）"""
    # prompt 作為命令行參數（不用 stdin）
    cmd = [
        _CLAUDE_PATH, "-p",
        "--output-format", "json",
        "--max-turns", "1",
        "--model", model,
        "--system-prompt", system_prompt,
        prompt,
    ]
    env = _clean_env()

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        out_text = result.stdout.decode("utf-8", errors="replace").strip()
        err_text = result.stderr.decode("utf-8", errors="replace").strip()

        if result.returncode != 0:
            raise RuntimeError(
                f"claude -p 失敗 (code {result.returncode}): "
                f"{err_text or out_text or '(no output)'}"
            )

        # Parse JSON wrapper: {"type":"result","result":"..."}
        if out_text:
            try:
                data = json.loads(out_text)
                if isinstance(data, dict) and "result" in data:
                    if data.get("is_error"):
                        raise RuntimeError(f"claude -p 錯誤: {data.get('result')}")
                    return data["result"]
            except json.JSONDecodeError:
                pass
        return out_text

    except subprocess.TimeoutExpired:
        raise TimeoutError(f"claude -p 逾時 ({timeout}s)")


def _call_sdk_sync(
    prompt: str,
    model: str,
    timeout: float,
    system_prompt: str,
) -> str:
    """透過 Anthropic SDK 呼叫"""
    response = _sync_client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    return response.content[0].text


async def _call_sdk_async(
    prompt: str,
    model: str,
    timeout: float,
    system_prompt: str,
) -> str:
    """透過 Anthropic AsyncClient 呼叫"""
    response = await asyncio.wait_for(
        _async_client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=timeout,
    )
    return response.content[0].text


# ── Public API ─────────────────────────────────────────

async def call_claude(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 60,
    system_prompt: str = "You are a JSON API. Return ONLY valid JSON. No markdown fences, no explanation.",
) -> str:
    """Async: 呼叫 Claude（SDK 優先，CLI fallback）

    預設 timeout 60s（CLI 冷啟動約需 20-40s）。
    """
    global _use_sdk

    if _use_sdk:
        try:
            return await _call_sdk_async(prompt, model, timeout, system_prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str:
                logger.warning("SDK 認證失敗，切換 CLI: %s", e)
                _use_sdk = False
            else:
                raise

    return await asyncio.to_thread(
        _call_cli_sync, prompt, model, timeout, system_prompt,
    )


def call_claude_sync(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 60,
    system_prompt: str = "You are a JSON API. Return ONLY valid JSON. No markdown fences, no explanation.",
) -> str:
    """Sync: 呼叫 Claude（SDK 優先，CLI fallback）

    預設 timeout 60s（CLI 冷啟動約需 20-40s）。
    """
    global _use_sdk

    if _use_sdk:
        try:
            return _call_sdk_sync(prompt, model, timeout, system_prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "authentication" in err_str or "401" in err_str:
                logger.warning("SDK 認證失敗，切換 CLI: %s", e)
                _use_sdk = False
            else:
                raise

    return _call_cli_sync(prompt, model, timeout, system_prompt)


def check_claude_available() -> bool:
    """檢查 LLM 是否可用（server 啟動時呼叫一次）

    只做輕量檢查（SDK ping / CLI --version），不做完整 API 呼叫。
    實際 API 可用性由第一次真實呼叫決定。
    """
    global _use_sdk

    # 1. 試 SDK（輕量 ping）
    if _sync_client is not None:
        try:
            response = _sync_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                messages=[{"role": "user", "content": "ping"}],
                timeout=15,
            )
            _use_sdk = True
            logger.info("LLM 就緒: Anthropic SDK (model: %s)", response.model)
            return True
        except Exception as exc:
            logger.warning("Anthropic SDK 驗證失敗: %s", exc)
            _use_sdk = False

    # 2. CLI: 只檢查 --version（快速，不做 API 呼叫）
    try:
        result = subprocess.run(
            [_CLAUDE_PATH, "--version"],
            capture_output=True,
            timeout=10,
            env=_clean_env(),
        )
        version = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode == 0:
            logger.info(
                "LLM 就緒: claude CLI %s (首次 API 呼叫可能較慢)",
                version,
            )
            return True
    except FileNotFoundError:
        logger.warning("claude CLI 未找到 (path: %s)", _CLAUDE_PATH)
    except Exception as exc:
        logger.warning("claude CLI 檢查失敗: %s", exc)

    logger.warning(
        "LLM 不可用: SDK 認證失敗且 CLI 未找到。"
        "請確認 .env 中的 ANTHROPIC_API_KEY 有效，"
        "或安裝 claude CLI。"
    )
    return False


def parse_json_response(text: str) -> dict | list:
    """從 LLM 回應文字中解析 JSON（含 markdown fence 處理）"""
    text = text.strip()
    if not text:
        raise ValueError("LLM 回傳空白文字")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    for pattern in [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"無法從回應中解析 JSON: {text[:200]}")
