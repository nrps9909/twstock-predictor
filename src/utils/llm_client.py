"""共用 LLM client — 僅使用 claude -p CLI（Claude Code 訂閱認證）

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
import threading
import time

logger = logging.getLogger(__name__)

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


def _kill_proc_tree(pid: int) -> None:
    """Windows: 殺死整個進程樹 (claude.cmd → node → ...)"""
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


def _call_cli_sync(
    prompt: str,
    model: str,
    timeout: float,
    system_prompt: str,
) -> str:
    """透過 claude -p CLI 呼叫，使用 watchdog timer 確保可靠超時。

    communicate(timeout=) 在 Windows 上有已知問題：TimeoutExpired 後
    子進程 (node.js) 繼續持有 pipe handle，導致清理永久阻塞。
    改用 threading.Timer watchdog：從獨立線程強制殺死整個進程樹，
    pipe handle 釋放後 communicate() 自然返回。
    """
    cmd = [
        _CLAUDE_PATH,
        "-p",
        "--output-format",
        "json",
        "--max-turns",
        "1",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
    ]
    env = _clean_env()

    logger.info(
        "claude -p: model=%s, len=%d, timeout=%.0fs",
        model,
        len(prompt),
        timeout,
    )
    t0 = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Watchdog: force-kill process tree + close pipes after timeout.
    # On Windows, orphaned node.js children can hold pipe handles open,
    # preventing communicate() from returning even after the parent is dead.
    # Closing pipes from the watchdog thread forces communicate() to error out.
    killed = threading.Event()

    def _force_kill():
        killed.set()
        logger.warning("claude -p watchdog 觸發: 強制終止 (pid=%s)", proc.pid)
        _kill_proc_tree(proc.pid)
        try:
            proc.kill()
        except OSError:
            pass
        # Force-close pipes to unblock communicate() on Windows
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe:
                try:
                    pipe.close()
                except OSError:
                    pass

    timer = threading.Timer(timeout, _force_kill)
    timer.daemon = True
    timer.start()

    try:
        out_bytes, err_bytes = proc.communicate(input=prompt.encode("utf-8"))
    except (OSError, ValueError):
        # Pipes closed by watchdog — expected on timeout
        out_bytes, err_bytes = b"", b""
    except Exception:
        if killed.is_set():
            raise TimeoutError(f"claude -p 逾時 ({timeout}s)")
        raise
    finally:
        timer.cancel()

    elapsed = time.perf_counter() - t0

    if killed.is_set():
        raise TimeoutError(f"claude -p 逾時 ({timeout}s)")

    out_text = out_bytes.decode("utf-8", errors="replace").strip()
    err_text = err_bytes.decode("utf-8", errors="replace").strip()

    logger.info(
        "claude -p 完成: rc=%s, len=%d, %.1fs",
        proc.returncode,
        len(out_text),
        elapsed,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p 失敗 (code {proc.returncode}): "
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


# ── Public API ─────────────────────────────────────────


async def call_claude(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 60,
    system_prompt: str = "You are a JSON API. Return ONLY valid JSON. No markdown fences, no explanation.",
) -> str:
    """Async: 呼叫 Claude（透過 CLI）

    預設 timeout 60s（CLI 冷啟動約需 20-40s）。
    """
    logger.info("call_claude: 提交到線程池 (model=%s, len=%d)", model, len(prompt))
    return await asyncio.to_thread(
        _call_cli_sync,
        prompt,
        model,
        timeout,
        system_prompt,
    )


def call_claude_sync(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 60,
    system_prompt: str = "You are a JSON API. Return ONLY valid JSON. No markdown fences, no explanation.",
) -> str:
    """Sync: 呼叫 Claude（透過 CLI）

    預設 timeout 60s（CLI 冷啟動約需 20-40s）。
    """
    return _call_cli_sync(prompt, model, timeout, system_prompt)


def check_claude_available() -> bool:
    """檢查 CLI 是否可用（server 啟動時呼叫一次）

    只做輕量檢查（CLI --version），不做完整 API 呼叫。
    """
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
        "LLM 不可用: claude CLI 未找到。"
        "請安裝 claude CLI (npm install -g @anthropic-ai/claude-code)。"
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
