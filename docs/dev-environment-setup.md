# Development Environment Setup Guide

Complete guide to reproduce the Windows development environment with PowerShell + Claude Code.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Core Tools Installation](#2-core-tools-installation)
3. [Windows Terminal Configuration](#3-windows-terminal-configuration)
4. [PowerShell Configuration](#4-powershell-configuration)
5. [Git Global Configuration](#5-git-global-configuration)
6. [Claude Code Installation & Configuration](#6-claude-code-installation--configuration)
7. [npm Global Packages](#7-npm-global-packages)
8. [Python Global Packages](#8-python-global-packages)
9. [Chrome Extensions](#9-chrome-extensions)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. System Requirements

- **OS**: Windows 11
- **GPU**: NVIDIA GPU (for CUDA support)
- **Disk**: Recommend SSD with at least 50 GB free
- **RAM**: 16 GB+

---

## 2. Core Tools Installation

Install in the following order.

### 2.1 Git 2.45+

Download from https://git-scm.com/download/win

- Install to default path (`C:\Program Files\Git\`)
- Enable Git LFS during installation
- Verify:

```powershell
git --version
# Expected: git version 2.45.1.windows.1+
git lfs version
```

### 2.2 Python 3.11

Download **Python 3.11.x** from https://www.python.org/downloads/

- **Custom install** to `C:\Python311\`
- Check "Add Python to PATH"
- Check "Install for all users"
- Verify:

```powershell
python --version
# Expected: Python 3.11.x
where python
# Expected: C:\Python311\python.exe
```

### 2.3 Node.js v22 (via fnm)

Install **fnm** (Fast Node Manager) first via Scoop or standalone:

```powershell
# Option A: via Scoop (if Scoop is installed)
scoop install fnm

# Option B: via winget
winget install Schniz.fnm
```

Configure fnm in PowerShell profile (add to `$PROFILE`):

```powershell
fnm env --use-on-cd --shell power-shell | Out-String | Invoke-Expression
```

Then install Node.js:

```powershell
fnm install 22
fnm default 22
```

Or install Node.js directly from https://nodejs.org/ to `C:\Program Files\nodejs\`.

Verify:

```powershell
fnm --version
# Expected: fnm 1.12.x
node --version
# Expected: v22.x.x
npm --version
```

### 2.4 uv 0.10+ (Python Package Manager)

Install via pip:

```powershell
pip install uv
```

Or via standalone installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
uv --version
# Expected: uv 0.10.x
where uv
# Expected: C:\Python311\Scripts\uv.exe
```

### 2.5 Docker Desktop 27+

Download from https://www.docker.com/products/docker-desktop/

- Enable WSL 2 backend during installation
- Verify:

```powershell
docker --version
# Expected: Docker version 27.x.x
docker compose version
```

### 2.6 NVIDIA CUDA 12.5 + cuDNN v9.3

1. **CUDA Toolkit 12.5**: Download from https://developer.nvidia.com/cuda-12-5-0-download-archive
   - Install to default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\`
2. **cuDNN v9.3**: Download from https://developer.nvidia.com/cudnn (requires NVIDIA account)
   - Install to: `C:\Program Files\NVIDIA\CUDNN\v9.3\`
   - Ensure `C:\Program Files\NVIDIA\CUDNN\v9.3\bin` is in system PATH

Verify:

```powershell
nvcc --version
# Expected: Build cuda_12.5.r12.5/compiler...
nvidia-smi
```

---

## 3. Windows Terminal Configuration

### 3.1 Install Font

Download and install **FiraCode Nerd Font** from https://www.nerdfonts.com/font-downloads

- Extract zip, select all `.ttf` files, right-click > "Install for all users"

### 3.2 Settings

Open Windows Terminal > Settings (Ctrl+,) > Open JSON file.

Replace `settings.json` at:
`%LOCALAPPDATA%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json`

```json
{
    "$help": "https://aka.ms/terminal-documentation",
    "$schema": "https://aka.ms/terminal-profiles-schema",
    "actions":
    [
        {
            "command":
            {
                "action": "copy",
                "singleLine": false
            },
            "id": "User.copy.644BA8F2"
        },
        {
            "command": "paste",
            "id": "User.paste"
        },
        {
            "command":
            {
                "action": "splitPane",
                "split": "auto",
                "splitMode": "duplicate"
            },
            "id": "User.splitPane.A6751878"
        },
        {
            "command": "find",
            "id": "User.find"
        }
    ],
    "copyFormatting": "none",
    "copyOnSelect": false,
    "defaultProfile": "{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
    "keybindings":
    [
        {
            "id": "User.copy.644BA8F2",
            "keys": "ctrl+c"
        },
        {
            "id": "User.find",
            "keys": "ctrl+shift+f"
        },
        {
            "id": "User.paste",
            "keys": "ctrl+v"
        },
        {
            "id": "User.splitPane.A6751878",
            "keys": "alt+shift+d"
        }
    ],
    "newTabMenu":
    [
        {
            "type": "remainingProfiles"
        }
    ],
    "profiles":
    {
        "defaults":
        {
            "font":
            {
                "face": "FiraCode Nerd Font"
            },
            "opacity": 75,
            "useAcrylic": false
        },
        "list":
        [
            {
                "commandline": "%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "guid": "{61c54bbd-c2c6-5271-96e7-009a87ff44bf}",
                "hidden": false,
                "name": "Windows PowerShell"
            },
            {
                "commandline": "%SystemRoot%\\System32\\cmd.exe",
                "guid": "{0caa0dad-35be-5f56-a8ff-afceeeaa6101}",
                "hidden": false,
                "name": "命令提示字元"
            }
        ]
    },
    "schemes": [],
    "themes": []
}
```

Key settings:
- **Font**: FiraCode Nerd Font
- **Opacity**: 75% (no acrylic blur)
- **Copy format**: none (plain text only)
- **Shortcuts**: Ctrl+C/V copy/paste, Ctrl+Shift+F find, Alt+Shift+D split pane

---

## 4. PowerShell Configuration

### 4.1 Execution Policy

Run PowerShell **as Administrator**:

```powershell
Set-ExecutionPolicy Unrestricted -Scope LocalMachine
```

Verify:

```powershell
Get-ExecutionPolicy -List
# LocalMachine should be "Unrestricted"
```

### 4.2 Profile

Find your profile path:

```powershell
echo $PROFILE
# Usually: C:\Users\<username>\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1
```

Create/edit the profile:

```powershell
notepad $PROFILE
```

Paste the following content:

```powershell
function cc { claude --dangerously-skip-permissions @args }

Set-Location D:\
```

This sets:
- `cc` alias: runs Claude Code in dangerous mode (skip permission prompts)
- Default directory: `D:\`

---

## 5. Git Global Configuration

Run the following commands:

```powershell
git config --global user.name "陳廷安"
git config --global user.email "73953029+nrps9909@users.noreply.github.com"
git config --global core.editor "\"C:\Users\$env:USERNAME\AppData\Local\Programs\Microsoft VS Code\bin\code\" --wait"

# Git LFS
git lfs install
```

Verify:

```powershell
git config --global --list
```

Expected `.gitconfig` content:

```ini
[filter "lfs"]
    clean = git-lfs clean -- %f
    smudge = git-lfs smudge -- %f
    process = git-lfs filter-process
    required = true
[user]
    name = 陳廷安
    email = 73953029+nrps9909@users.noreply.github.com
[core]
    editor = "C:\Users\<username>\AppData\Local\Programs\Microsoft VS Code\bin\code" --wait
```

---

## 6. Claude Code Installation & Configuration

### 6.1 Install Claude Code

```powershell
npm install -g @anthropic-ai/claude-code
```

Verify:

```powershell
claude --version
```

### 6.2 Global Settings

Create `~/.claude/settings.json` (i.e. `C:\Users\<username>\.claude\settings.json`):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python C:/Users/<username>/.claude/hooks/guard-dangerous.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python C:/Users/<username>/.claude/hooks/auto-format.py"
          }
        ]
      }
    ]
  },
  "statusLine": {
    "type": "command",
    "command": "ccusage statusline | sed 's/ | 🔥.*//' "
  },
  "enabledPlugins": {
    "feature-dev@claude-plugins-official": true,
    "context7@claude-plugins-official": true,
    "code-review@claude-plugins-official": true,
    "frontend-design@claude-plugins-official": true
  },
  "alwaysThinkingEnabled": true,
  "autoUpdatesChannel": "latest",
  "skipDangerousModePermissionPrompt": true,
  "effortLevel": "high"
}
```

> Replace `<username>` with your Windows username.

### 6.3 Local Settings

Create `~/.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(powershell:*)"
    ]
  }
}
```

### 6.4 Global Instructions

Create `~/.claude/CLAUDE.md`:

```markdown
# Global Instructions

## Language & Communication
- Communicate with user in Traditional Chinese (zh-TW)
- Write all code, comments, commit messages, and variable names in English
- Keep responses concise; avoid unnecessary explanations

## Python Conventions
- Use Python 3.11+ features (type hints, match-case, f-strings)
- Prefer `pathlib.Path` over `os.path`
- Use `ruff` for formatting and linting
- Follow PEP 8 naming: `snake_case` for functions/variables, `PascalCase` for classes

## TypeScript / JavaScript Conventions
- Prefer `const` over `let`; never use `var`
- Use TypeScript strict mode when applicable
- Use `prettier` for formatting

## Anti-Patterns (NEVER do these)
- Do NOT add unused imports or dead code
- Do NOT wrap single-use logic in unnecessary abstractions
- Do NOT add try/except around code that cannot fail
- Do NOT create files unless absolutely necessary
- Do NOT add comments that merely restate the code

## Context Management
- When context is getting long, summarize findings before continuing
- Prefer targeted file reads over reading entire large files
- Use subagents for independent research tasks to protect main context
```

### 6.5 Hooks

Create directory `~/.claude/hooks/`.

#### `~/.claude/hooks/auto-format.py`

```python
"""PostToolUse hook: auto-format written/edited files."""

import sys
import json
import os
import subprocess

data = json.load(sys.stdin)
fp = data.get("tool_input", {}).get("file_path", "")
if not fp or not os.path.isfile(fp):
    sys.exit(0)

ext = os.path.splitext(fp)[1].lower()

if ext in (".py", ".pyi"):
    try:
        from ruff_api import format_string
        from pathlib import Path

        p = Path(fp)
        code = p.read_text(encoding="utf-8")
        formatted = format_string(fp, code)
        if formatted != code:
            p.write_text(formatted, encoding="utf-8")
    except Exception:
        pass  # formatting failure should not block

elif ext in (".js", ".ts", ".tsx", ".jsx", ".css", ".json", ".yaml", ".yml"):
    try:
        subprocess.run(
            ["npx", "prettier", "--write", fp],
            capture_output=True,
            timeout=15,
        )
    except Exception:
        pass  # formatting failure should not block
```

#### `~/.claude/hooks/guard-dangerous.py`

```python
"""PreToolUse hook: block dangerous shell commands."""
import sys
import json

data = json.load(sys.stdin)
cmd = data.get("tool_input", {}).get("command", "")

BLOCKED = [
    "rm -rf /",
    "rm -rf ~",
    "push --force",
    "push -f",
    "--force push",
    "force push",
    "DROP TABLE",
    "DROP DATABASE",
    "reset --hard",
    "checkout -- .",
    "clean -fd",
    "> /dev/sda",
]

for pattern in BLOCKED:
    if pattern in cmd:
        print(f"BLOCKED: command contains '{pattern}'", file=sys.stderr)
        sys.exit(2)
```

### 6.6 Agents

Create directory `~/.claude/agents/`.

#### `~/.claude/agents/code-reviewer.md`

```markdown
# Code Reviewer Agent

You are a focused code review agent. Your job is to find **real bugs, security issues, and correctness problems** - not style opinions.

## Review Checklist
1. **Bugs**: logic errors, off-by-one, null/undefined access, race conditions
2. **Security**: injection (SQL/XSS/command), hardcoded secrets, auth bypass, path traversal
3. **Data issues**: missing validation at boundaries, silent data loss, type coercion bugs
4. **Error handling**: unhandled promise rejections, bare except clauses, swallowed errors
5. **Performance**: N+1 queries, unbounded loops, missing pagination, memory leaks

## Output Format
For each finding, provide:
- **File and line**: exact location
- **Severity**: critical / warning / info
- **Issue**: one-line description
- **Fix**: concrete suggestion

## Rules
- Skip style/formatting opinions (formatters handle that)
- Skip suggestions that are "nice to have" but not actual issues
- Do NOT suggest adding comments or docstrings
- Do NOT suggest renaming for personal preference
- Focus on code that was CHANGED, not surrounding unchanged code
- If you find nothing significant, say "No issues found" - don't invent problems
```

#### `~/.claude/agents/explorer.md`

````markdown
# Project Explorer Agent

You are a systematic project exploration agent. Your job is to quickly understand an unfamiliar codebase and produce a structured summary.

## Exploration Steps
1. **Root files**: Read `package.json`, `pyproject.toml`, `Cargo.toml`, or equivalent for dependencies and scripts
2. **Entry points**: Find `main`, `index`, `app`, or `server` files
3. **Directory structure**: Map top-level directories and their purposes
4. **Configuration**: Check for `.env.example`, config files, CI/CD pipelines
5. **Architecture**: Identify patterns (MVC, layered, monorepo, microservices)

## Output Format
```
## Project: {name}
**Type**: {web app / CLI / library / API / monorepo}
**Stack**: {languages, frameworks, key dependencies}
**Entry point**: {main file(s)}

### Structure
- `src/` - {purpose}
- `tests/` - {purpose}
...

### Key Patterns
- {pattern 1}
- {pattern 2}

### Build & Run
- Install: {command}
- Dev: {command}
- Test: {command}

### Notes
- {anything unusual or important}
```

## Rules
- Read only what's necessary; don't read every file
- Prioritize understanding structure over implementation details
- Note any potential issues (missing tests, no CI, outdated deps)
- Keep the summary under 50 lines
````

### 6.7 Skills

Create directory `~/.claude/skills/`.

#### `~/.claude/skills/testing-patterns.md`

````markdown
---
name: testing-patterns
description: Testing best practices for Python (pytest) and JavaScript (Jest/Vitest)
globs: ["**/*test*", "**/*spec*", "**/tests/**", "**/test/**", "**/__tests__/**"]
---

# Testing Patterns

## General Principles
- Follow **AAA pattern**: Arrange, Act, Assert
- One logical assertion per test (multiple asserts OK if testing one behavior)
- Test names describe behavior: `test_user_cannot_login_with_expired_token`
- Never test implementation details; test observable behavior

## pytest (Python)
```python
# Use fixtures for setup, parametrize for variations
@pytest.fixture
def sample_user(db_session):
    return UserFactory.create()

@pytest.mark.parametrize("status,expected", [
    (200, True),
    (404, False),
])
def test_api_response(status, expected):
    assert is_success(status) == expected
```
- Use `tmp_path` fixture instead of creating temp files manually
- Use `monkeypatch` instead of `unittest.mock.patch` when possible
- Group related tests in classes: `class TestUserRegistration:`

## Jest / Vitest (TypeScript)
```typescript
describe("calculateTotal", () => {
  it("applies discount when quantity exceeds threshold", () => {
    const items = [{ price: 100, qty: 5 }];
    const result = calculateTotal(items);
    expect(result).toBe(450);
  });
});
```
- Use `describe` blocks to group related tests
- Prefer `toEqual` for objects, `toBe` for primitives
- Use `vi.fn()` / `jest.fn()` for mocks; reset in `beforeEach`

## Anti-Patterns
- Do NOT test private/internal functions directly
- Do NOT use `sleep` / fixed delays; use proper async waiting
- Do NOT mock what you don't own; wrap external deps first
- Do NOT write tests that pass when the code is broken
````

#### `~/.claude/skills/git-workflow.md`

```markdown
---
name: git-workflow
description: Git branching strategy, commit conventions, and PR guidelines
globs: ["**/.gitignore", "**/.gitattributes"]
---

# Git Workflow

## Branch Naming
- `feat/short-description` - new features
- `fix/issue-number-description` - bug fixes
- `refactor/description` - code restructuring
- `docs/description` - documentation only

## Commit Messages
Follow Conventional Commits format:
```
type(scope): concise description

Optional body explaining WHY, not WHAT.
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

## Pre-Commit Checklist
1. Run tests: `pytest` or `npm test`
2. Check for debug artifacts: `console.log`, `print()`, `debugger`
3. Ensure no secrets in staged files (.env, credentials, API keys)
4. Review diff: `git diff --staged`

## PR Guidelines
- Title: short, imperative mood (< 70 chars)
- Body: Summary (what & why), Test Plan (how verified)
- Keep PRs small and focused (< 400 lines changed ideally)
- One logical change per PR

## Safety Rules
- NEVER force push to main/master
- NEVER use `--no-verify` to skip hooks
- NEVER amend published commits
- Prefer `git revert` over `git reset --hard` for shared branches
- Always create new commits when fixing hook failures
```

#### `~/.claude/skills/api-design.md`

````markdown
---
name: api-design
description: REST API design patterns for FastAPI and Next.js
globs: ["**/api/**", "**/routes/**", "**/routers/**", "**/endpoints/**"]
---

# API Design Patterns

## REST Principles
- Use nouns for resources: `/users`, `/orders/{id}`
- HTTP methods: GET (read), POST (create), PUT (full update), PATCH (partial), DELETE
- Return appropriate status codes: 200, 201, 204, 400, 401, 403, 404, 422, 500
- Use pagination for list endpoints: `?page=1&limit=20`

## FastAPI (Python)
```python
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```
- Always use `response_model` for type safety
- Use `Depends()` for dependency injection
- Validate input with Pydantic models, not manual checks
- Use `status.HTTP_xxx` constants for clarity

## Next.js API Routes (App Router)
```typescript
import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const data = await fetchData(searchParams.get("q"));
  return NextResponse.json(data);
}
```
- Export named functions matching HTTP methods
- Use `NextResponse.json()` for responses
- Handle errors with try/catch and appropriate status codes

## Anti-Patterns
- Do NOT put business logic in route handlers; delegate to services
- Do NOT return raw database models; use response schemas
- Do NOT silently swallow errors; log and return proper error responses
````

#### `~/.claude/skills/refactoring.md`

```markdown
---
name: refactoring
description: Safe refactoring workflow and common techniques
globs: ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx"]
---

# Safe Refactoring

## Workflow (ALWAYS follow this order)
1. **Verify tests pass** before making any changes
2. **Make one small change** at a time
3. **Run tests** after each change
4. **Commit** working state before next refactoring step

## Common Techniques
- **Extract Function**: when a block does one identifiable thing
- **Inline Function**: when the body is as clear as the name
- **Rename Variable/Function**: when the name doesn't convey intent
- **Replace Magic Number**: use named constants
- **Simplify Conditional**: extract complex conditions into named booleans
- **Remove Dead Code**: delete unused functions, imports, variables

## When to Refactor
- Before adding a feature (make the change easy, then make the easy change)
- When you see duplicated logic (3+ occurrences = extract)
- When a function exceeds ~30 lines or has 4+ parameters

## What NOT to Do
- Do NOT refactor and change behavior simultaneously
- Do NOT refactor without test coverage
- Do NOT rename across the codebase without search-and-verify
- Do NOT "improve" code you're not otherwise modifying
- Do NOT add abstractions for fewer than 3 use cases
- Do NOT refactor during an urgent bug fix
```

### 6.8 Plugins

The following plugins are enabled in `settings.json` (they activate automatically):

| Plugin | Purpose |
|--------|---------|
| `feature-dev` | Guided feature development with architecture focus |
| `context7` | Up-to-date library documentation lookup |
| `code-review` | PR code review agent |
| `frontend-design` | Production-grade frontend interface design |

### 6.9 Status Line

The status line uses `ccusage` to display API usage. Configured in `settings.json`:

```json
"statusLine": {
    "type": "command",
    "command": "ccusage statusline | sed 's/ | 🔥.*//' "
}
```

---

## 7. npm Global Packages

```powershell
npm install -g ccusage@18.0.8
```

Verify:

```powershell
npm list -g --depth=0
# Should show: ccusage@18.0.8
```

---

## 8. Python Global Packages

```powershell
pip install ruff-api
```

This is required by the `auto-format.py` hook.

Verify:

```powershell
pip list | findstr ruff
# Should show: ruff-api 0.2.x
```

---

## 9. Chrome Extensions

### 9.1 Claude in Chrome

Install the **Claude in Chrome** extension for browser automation MCP.

This provides the `mcp__claude-in-chrome__*` tools for:
- Page reading, navigation, form input
- JavaScript execution
- Screenshot and GIF recording
- Tab management

### 9.2 Connected MCP Services

After installing Claude in Chrome, connect the following MCP servers:
- **Gmail** — provides `gmail_search_messages`, `gmail_read_message`, etc.
- **Google Calendar** — provides `gcal_list_events`, `gcal_create_event`, etc.

These are configured through the Claude in Chrome extension settings.

---

## 10. Verification Checklist

Run each command to confirm the environment is correctly set up:

```powershell
# 1. Git
git --version                    # 2.45+
git lfs version                  # git-lfs/3.x.x

# 2. Python
python --version                 # Python 3.11.x
where python                     # C:\Python311\python.exe

# 3. Node.js
node --version                   # v22.x.x
npm --version                    # 10.x.x

# 4. uv
uv --version                     # uv 0.10.x

# 5. Docker
docker --version                 # Docker version 27.x.x
docker compose version           # Docker Compose version v2.x.x

# 6. CUDA
nvcc --version                   # Build cuda_12.5
nvidia-smi                       # Shows GPU info

# 7. Claude Code
claude --version                 # Should print version

# 8. ccusage
ccusage --version                # 18.0.8

# 9. ruff-api
python -c "import ruff_api; print(ruff_api.__version__)"

# 10. PowerShell profile
powershell -Command "Get-Content $PROFILE"
# Should show: cc function + Set-Location D:\

# 11. Execution Policy
powershell -Command "Get-ExecutionPolicy -Scope LocalMachine"
# Should show: Unrestricted

# 12. Git config
git config --global user.name    # 陳廷安
git config --global user.email   # 73953029+nrps9909@users.noreply.github.com

# 13. Claude Code config files
ls ~/.claude/settings.json       # Should exist
ls ~/.claude/settings.local.json # Should exist
ls ~/.claude/CLAUDE.md           # Should exist
ls ~/.claude/hooks/              # auto-format.py, guard-dangerous.py
ls ~/.claude/agents/             # code-reviewer.md, explorer.md
ls ~/.claude/skills/             # 4 skill files
```

---

## Quick Setup Script

For convenience, here is a PowerShell script that creates all Claude Code config files at once.
Run this **after** installing all tools:

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\hooks"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\agents"
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\skills"

Write-Host "Directories created. Copy the config file contents from the sections above."
Write-Host "Required files:"
Write-Host "  ~/.claude/settings.json"
Write-Host "  ~/.claude/settings.local.json"
Write-Host "  ~/.claude/CLAUDE.md"
Write-Host "  ~/.claude/hooks/auto-format.py"
Write-Host "  ~/.claude/hooks/guard-dangerous.py"
Write-Host "  ~/.claude/agents/code-reviewer.md"
Write-Host "  ~/.claude/agents/explorer.md"
Write-Host "  ~/.claude/skills/testing-patterns.md"
Write-Host "  ~/.claude/skills/git-workflow.md"
Write-Host "  ~/.claude/skills/api-design.md"
Write-Host "  ~/.claude/skills/refactoring.md"
```
