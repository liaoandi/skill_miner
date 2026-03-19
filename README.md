# skill_miner

Mine reusable skills from your AI agent session history.

You use Claude Code, Codex, or other AI coding agents daily. Over time, recurring workflows emerge — "diagnose service failures", "cross-agent code review", "run data pipelines". `skill_miner` finds these patterns by analyzing your session histories, extracts them as skill candidates, and generates ready-to-use SKILL.md files following the [Agent Skills](https://agentskills.io) open standard.

## Install

```bash
pip install skill-miner[anthropic]
```

Requires Python 3.11+ and `ANTHROPIC_API_KEY` environment variable.

## Usage

```bash
skill-miner scan      # Scan session history and extract skill candidates
skill-miner review    # Review candidates interactively, generate SKILL.md for accepted ones
skill-miner status    # Show detected agents and run history
```

Zero-config — auto-detects Claude Code, Codex, OpenClaw, and Gemini CLI. Run `skill-miner init` to customize.

## How it works

1. **scan** — Reads recent sessions (14 days), sends them to an LLM to identify recurring multi-step workflows, classifies each candidate as accept / observe / reject, and writes output to `~/.config/skill_miner/review_queue/`
2. **review** — Presents candidates one by one for your decision. Accepted candidates get a SKILL.md generated via LLM, ready to drop into `~/.claude/skills/` or any compatible tool
3. **observe** — Candidates without enough evidence carry forward to the next run, accumulating sessions until promoted or expired

## Privacy

All output is auto-redacted before writing — API keys, email addresses, IP addresses, and absolute paths are stripped.

## License

MIT

---

# skill_miner

从你的 AI agent 使用历史中挖掘可复用的 skill。

你每天都在用 Claude Code、Codex 等 AI 编程工具，日积月累会形成重复的工作流模式——"诊断服务故障"、"跨 agent 代码审查"、"跑数据流水线"。`skill_miner` 自动分析你的 session 历史，发现这些模式，提取为 skill 候选，并生成符合 [Agent Skills](https://agentskills.io) 开放标准的 SKILL.md 文件。

## 安装

```bash
pip install skill-miner[anthropic]
```

需要 Python 3.11+ 和 `ANTHROPIC_API_KEY` 环境变量。

## 使用

```bash
skill-miner scan      # 扫描 session 历史，提取 skill 候选
skill-miner review    # 交互式评审候选，为接受的候选生成 SKILL.md
skill-miner status    # 显示检测到的 agent 和运行历史
```

开箱即用——自动检测 Claude Code、Codex、OpenClaw、Gemini CLI。运行 `skill-miner init` 可自定义配置。

## 工作原理

1. **scan** — 读取近 14 天的 session，用 LLM 识别重复出现的多步骤工作流，将每个候选分类为 accept / observe / reject，结果写入 `~/.config/skill_miner/review_queue/`
2. **review** — 逐条展示候选供你决策。接受的候选会通过 LLM 生成 SKILL.md，可直接放入 `~/.claude/skills/` 或其他兼容工具
3. **observe** — 证据不足的候选会保留到下次运行，持续积累 session 证据，直到被提升或过期

## 隐私

所有输出在写入前自动脱敏——API key、邮箱、IP 地址、绝对路径均被清除。

## 许可

MIT
