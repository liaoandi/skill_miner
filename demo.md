# skill_miner demo

A real walkthrough of mining skills from 75 AI agent sessions.

## Step 1: Check detected agents

```
$ skill-miner status

Agents:
  [+] claude_code: 0 sessions
  [+] codex: 37 sessions
  [+] openclaw: 38 sessions
  [+] gemini: 0 sessions

No previous runs.
```

Zero-config auto-detected 4 agents, found 75 sessions across Codex and OpenClaw.

## Step 2: Scan and extract candidates

```
$ skill-miner scan

Loading sessions...
codex: loaded 37 sessions
openclaw: loaded 38 sessions
Total sessions loaded: 75
Extracting patterns via LLM...
Classifying...
Done: 5A / 2O / 0R

Accepted: 5
Observed: 2
Rejected: 0
  ~/.config/skill_miner/review_queue/2026-03-19_skill_candidates.json
  ~/.config/skill_miner/review_queue/2026-03-19_skill_candidates.md

Run `skill-miner review` to review candidates.
```

From 75 sessions, the LLM identified 7 recurring workflow patterns:
- **5 accepted** (enough evidence to become skills)
- **2 observed** (promising but need more evidence, will be re-evaluated next run)

### Extracted candidates

| Candidate | Sessions | Sources | Decision |
|-----------|----------|---------|----------|
| openclaw_health_diagnostic | 4 | codex | accept |
| vpn_connection_troubleshooting | 3 | codex | accept |
| financial_watchlist_monitoring | 3 | openclaw | accept |
| code_cross_review_workflow | 3 | codex | accept |
| session_data_analysis | 3 | codex, openclaw | accept |
| github_issue_management | 2 | codex | observe |
| repository_audit_cleanup | 2 | codex | observe |

## Step 3: Review and generate SKILL.md

```
$ skill-miner review

Reviewing 7 candidates from 2026-03-19_skill_candidates.json
Commands: a=accept, o=observe, r=reject, s=skip, q=quit

------------------------------------------------------------

[4/7] code_cross_review_workflow
  Recommendation: accept
  Summary: Implement cross-agent code review by having one agent write
           code, another review it, and loop the feedback back for
           iterative improvement.
  Evidence: 3 sessions, 1 sources
    [codex] 2026-03-10: Set up cross-agent review workflow between
                         claude and codex
    [codex] 2026-03-16: Review PR code across multiple files before
                         submission
    [codex] 2026-03-17: Strict bug-finding review of Browser Relay
                         patch and PR

  Decision: a

Generating SKILL.md for 1 accepted candidates...
  Generated: ~/.config/skill_miner/review_queue/skills/
             code_cross_review_workflow/SKILL.md

============================================================
Done: 1 accepted, 0 observed, 0 rejected
```

## Generated SKILL.md

```yaml
---
name: code_cross_review_workflow
description: Use this skill when you need to implement a cross-agent
  code review process where one agent writes code, another reviews it,
  and feedback is looped back for iterative improvement.
---

# Code Cross Review Workflow

## When to Use

- When multiple agents are available for collaborative code development
- Before submitting pull requests or merging code changes
- When strict code quality standards need to be enforced

## Steps

1. **Designate Agent Roles**
   - Assign one agent as the "author" (writes/modifies code)
   - Assign another agent as the "reviewer" (performs code review)

2. **Author Phase**
   - Author agent writes or modifies code files
   - Documents changes and implementation decisions

3. **Review Phase**
   - Reviewer agent examines all relevant files
   - Checks for: logic errors, style, performance, security

4. **Feedback Loop**
   - Reviewer provides specific, actionable feedback
   - Author addresses comments and makes changes
   - Process repeats until reviewer approves

5. **Quality Gates**
   - Run tests and linting
   - Verify all feedback addressed
   - Prepare final submission
```

Ready to drop into `~/.claude/skills/` or sync via skillshare.

## Observation carry-forward

The 2 observed candidates (github_issue_management, repository_audit_cleanup) will be re-evaluated on the next `skill-miner scan`. If they appear in new sessions, they accumulate evidence and may get promoted to accept. If no new evidence after 28 days, they expire.

```
$ skill-miner status

Agents:
  [+] claude_code: 0 sessions
  [+] codex: 37 sessions
  [+] openclaw: 38 sessions
  [+] gemini: 0 sessions

Last run: 2026-03-19 08:08 UTC
Observing: 2
  - github_issue_management (week 0, 2 sessions)
  - repository_audit_cleanup (week 0, 2 sessions)

History:
  2026-03-19: 5A / 2O / 0R
```
