# Coding Practices Skills

24 reference skill files encoding production-grade software engineering
workflows: spec-first design, TDD, incremental implementation, debugging,
code review, git workflow, and shipping checklists.

**Source:** vendored verbatim from
[addyosmani/agent-skills](https://github.com/addyosmani/agent-skills)
(MIT licensed — see `ATTRIBUTION.md`). Not written by us; pulled in as
reference material for this bot's own coding-workspace work (cloning repos,
editing multi-file projects, testing, and pushing changes).

These are plain Markdown, framework-agnostic instructions — no CLI, API, or
docker dependency.

They are read through the `find_skill` tool, which searches this directory by
term overlap against each file's frontmatter `description` (written as "use
when X", which is the question being asked) and returns the matching playbook
in full. That is the "future tool that can read this repo directly" this note
used to anticipate: `read_file` basenames into `~/agent_workspace` and
`repo_read_file` resolves under the clone directory, so neither could ever
open this checkout.

Retrieval rather than prompt text on purpose. These 24 files are ~289 KB, and
the system prompt is re-sent on every AI call in a loop that fires up to
MAX_ITERATIONS times — inlining even the index would cost that on every step
of every turn. `agent.py`'s prompt now carries a single pointer to
`find_skill` instead, so the cost is one tool call, only when a task warrants
one. A query no playbook covers returns nothing rather than the least-bad
match: a returned playbook reads as instruction, so an irrelevant one is worse
than none.

## Index

| Skill | What It Does |
|-------|--------------|
| [using-agent-skills](using-agent-skills/SKILL.md) | Maps incoming work to the right skill workflow |
| [interview-me](interview-me/SKILL.md) | One-question-at-a-time interview to extract real requirements |
| [idea-refine](idea-refine/SKILL.md) | Structured divergent/convergent thinking for vague ideas |
| [spec-driven-development](spec-driven-development/SKILL.md) | Write a PRD before any code |
| [planning-and-task-breakdown](planning-and-task-breakdown/SKILL.md) | Decompose specs into small, verifiable tasks |
| [incremental-implementation](incremental-implementation/SKILL.md) | Thin vertical slices — implement, test, verify, commit |
| [test-driven-development](test-driven-development/SKILL.md) | Red-Green-Refactor, test pyramid, DAMP over DRY |
| [context-engineering](context-engineering/SKILL.md) | Feed agents the right information at the right time |
| [source-driven-development](source-driven-development/SKILL.md) | Ground framework decisions in official docs, cite sources |
| [doubt-driven-development](doubt-driven-development/SKILL.md) | Adversarial review of in-flight decisions on high-stakes work |
| [frontend-ui-engineering](frontend-ui-engineering/SKILL.md) | Component architecture, design systems, accessibility |
| [api-and-interface-design](api-and-interface-design/SKILL.md) | Contract-first design, error semantics, boundary validation |
| [browser-testing-with-devtools](browser-testing-with-devtools/SKILL.md) | Live runtime data via Chrome DevTools MCP |
| [debugging-and-error-recovery](debugging-and-error-recovery/SKILL.md) | Reproduce, localize, reduce, fix, guard |
| [code-review-and-quality](code-review-and-quality/SKILL.md) | Five-axis review, change sizing, severity labels |
| [code-simplification](code-simplification/SKILL.md) | Chesterton's Fence, Rule of 500, reduce complexity |
| [security-and-hardening](security-and-hardening/SKILL.md) | OWASP Top 10, auth patterns, secrets management |
| [performance-optimization](performance-optimization/SKILL.md) | Measure-first profiling and optimization |
| [git-workflow-and-versioning](git-workflow-and-versioning/SKILL.md) | Trunk-based dev, atomic commits, ~100-line change sizing |
| [ci-cd-and-automation](ci-cd-and-automation/SKILL.md) | Shift Left, feature flags, quality gate pipelines |
| [deprecation-and-migration](deprecation-and-migration/SKILL.md) | Code-as-liability mindset, migration patterns |
| [documentation-and-adrs](documentation-and-adrs/SKILL.md) | Architecture Decision Records, documenting the "why" |
| [observability-and-instrumentation](observability-and-instrumentation/SKILL.md) | Structured logging, RED metrics, tracing |
| [shipping-and-launch](shipping-and-launch/SKILL.md) | Pre-launch checklists, staged rollouts, rollback plans |
