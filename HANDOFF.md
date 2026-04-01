# DL-Projects Handoff

This file is the model-neutral handoff point for this repository.

Use it when switching between Codex, Claude Code, or any other coding agent.
The goal is simple: keep progress continuous even if the tool changes.

---

## 1. Source of Truth

Read files in this order before continuing work:

1. `HANDOFF.md`
2. `AGENTS.md` when working in Codex
3. `CLAUDE.md` when working in Claude Code
4. `README.md`
5. The relevant project README
6. The relevant `docs/design.md` and `docs/plan.md`
7. `git status --short`

### Important note

`AGENTS.md` and `CLAUDE.md` currently describe nearly the same workflow and teaching style.
They are the instruction layer.

`HANDOFF.md` is the continuity layer.
If a future session changes shared workflow, update both model-specific files and also record the change here.

---

## 2. How To Continue Work

Do not rely on hidden model-specific tools, skills, or memory.
Instead, continue from artifacts inside the repo:

- design docs
- plan docs
- README narrative
- existing code patterns
- saved results
- notebook progress

If a plan document mentions Claude-only skills like `superpowers:subagent-driven-development`,
treat that as historical implementation context, not as a required dependency.
The actionable part is the checklist, file map, and design decisions in the document.

---

## 3. Repo Narrative

The portfolio story is:

`MNIST scratch CNN -> CIFAR-10 scratch ceiling -> Transfer Learning breakthrough -> RNN for sequences -> LSTM for long-term memory`

The learner-facing teaching pattern is:

- start from the limitation of the previous project
- explain the new concept before coding when it is genuinely new
- go one cell or one unit at a time
- connect implementation back to DLFS/HOML intuition
- avoid dumping large blocks of code without explanation
- act as a pair teacher who improves learner understanding, not just a code generator

---

## 4. Stable Conventions

These conventions are already established and should stay consistent across agents.

### Code structure

- Use `__file__`-based paths, never depend on CWD.
- Keep project-local `scripts/`, `results/`, `notebooks/`, `data/`.
- Put reusable training helpers in `utils.py`.
- Keep train entrypoints in `train.py` or versioned variants like `train_finetune.py`.

### Utility patterns

Expected recurring helpers:

- `get_device()`
- `get_dataloaders(...)`
- `train_one_epoch(...)`
- `evaluate(...)`
- plotting helpers appropriate to the task

### Teaching style

- Add tensor shape comments where they materially help.
- Explain why a new technique is needed compared with the previous project.
- When the learner is stuck, give the fix directly and explain the mistake clearly.

### Documentation

- When code or project behavior changes, update the relevant README in the same workstream.
- Preserve the portfolio narrative, not just the raw metrics.
- Treat `summary.html` as a secondary visual study guide. The main public explanation lives in `README.md`.
- If `summary.html` is kept in git, position it as an optional supplementary asset, not the primary artifact.

---

## 5. Current Project Status

### Completed

- `1_MNIST_CNN`
- `2_CIFAR10_CNN`
- `3_Transfer_Learning`
- `4_RNN_Shakespeare`

`4_RNN_Shakespeare` has:

- README
- design doc
- plan doc
- scripts
- notebook
- trained checkpoint
- generated outputs and plots

### In Progress

- `5_LSTM_Sentiment`

Current state of `5_LSTM_Sentiment`:

- `docs/design.md` exists
- `docs/plan.md` exists
- `scripts/utils.py` exists but is only partially implemented
- `scripts/model.py` does not exist yet
- `scripts/train.py` does not exist yet
- `README.md` does not exist yet
- `results/` is still empty
- `notebooks/` is still empty

The next implementation target is clearly `5_LSTM_Sentiment`.

---

## 6. Immediate Next Task

Unless the user redirects, continue by finishing `5_LSTM_Sentiment` in this order:

1. Complete `scripts/utils.py`
2. Add `scripts/model.py`
3. Add `scripts/train.py`
4. Add `5_LSTM_Sentiment/README.md`
5. Update root `README.md`
6. Run a smoke test if environment permits

### Known missing pieces in `5_LSTM_Sentiment/scripts/utils.py`

- `train_one_epoch(...)`
- `evaluate(...)`
- `load_glove(...)`
- `plot_history(...)`
- `plot_confusion_matrix(...)`
- `plot_wrong_predictions(...)`

The earlier functions in that file are already partially or fully drafted and should be preserved where correct.

---

## 7. Cross-Agent Rules

When switching between Codex and Claude:

- keep `AGENTS.md` and `CLAUDE.md` semantically aligned
- keep `HANDOFF.md` model-neutral
- write down important progress in files, not only in chat
- do not assume either agent can access the other's private tool memory
- prefer documented workflows over tool-specific magic

If one agent introduces a new shared convention, do this:

1. update `AGENTS.md`
2. update `CLAUDE.md`
3. note the change in `HANDOFF.md`

That keeps future switching cheap and safe.

---

## 8. Session Start Checklist

At the beginning of a new session:

1. Read `HANDOFF.md`
2. Read the model-specific instruction file for the current tool
3. Check `git status --short`
4. Open the active project's README and docs
5. Continue from the next unfinished checklist item, not from memory

---

## 9. Session End Checklist

Before ending a substantial work session:

1. Update the relevant README if behavior or scope changed
2. Update `HANDOFF.md` if the active project status changed
3. Leave the next step obvious from the repo contents
4. Avoid leaving important decisions only in chat history

---

## 10. AGENTS.md Usage

For Codex specifically, `AGENTS.md` is not something you manually "run".
It is an instruction file the agent reads as guidance for how to work in this repo.

In practice, it means:

- it sets the teaching style
- it sets code organization expectations
- it defines README expectations
- it tells the agent what the learner already knows
- it influences how the agent explains and implements changes

So your job is mostly:

- keep `AGENTS.md` accurate
- tell the agent to inspect the repo
- point it to the active project if needed

You do not need to paste its contents every time if the file is already in the repo and the agent can see it.

---

## 11. One-Line Operating Principle

Shared repo documents should be strong enough that either Claude or Codex can resume work correctly with no hidden context.
