# Unify ToolEnv Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training and inference share the same ToolEnv/Tool single-step semantics while preserving training-only batch scheduling.

**Architecture:** Promote `core.tool.tool_env.ToolEnv` and `core.tool.tool_base.Tool` to the single source of truth. Keep `training/tool/tool_env.py` only as a thin compatibility layer that re-exports the core classes and provides `step_batch` built on the same semantics.

**Tech Stack:** Python, unittest, existing core/training tool environment modules.

---

### Task 1: Lock semantic expectations with tests

**Files:**
- Create: `tests/test_training_tool_env.py`

- [ ] **Step 1: Write failing tests for ToolEnv parity and batch behavior**
- [ ] **Step 2: Run tests to verify failure against current divergent implementation**
- [ ] **Step 3: Cover `extract_tool_call`, `step`, and `step_batch == repeated core.step` equivalence**

### Task 2: Convert training tool base/env into compatibility wrappers

**Files:**
- Modify: `training/tool/tool_base.py`
- Modify: `training/tool/tool_env.py`

- [ ] **Step 1: Re-export `Tool` from `core.tool.tool_base`**
- [ ] **Step 2: Re-export `ToolEnv` and `step` from `core.tool.tool_env`**
- [ ] **Step 3: Reimplement `step_batch` to use the shared core semantics instead of private copies**
- [ ] **Step 4: Keep only training-only helpers that do not change single-step semantics**

### Task 3: Verify affected training code still imports correctly

**Files:**
- Test: `tests/test_agent_ray_trainer.py`
- Test: `tests/test_db_tool_env.py`
- Test: `tests/test_multi_client.py`
- Test: `tests/test_training_scripts.py`

- [ ] **Step 1: Run targeted unit tests**
- [ ] **Step 2: Run py_compile for touched modules**
- [ ] **Step 3: Commit and push only the ToolEnv unification changes**
