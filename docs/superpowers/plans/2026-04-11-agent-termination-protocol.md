# Agent Termination Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the new DB tuning agent termination protocol across rollout, tool env, eval input handling, and debug logging.

**Architecture:** Extend the shared tool environment with explicit termination bookkeeping and a `finish_tuning` tool, then update both single-sample and batch rollout paths to honor natural stop, predict budget, repeated-call guardrails, and structured termination reasons. Align eval input formatting with training by passing chat messages directly instead of flattening them.

**Tech Stack:** Python, Hydra, Ray, verl, unittest

---

### Task 1: Tool Environment Termination Semantics

**Files:**
- Modify: `core/tool/tool_env.py`
- Modify: `environment/tools/db_tools.py`
- Modify: `environment/tools/__init__.py`
- Test: `tests/test_db_tool_env.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_finish_tuning_marks_episode_done():
    ...

def test_invalid_tool_call_streak_ends_episode_after_threshold():
    ...

def test_repeated_same_tool_same_args_ends_episode():
    ...

def test_predict_budget_exhausted_after_third_predict():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_db_tool_env -v`
Expected: FAIL on missing `finish_tuning` support and missing termination bookkeeping.

- [ ] **Step 3: Write minimal implementation**

```python
class FinishTuningTool(Tool):
    ...

class ToolEnv:
    def reset(self):
        self.termination_reason = None
        self.invalid_tool_call_streak = 0
        self.predict_calls_used = 0
        self.same_tool_same_args_streak = 0
        self.last_tool_fingerprint = None
        self.last_valid_tool_call = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_db_tool_env -v`
Expected: PASS

### Task 2: Single-Step Rollout Protocol

**Files:**
- Modify: `core/agent.py`
- Test: `tests/test_core_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_rollout_stops_on_missing_tool_call_with_no_tool_call_reason():
    ...

def test_rollout_preserves_last_valid_state_on_invalid_tool_call_end():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_core_agent -v`
Expected: FAIL because missing-tool-call still becomes invalid-format retry.

- [ ] **Step 3: Write minimal implementation**

```python
if not has_tool_call(response):
    trajectory.append(...)
    info["termination_reason"] = "no_tool_call"
    break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_core_agent -v`
Expected: PASS

### Task 3: Batch Rollout / VERL Integration

**Files:**
- Modify: `training/llm_agent/generation.py`
- Modify: `training/verl/agent_ray_trainer.py`
- Modify: `training/verl/main_grpo.py`
- Test: `tests/test_async_rollout_integration.py`
- Test: `tests/test_agent_ray_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_batch_rollout_records_termination_reason():
    ...

def test_rollout_debug_log_contains_termination_metadata():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_async_rollout_integration tests.test_agent_ray_trainer -v`
Expected: FAIL because termination metadata is not propagated.

- [ ] **Step 3: Write minimal implementation**

```python
meta_info["termination_reason"] = ...
non_tensor_batch["termination_reason"] = ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_async_rollout_integration tests.test_agent_ray_trainer -v`
Expected: PASS

### Task 4: Eval Message Handling

**Files:**
- Modify: `core/llm/multi_client.py`
- Modify: `data_pipeline/synthesis/trajectory/sampler.py`
- Test: `tests/test_multi_client.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_accepts_chat_messages_without_flattening():
    ...

def test_sampler_build_llm_fn_passes_message_list_to_client():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_multi_client -v`
Expected: FAIL because `generate()` only accepts string prompts.

- [ ] **Step 3: Write minimal implementation**

```python
def generate(self, prompt_or_messages, ...):
    if isinstance(prompt_or_messages, list):
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": prompt_or_messages}]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_multi_client -v`
Expected: PASS

### Task 5: Data Synthesis Contract Cleanup

**Files:**
- Modify: `data_pipeline/synthesis/trajectory/sampler.py`
- Test: `tests/test_sampler_contract.py`

- [ ] **Step 1: Write the failing test**

```python
def test_messages_to_sft_does_not_append_terminal_assistant():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_sampler_contract -v`
Expected: FAIL because the synthetic closing assistant is still appended.

- [ ] **Step 3: Write minimal implementation**

```python
def _messages_to_sft(...):
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_sampler_contract -v`
Expected: PASS

### Task 6: End-to-End Verification

**Files:**
- Modify: `docs/agent/1.termination_protocol_design.md` (only if implementation forces wording updates)

- [ ] **Step 1: Run targeted verification suite**

Run: `python -m unittest tests.test_db_tool_env tests.test_core_agent tests.test_async_rollout_integration tests.test_agent_ray_trainer tests.test_multi_client tests.test_sampler_contract -v`
Expected: PASS

- [ ] **Step 2: Run existing regression suite most likely to be affected**

Run: `python -m unittest tests.test_main_grpo tests.test_training_scripts -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add core/tool/tool_env.py environment/tools/db_tools.py environment/tools/__init__.py core/agent.py training/llm_agent/generation.py training/verl/agent_ray_trainer.py training/verl/main_grpo.py core/llm/multi_client.py data_pipeline/synthesis/trajectory/sampler.py tests/test_db_tool_env.py tests/test_core_agent.py tests/test_async_rollout_integration.py tests/test_agent_ray_trainer.py tests/test_multi_client.py tests/test_sampler_contract.py docs/agent/1.termination_protocol_design.md
git commit -m "Implement agent termination protocol"
```
