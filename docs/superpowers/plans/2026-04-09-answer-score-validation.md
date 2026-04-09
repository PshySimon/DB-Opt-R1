# Answer Score Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic validation path proving `answer_score` becomes positive for a realistic `set_knob` trajectory.

**Architecture:** Reuse the existing reward scoring implementation and add one small verification script that drives it with a fake cost model. Lock the behavior with unit tests so future reward-schema changes do not silently break `answer_score` again.

**Tech Stack:** Python, unittest, existing reward scoring module

---

### Task 1: Extend reward-score tests

**Files:**
- Modify: `tests/test_reward_score.py`
- Test: `tests/test_reward_score.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_score_answer_returns_positive_for_valid_set_knob_payload(self):
    class FakeCostModel:
        def predict(self, knobs, hardware):
            return 120.0 if knobs.get("shared_buffers") == "8GB" else 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_reward_score -v`
Expected: FAIL because the reward path does not yet expose a deterministic positive validation case.

- [ ] **Step 3: Write minimal implementation**

```python
class FakeCostModel:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_reward_score -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_reward_score.py
git commit -m "Add answer score reward validation tests"
```

### Task 2: Add executable verification entrypoint

**Files:**
- Create: `scripts/verify_answer_score.py`
- Modify: `tests/test_reward_score.py`
- Test: `tests/test_reward_score.py`

- [ ] **Step 1: Write the failing test**

```python
def test_verify_answer_score_script_reports_positive_score(self):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_reward_score -v`
Expected: FAIL because the verification script does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
solution = ...
ground_truth = ...
score = compute_score_answer(...)
print(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_reward_score tests.test_main_grpo tests.test_agent_ray_trainer tests.test_async_rollout_integration tests.test_training_scripts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_answer_score.py tests/test_reward_score.py
git commit -m "Add answer score verification script"
```
