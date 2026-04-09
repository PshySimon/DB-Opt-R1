# verl GRPO Async Worker Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `training.verl.main_grpo` with the async worker interfaces expected by the installed `verl 0.7.1` package.

**Architecture:** Extract worker selection into a small helper in `training/verl/main_grpo.py`, then switch the FSDP path to `AsyncActorRolloutRefWorker` so the actor rollout worker group exposes `update_weights`. Cover the selection logic with a focused unit test and verify the targeted test suite.

**Tech Stack:** Python, unittest, OmegaConf, verl 0.7.1 worker APIs

---

### Task 1: Align async worker selection

**Files:**
- Modify: `training/verl/main_grpo.py`
- Test: `tests/test_main_grpo.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_worker_components_uses_async_worker_for_fsdp():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_main_grpo -v`
Expected: FAIL because `main_grpo` still selects `ActorRolloutRefWorker`.

- [ ] **Step 3: Write minimal implementation**

```python
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker
role_worker_mapping = {
    Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
    Role.RefPolicy: ray.remote(AsyncActorRolloutRefWorker),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_main_grpo -v`
Expected: PASS

- [ ] **Step 5: Run adjacent regression coverage**

Run: `python -m unittest tests.test_agent_ray_trainer -v`
Expected: PASS
