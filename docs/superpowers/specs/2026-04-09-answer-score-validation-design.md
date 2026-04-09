# Answer Score Validation Design

**Goal:** Add a deterministic way to verify that `answer_score` is no longer stuck at zero after the `set_knob` reward parsing fix.

**Approach:** Keep the validation path small and local to reward scoring. Reuse the current `compute_score_answer()` logic, add a focused executable verification script, and back it with unit tests that exercise the real `set_knob` payload shape used by the DB tool environment.

**Scope:**
- Add one deterministic verification entrypoint for reward scoring.
- Reuse the existing reward code instead of adding a parallel implementation.
- Keep training code unchanged unless the validation reveals another bug.

**Files:**
- Modify: `training/reward_score.py`
- Create: `scripts/verify_answer_score.py`
- Modify: `tests/test_reward_score.py`

**Success Criteria:**
- A local command can print a positive `answer_score` for a realistic `set_knob` tool-call trajectory.
- Unit tests cover both knob extraction and positive answer-score computation.
- No existing GRPO/verl regression tests fail.
