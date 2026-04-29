# v3 Step-Level SFT Data

v3 converts the v2 full DB-agent trajectories into step-level LLaMA-Factory
SFT samples. Each assistant tool-call step becomes one row, and the previous
steps are stored in `history` so training can mask historical turns.

All variants below are built from the first 3,000 v2 trajectories:

| Directory | Purpose |
| --- | --- |
| `full_v3_a_step_raw_history_3k` | Keeps `<think>...</think>` in historical assistant turns. |
| `full_v3_b_step_no_think_history_3k` | Removes historical `<think>...</think>` blocks; current output is unchanged. |
| `full_v3_c_step_no_think_finish2x_3k` | Same as B, with `finish_tuning` steps oversampled 2x. |
| `full_v3_d_step_no_think_finish4x_3k` | Same as B, with `finish_tuning` steps oversampled 4x. |

Files are JSONL shards named `train-xxxxx-of-yyyyy.jsonl` to keep each file
under GitHub's 100MB single-file limit. `stats.json` records row counts,
skipped counts, source path, and shard paths.
