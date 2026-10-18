# Local work on top of upstream

This fork (`eejd/rs-mlxcel`) mirrors `lablup/mlxcel` on `main` (never edited). `local` is the release
base `v0.7.0` (MLX pin `9a795735ad9a42664e08f42361b405ed570bcf1a`, the pin the `mlxcel` port builds)
plus the `topic/*` branches below, merged with `--no-ff`. Port `mlxcel-local` in
`eejd/macports-ports-local` builds the tags `v<base>-local<NN>` on `local`.

Why v0.7.0 and not `main` as the base: `main` is 362 commits past the tag, pins another MLX commit
(`243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`) and has no release. It contains the upstream fixes below
(its run against the matrices is recorded in eejd/rs-mlxcel#1), but the deployed engine is the v0.7.0
release and its numerics are the ones the diagnosis measured. The fixes are carried as attributed
backports.

| Topic branch | What | Origin | Upstream status |
|---|---|---|---|
| `topic/paged-decode-pad-rows` | `KVCache::trim` rewinds the block table of a pool-backed cache and moves `offset`, so the pad rows of a NAX tile-padded prefill leave the pool (default launch answered wrongly at every N mod 32 != 0), eejd/rs-mlxcel#2 | backport of lablup/mlxcel#2097 (`e6c9cda9`), no changes | on upstream `main` (closes lablup/mlxcel#2096), not in v0.7.0 |
| `topic/eos-precedence` | `read_eos_token_ids` returns the union of generation_config, tokenizer_config and config.json (`<\|eom_id\|>` 128008 now stops Llama-3.2 snapshots without generation_config.json), eejd/rs-mlxcel#3 | backport of the `read_eos_token_ids` hunks of lablup/mlxcel#2439 (`c480ed67`) | on upstream `main` (closes lablup/mlxcel#2384), not in v0.7.0 |
| `topic/cache-hit-full-prompt` | a whole-prompt prompt-cache hit adopts at most `len - 1` tokens (no duplicated last token, no `broadcast_shapes` abort), eejd/rs-mlxcel#4 | backport of lablup/mlxcel#2136 (`7c3bece5`); conflict in the snapshot branch resolved (no multimodal whole-entry gate in v0.7.0) | on upstream `main` (closes lablup/mlxcel#1760), not in v0.7.0 |
| `topic/chat-no-double-bos` | `MlxcelTokenizer::prompt_carries_bos`: no second `<\|begin_of_text\|>` on Llama-3 chat prompts, eejd/rs-mlxcel#5 | backport of the tokenizer/BOS hunks of `5a5e73b4` (closes lablup/mlxcel#1347); the Laguna model is not carried | on upstream `main`, not in v0.7.0 |
| `topic/no-batch-500` | the pool capacity `max_batch_size + max_queue_depth` saturates; `--no-batch` (queue depth `usize::MAX`) no longer wraps it to 0 and answers 500, eejd/rs-mlxcel#9 | local, own change | not submitted; upstream `main` has the same `+` and the same `usize::MAX` |

Nothing here is posted to `lablup/mlxcel` before the quantbench + agentbench soak (ADR-0008).
