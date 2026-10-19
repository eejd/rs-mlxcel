# Local work on top of upstream

This fork (`eejd/rs-mlxcel`) mirrors `lablup/mlxcel` on `main` (fast-forward only, never edited).
`local` is `main` plus the `topic/*` branches below, merged with `--no-ff`. Port `mlxcel-local` in
`eejd/macports-ports-local` builds the tags `v<base>-local<NN>` on `local`, `<base>` being the last
upstream release `main` contains (`v0.7.0`; upstream has cut no release since, though `main` is
364 commits past it). The first tag is therefore `v0.7.0-local01` (docs/development-model.md,
ADR-0008 §2: the dated `<last-release>.<YYYYMMDD>` form is the `-devel` version scheme only).

## Re-cut of 2026-10-19 (owner decision, option B of eejd/rs-mlxcel#1)

| | |
|---|---|
| Base | `main` @ `99967257` (`v0.7.0-364-g99967257`, MLX pin `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`) |
| Old `local` | `c7832e5e` = `v0.7.0` (`ec9db8bf`, MLX pin `9a795735`) + this file; archived as tag `archive/local-pre-recut-c7832e5` |
| New `local` | initial tip = the commit that adds this file on top of `99967257`; its SHA is in the re-cut PR (it cannot name itself) |

The old `local` carried five topics on a `v0.7.0` base. Four of them are on upstream `main`, so
they are not carried on the new `local`; their old heads are archived as tags and their draft PRs
(eejd/rs-mlxcel#10-#13) are closed as superseded.

| Old topic head (archive tag) | Upstream `main` commit that contains the fix |
|---|---|
| `topic/paged-decode-pad-rows` `03c24b30` (`archive/topic-paged-decode-pad-rows-03c24b3`) | `e6c9cda9` (lablup/mlxcel#2097, closes #2096) |
| `topic/eos-precedence` `c3d5c649` (`archive/topic-eos-precedence-c3d5c64`) | `c480ed67` (lablup/mlxcel#2439, closes #2384) |
| `topic/cache-hit-full-prompt` `ef5c2c0a` (`archive/topic-cache-hit-full-prompt-ef5c2c0`; earlier tip `0c516210`, replaced by a force-push before tagging existed: `archive/topic-cache-hit-full-prompt-old-0c51621`) | `7c3bece5` (lablup/mlxcel#2136, closes #1760; adds a multimodal gate the backport lacked) |
| `topic/chat-no-double-bos` `5ea5aeac` (`archive/topic-chat-no-double-bos-5ea5aea`) | `5a5e73b4` (lablup/mlxcel#1739, the BOS change is a side change of the Laguna model PR: `prompt_carries_bos`) |
| `topic/no-batch-500` `b9f2ed8e` (`archive/topic-no-batch-500-b9f2ed8`) | not upstream: rebased onto `main`, see below |

Also archived: `archive/local-integ-evidence-728d339`, the integration commit (`v0.7.0` base + all
five topics) that rs-mlxcel#1 built and measured.

## Topics carried

| Topic branch | What | Origin | Upstream status |
|---|---|---|---|
| `topic/no-batch-500` | the pool capacity `max_batch_size + max_queue_depth` saturates; `--no-batch` (queue depth `usize::MAX`) no longer wraps it to 0 and answers HTTP 500 `CachePool: max capacity (0) reached`, eejd/rs-mlxcel#9; the regression test builds a scheduler with an unbounded queue depth | local, own change (re-cut on `main`; the v0.7.0 version was `b9f2ed8e`) | not submitted; `main` has the same `+` and the same `usize::MAX` in the legacy worker |

Nothing here is posted to `lablup/mlxcel` before the quantbench + agentbench soak (ADR-0008).
