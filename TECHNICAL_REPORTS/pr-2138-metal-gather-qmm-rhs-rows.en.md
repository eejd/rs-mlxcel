# PR #2138: Size Metal `gather_qmm_rhs` by the broadcast x rows

**Date**: 2026-10-13
**Status**: The kernel fix passed focused Metal and model-smoke validation on M1 Ultra at exact head `25202198`; the current candidate passes Linux build, focused tests, contracts, and independent review and awaits a fresh Metal nightly
**Risk**: Low (no-op for every production caller; changes behavior only for an x that `gather_qmm_rhs` has to broadcast)

## Summary

The Metal nightly on the M1 Ultra runner has failed since 2026-09-30 on `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`, both in the `SortedSharedActivation` case. `GatherQMM::eval_gpu` in MLX's Metal backend passes `x.size() / K` to `gather_qmm_rhs` as the row count. That is the row count of x before `gather_qmm_rhs` broadcasts x against the indices. For one activation row gathered against 32 sorted slots it is 1, so the grid and the kernel's `M` bound cover one row and output rows 1..31 are never written. The overlay now passes `B * M`. The PR also makes two split-MTP tests portable across macOS path canonicalization and filesystems that reject invalid-byte names; the production resolver and its symlink protections are unchanged.

Refs #1599.

## 1. Nightly history

| Dates | Failure | Status |
|---|---|---|
| 2026-09-28 | `verify-clippy`: unused `Result` in `fused_norm_parity_tests.rs:158` and `fused_rope_parity_tests.rs:133` | fixed by #2029 (d8d34e2b) |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER` missing `Speech` | fixed by #2080 (5486e404) |
| 2026-09-30 to 10-11 | the two mxfp gather tests | fixed by this PR; focused Metal validation passed at `25202198` |
| 2026-10-12 (`1e8c7022`) | the same mxfp tests; split-MTP `/var` canonicalization and invalid-byte filename assumptions; three YAML `err_expect` lints | kernel and split-MTP tests fixed by this PR; YAML lint already fixed on current main |

The mxfp tests arrived with #2071 (2cee9cf4) on 2026-09-30, so the last signature is a new test exposing a latent upstream edge case, not a regression.

## 2. Root cause

- The case uses x `[1, 1, K]`, indices `[32]` over 8 experts, `sorted = true`. With M == 1, B = 32, B / E = 4 and `right_sorted_`, `eval_gpu` takes `gather_qmm_rhs`.
- `gather_qmm_rhs` broadcasts x to 32 rows (`broadcast_with_indices`), but its `M` argument was computed by the caller from the original x: 1. `grid_dims.y = ceil(M / 16)` and the kernel's `tgp_bm = min(BM, M - y_row)` then write row 0 only. `out` comes from `allocator::malloc`, so the other rows hold whatever was there: 7.4e32 relative error (mxfp4) and non-finite values (mxfp8).
- The same runs pass the three earlier gather cases in bf16, f16 and f32, and `mxfp_matmuls_match_host_reference_on_cpu_device`, which runs `SortedSharedActivation` on MLX's CPU backend. The reference and the test are correct; "bf16" in the message is just the first dtype tried.
- The defect is mode-independent (affine would hit it too) and is still present on ml-explore/mlx main.

## 3. Change

`B * M` (the output row count) replaces `x.size() / K` at the one call site. It equals the old value whenever x is already expanded, which is every production caller (`SwitchLinear::forward(.., true)` always receives `gather_sort` output), and it also fixes the NAX variant, which receives the same argument. The overlay header and the CMake overlay notes list the new hunk.

The split-MTP test for `.` and `..` outputs now compares the staging parent with the canonical temporary-directory path, matching the resolver's intentional behavior on macOS where `/var` resolves to `/private/var`. The non-UTF-8 test first probes the temporary filesystem: filesystems that accept the byte name retain the successful round trip, while rejecting filesystems must return the same I/O error and leave no output or staging entry.

## 4. Validation

- Historical exact-head Metal gate: Apple M1 Ultra (128 GB), macOS 27.0 (26A428), full Xcode 27 RC, PR head `25202198` on base `b34d517b`. All five `models::switch_layers::mxfp_tests` passed with the Metal overlay rebuilt, and the release build passed.
- Historical exact-head model smoke: Qwen3-30B-A3B-4bit, greedy 64 tokens, produced byte-identical text on base, repeated base, and PR; SHA-256 `d2f40b2d5831c3d7790d467fb51123f0d1f615fb7300aa2f92d86a9d8831a0de`.
- CUDA (GB10): the original PR branch built and all 32 `models::switch_layers` tests passed. This does not validate Metal.
- Current Linux candidate: the exact `mlxcel` plus `mlxcel-surgery` CUDA test graph compiled successfully in 25m54s. Both focused portability tests pass (2/2), repository contracts pass (123 Python tests and 49 WebUI fixtures), and formatting and diff checks pass.
- Independent review found no source, security, or documentation issue in the three-source-file change, canonical source SHA-256 `2e36ccb93dfe3fa49206f5741249ebb45c49d3e9e19195f94f6d1282e55155a5`.
- The current branch retains the historical M1 evidence while awaiting a fresh candidate Metal nightly. No current-head model smoke is claimed.

## 5. Follow-up

Report the call to ml-explore/mlx, and drop the hunk when a pin bump carries an upstream fix. Keep #1599 open through merge and close it only after a green post-merge Metal nightly.
