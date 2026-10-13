# PR #2138: Metal `gather_qmm_rhs`의 행 수를 broadcast된 x 기준으로 지정

**작성일**: 2026-10-14
**상태**: 커널 수정은 정확한 head `25202198`에서 M1 Ultra 집중 Metal 검증과 모델 스모크를 통과. 갱신한 #1599 변경은 Linux 빌드, 집중 테스트, 계약 검사, 독립 리뷰와 갱신한 후보 Metal nightly를 통과
**위험도**: 낮음 (모든 프로덕션 호출에서 no-op이며, `gather_qmm_rhs`가 x를 broadcast해야 하는 경우에만 동작이 바뀜)

## 요약

M1 Ultra 러너의 Metal nightly는 2026-09-30부터 `models::switch_layers::mxfp_tests::{mxfp4,mxfp8}_gather_qmm_matches_host_reference`의 `SortedSharedActivation` 케이스에서 실패했다. MLX Metal 백엔드의 `GatherQMM::eval_gpu`는 `gather_qmm_rhs`에 행 수로 `x.size() / K`를 넘긴다. 이 값은 `gather_qmm_rhs`가 x를 indices에 맞춰 broadcast하기 전의 행 수다. 활성 행 하나를 정렬된 슬롯 32개에 gather하면 이 값은 1이 되어, grid와 커널의 `M` 경계가 한 행만 덮고 출력의 1..31행은 기록되지 않는다. 이 PR은 overlay에서 `B * M`을 넘긴다. 또한 macOS 경로 정규화와 유효하지 않은 바이트 이름을 거부하는 파일시스템에서 split-MTP 테스트 두 개가 이식성 있게 동작하도록 고친다. 프로덕션 resolver와 심볼릭 링크 보호는 변경하지 않는다.

Refs #1599.

## 1. Nightly 이력

| 날짜 | 실패 | 상태 |
|---|---|---|
| 2026-09-28 | `verify-clippy`: `fused_norm_parity_tests.rs:158`, `fused_rope_parity_tests.rs:133`의 미사용 `Result` | #2029 (d8d34e2b)에서 수정 |
| 2026-09-29 | `tests::family_order_is_exhaustive`: `FAMILY_ORDER`에 `Speech` 누락 | #2080 (5486e404)에서 수정 |
| 2026-09-30 ~ 10-11 | mxfp gather 테스트 두 개 | 이 PR에서 수정; `25202198`에서 집중 Metal 검증 통과 |
| 2026-10-12 (`1e8c7022`) | 같은 mxfp 테스트, split-MTP의 `/var` 정규화 및 유효하지 않은 바이트 파일명 가정, YAML `err_expect` lint 세 개 | 커널과 split-MTP 테스트는 이 PR에서 수정; YAML lint는 현재 main에서 이미 수정됨 |

mxfp 테스트는 2026-09-30 #2071 (2cee9cf4)과 함께 추가되었으므로, 마지막 실패는 회귀가 아니라 새 테스트가 upstream의 잠재된 경계 사례를 드러낸 것이다.

## 2. 근본 원인

- 케이스 입력은 x `[1, 1, K]`, expert 8개에 대한 indices `[32]`, `sorted = true`다. M == 1, B = 32, B / E = 4, `right_sorted_`이므로 `eval_gpu`는 `gather_qmm_rhs`로 간다.
- `gather_qmm_rhs`는 x를 32행으로 broadcast하지만(`broadcast_with_indices`), `M` 인자는 호출자가 원래 x로 계산한 1이다. `grid_dims.y = ceil(M / 16)`과 커널의 `tgp_bm = min(BM, M - y_row)` 때문에 0행만 기록된다. `out`은 `allocator::malloc`에서 오므로 나머지 행에는 이전 메모리 값이 남는다: mxfp4는 상대 오차 7.4e32, mxfp8은 non-finite.
- 같은 실행에서 앞선 gather 케이스 세 개는 bf16, f16, f32 모두 통과했고, MLX CPU 백엔드에서 `SortedSharedActivation`을 돌리는 `mxfp_matmuls_match_host_reference_on_cpu_device`도 통과했다. reference와 테스트는 올바르며, 메시지의 "bf16"은 처음 시도한 dtype일 뿐이다.
- 결함은 모드와 무관하고(affine도 동일), ml-explore/mlx main에도 그대로 있다.

## 3. 변경

유일한 호출 지점에서 `x.size() / K`를 `B * M`(출력 행 수)으로 바꿨다. x가 이미 확장된 경우 기존 값과 같으며, 모든 프로덕션 호출(`SwitchLinear::forward(.., true)`는 항상 `gather_sort` 출력을 받음)이 이에 해당한다. 같은 인자를 받는 NAX 변형도 함께 고쳐진다. overlay 헤더와 CMake overlay 주석에 새 hunk를 기록했다.

`.` 및 `..` 출력의 split-MTP 테스트는 staging 부모를 정규화된 임시 디렉터리 경로와 비교한다. 이는 macOS에서 `/var`가 `/private/var`로 해석되는 production resolver의 의도와 일치한다. 비 UTF-8 테스트는 임시 파일시스템의 바이트 이름 지원을 먼저 검사한다. 지원하는 파일시스템에서는 성공 round trip을 그대로 검증하고, 거부하는 파일시스템에서는 같은 I/O 오류가 반환되며 출력이나 staging 항목이 남지 않아야 한다.

## 4. 검증

- 이전 정확한 head의 Metal gate: Apple M1 Ultra (128 GB), macOS 27.0 (26A428), full Xcode 27 RC, base `b34d517b` 위 PR head `25202198`. Metal overlay를 다시 빌드한 뒤 `models::switch_layers::mxfp_tests` 다섯 개가 모두 통과했고 release 빌드도 통과했다.
- 이전 정확한 head의 모델 스모크: Qwen3-30B-A3B-4bit, greedy 64 tokens에서 base, base 반복, PR이 byte-identical text를 생성했다. SHA-256은 `d2f40b2d5831c3d7790d467fb51123f0d1f615fb7300aa2f92d86a9d8831a0de`다.
- CUDA (GB10): 기존 PR 브랜치가 빌드됐고 `models::switch_layers` 테스트 32개가 모두 통과했다. 이는 Metal 검증이 아니다.
- 갱신한 #1599 Linux 후보: 정확한 `mlxcel` 및 `mlxcel-surgery` CUDA 테스트 그래프가 25분 54초에 성공적으로 컴파일됐다. 이식성 집중 테스트 두 개가 모두 통과했고, 저장소 계약 검사도 Python 테스트 123개와 WebUI fixture 49개 모두 통과했다. formatting과 diff 검사도 통과했다.
- 세 소스 파일 변경에 대한 독립 소스·보안·문서 리뷰에서 발견 사항이 없었다. 정규 source SHA-256은 `2e36ccb93dfe3fa49206f5741249ebb45c49d3e9e19195f94f6d1282e55155a5`다.
- 별도로 검증된 Kimi Linear chunked-prefill 수정이 담긴 main `398f9ba8`을 그 뒤 충돌 없이 병합했다. 이 비중첩 동기화 때문에 #1599 native graph를 다시 빌드하지 않았으며 PR CI와 최종 통합 gate가 이를 검증한다.
- 후보 Metal nightly [38418557399](https://github.com/lablup/mlxcel/actions/runs/38418557399)가 소스 head `2dbffe6d`와 `lablup-macmini01`에서 통과했다. formatting, Clippy, 테스트 12,283개(390개 ignored), 영상 테스트 51개가 통과했다. 현재 head의 모델 스모크를 주장하지 않으며 이전 정확한 head의 체크포인트 증거와 구분한다.

## 5. 후속 작업

ml-explore/mlx에 이 호출을 보고하고, upstream 수정이 포함된 pin bump 때 이 hunk를 제거한다. #1599는 merge 후에도 열어 두고 post-merge Metal nightly가 통과한 뒤에만 닫는다.
