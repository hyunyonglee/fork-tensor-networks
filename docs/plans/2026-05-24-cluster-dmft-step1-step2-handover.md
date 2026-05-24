# Cluster DMFT — step1 + step2 인수인계 (2026-05-24)

2-orbital × 2-site cluster AIM 의 step1 (ground state) + step2 (matrix Green's function)
까지가 현재 동작합니다. 본 문서는 **현재 코드를 그대로 돌리는 방법, 입출력 파일, config
키의 의미** 만 정리합니다. step3 (spectral function + bath update) 는 직접 구현하세요.

---

## 1. 디렉터리 구조

```
ForkTensorNetworks/
├── src/                          # FTN 엔진 (DMRG, TDVP, IO, …)
├── models/
│   ├── ClusterAndersonImpurityModel.jl    # FTNO/FTNS 빌더 (2-orb 2-site)
│   └── TwoSiteFermion.jl                  # "2SiteFermion" sitetype
└── dmft/
    ├── ClusterDMFTLoop.jl                 # 오케스트레이터 (현재는 step1+2 만)
    ├── Bath.jl                            # 초기 bath 생성, 이산화 함수들
    ├── GreensFunction.jl                  # ESPRIT, ComplexPoleGF (step3 에서 사용)
    ├── ESPRIT.jl
    ├── cluster_test_config.toml           # 테스트용 config
    └── scripts/
        ├── cluster_step1_ground_state.jl
        ├── cluster_step2_greens_function.jl
        └── cluster_step3_spectral.jl      # ← 다시 작성/대체 대상
```

---

## 2. 실행 방법

작업 디렉터리는 `dmft/`. Julia project 는 repo root.

### 2.1 자동 (오케스트레이터)

```bash
cd dmft
julia --project=.. ClusterDMFTLoop.jl cluster_test_config.toml
```

`ClusterDMFTLoop.jl` 가 하는 일:

1. config TOML 을 `work_dir/dmft_config.toml` 로 복사 (실행 시점 스냅샷)
2. `initialize_cluster_bath(...)` 로 `work_dir/initial_bath.dat` 생성 (반-원 DOS Bethe 시드)
3. `iter1`, `iter2`, … 폴더를 만들면서 각 iteration 안에서 step1 → step2 → step3 을 subprocess
   로 호출
4. step3 이 만든 `convergence.dat` 에 `converged true` 가 있으면 종료, 없으면 다음 iteration

> step3 가 미구현이라면 step3 호출에서 실패합니다. 한 iteration 만 step1+2 만 보고 싶다면
> `cluster_test_config.toml` 의 `[paths] work_dir` 을 적당히 둔 채로
> `cluster_step1_*` 과 `cluster_step2_*` 만 수동 호출하는 게 안전합니다.

### 2.2 수동 (per-step)

```bash
cd dmft

# step 1: 한 번만 호출
julia --project=.. scripts/cluster_step1_ground_state.jl \
    --config cluster_test_config.toml --iteration 1

# step 2: --all 옵션으로 symmetry 가 압축한 모든 (arm, site_j, direction) 조합 실행
julia --project=.. scripts/cluster_step2_greens_function.jl \
    --config cluster_test_config.toml --iteration 1 --all

# step 2 한 조합만 (디버깅용)
julia --project=.. scripts/cluster_step2_greens_function.jl \
    --config cluster_test_config.toml --iteration 1 \
    --arm 1 --site_j 1 --type greater
```

step1 이 `iter1/ground_state.h5` 를 만들어 두어야 step2 가 거기서 ψ_gs, H 를 로드합니다.

### 2.3 threading

```bash
# Julia threads, BLAS threads 는 config 의 [parallel.step{1,2}] 키로 조절 (아래 §4 참고)
# 블록 sparse 멀티스레딩을 켜려면 config 에 threaded_blocksparse = true
```

`ClusterDMFTLoop` 가 subprocess 환경에 `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`ITENSORS_THREADED_BLOCKSPARSE` 를 자동으로 주입합니다.

---

## 3. 인풋

### 3.1 사람이 만들어야 하는 파일

- **config TOML** — 예: `dmft/cluster_test_config.toml`. 자세한 키는 §4.

그 외 모든 입력은 코드가 알아서 만듭니다.

### 3.2 iteration 1 이 자동 생성하는 파일

`work_dir/initial_bath.dat` — Bethe semicircular `Γ(ω) = (1/2π)√(D²-ω²)` 로부터
`Bath.initial_cluster_bath` 가 만든 초기 bath 파라미터.

포맷 (cluster_bath_v2):
```
# format: cluster_bath_v2
# k  εₖ[1]  ReV₁[1]  ImV₁[1]  ReV₂[1]  ImV₂[1]  εₖ[2]  ReV₁[2]  ImV₁[2]  ReV₂[2]  ImV₂[2]  ...
1   ε_d   0           0           0           0           ε_d   ...
2   ε_d   0           0           0           0           ε_d   ...
3   ε_3   ReV₁_3      ImV₁_3      ReV₂_3      ImV₂_3      ε_3   ...
...
N_bath+2  ε_last  ...
```

- 각 arm `m ∈ {1..2·N_orb}` (1=orb1↑, 2=orb1↓, 3=orb2↑, 4=orb2↓) 마다
  `(εₖ, V₁, V₂)` 의 5개 열을 차지
- 행은 `N_bath + 2` 개 — **앞 2 행은 impurity site 1, 2 (V=0, ε=ε_d)**, 그 뒤가 bath
- `V₁, V₂` 는 각 bath level 이 cluster site 1, 2 에 결합하는 hybridization (복소수)
- `ε_d = -U/2 - (N_orb-1)(2U'-J)/2 + μ` (코드에서 자동 계산)

읽기/쓰기 헬퍼:
- 쓰기: `Bath.save_cluster_bath_params(filepath, εₖ, Vₖ)`
- 읽기: `Bath.load_cluster_bath_params(filepath, Lx)` → `(εₖ::Matrix{Float64}, Vₖ::Array{ComplexF64,3})`

### 3.3 iteration `n` (n≥2) 의 인풋

`work_dir/iter{n-1}/bath_params.dat` — 이전 iteration step3 가 만들어 둔 bath.
같은 cluster_bath_v2 포맷. **step3 의 출력 책임.**

step2 의 인풋은 같은 iter 의 `ground_state.h5` + `energy.dat` (step1 출력).

---

## 4. Config 키 설명

`dmft/cluster_test_config.toml` 을 기준으로 섹션별 정리.

### `[model]`

| 키 | 의미 |
|---|---|
| `U` | 같은-사이트 같은-spin Coulomb |
| `J` | Hund's coupling |
| `U_prime` | 다른-orbital 간 Coulomb. 생략 시 `U − 2J` |
| `t_cluster` | 두 cluster site 간 hopping (`t=0` 이면 single-site 와 일치해야 함 — 1차 sanity check) |
| `half_bandwidth` | `D`. Bethe lattice DOS `ρ(ε) = (2/πD²)√(D²−ε²)` |
| `N_orb` | orbital 수 (현재 2 만 검증) |
| `N_bath` | arm 당 bath site 수. 셀 크기 ∝ `2·N_orb·(N_bath+1)` |
| `rho` | 초기상태의 site 당 평균 점유. `0.5` = half-filling |
| `imp_init` | impurity 사이트 초기상태. `"Emp"`, `"Occ1"`, `"Occ2"`, `"Occ12"`, `"Bond"`, `"afm"` |
| `bath_init` | bath 초기 점유. `"pair_symmetric"` (기본) 또는 `"sequential"` |
| `bath_seed` | 초기 bath ε,V 생성 방식. `"diagonal"` (기본, site-resolved) 또는 `"common"` (rank-1 공통 채널) |
| `conserve_qns` | QN 보존 (`Nf`, `Sz`). 거의 항상 `true` |
| `mu` | chemical potential (기본 0) — `ε_d` 정의에만 들어감 |

### `[solver]`

| 키 | 의미 |
|---|---|
| `method` | `"complex_time"` 고정 |
| `theta` | 복소시간 회전각 `α`. `δt_ct = δt · exp(-iα)`. 일반적으로 `0.2~0.3` |
| `ph_symmetric` | true 면 `G^<(t) = conj(G^>(t))` 로 lesser 계산 생략. half-filling + PH 에서만 |
| `spin_symmetric` | true 면 down spin arm 을 up 으로부터 copy |
| `orbital_symmetric` | true 면 orb 2+ 를 orb 1 로부터 copy |

세 symmetry 다 켜면 step2 task 수가 `Lx × 2 × 2 = 16` 에서 `1 × 2 × 1 = 2` 로 줄어듭니다.

### `[dmrg]`

| 키 | 의미 |
|---|---|
| `skip` | true 면 DMRG 건너뛰고 초기상태를 그대로 ψ_gs 로 사용 (디버깅용) |
| `chi_x`, `chi_y` | 백본·암 결합 차원 |
| `max_iter` | 최대 sweep 수 |
| `convergence_tol` | δE 기준 |
| `method` | `"single_site"` 또는 `"two_site"` |
| `subspace_expansion` | `"cbe_3s"` 등 |
| `alpha`, `alpha_decay`, `delta`, `max_3s_sweeps` | 부분공간 확장 파라미터 |

`chi_x_schedule`, `chi_y_schedule` 은 코드에 하드코딩 (`[5=>10, 10=>20, 15=>30]`)되어 있어
config 에서 바꿀 수 없습니다 — 필요하면 [dmft/scripts/cluster_step1_ground_state.jl:141-142](dmft/scripts/cluster_step1_ground_state.jl#L141-L142)
에서 직접 수정.

### `[tdvp]`

| 키 | 의미 |
|---|---|
| `chi_x`, `chi_y` | TDVP 결합 차원 (DMRG 보다 약간 크게 잡는 게 보통) |
| `dt` | 실수 시간 간격 (`δt_ct` 의 modulus) |
| `max_step` | 시간 step 수. 총 시간 `t_max = dt · max_step` |
| `Ncut` | Krylov 차원 |
| `method` | `"hybrid"` (권장) 또는 `"two_site"`, `"single_site"` |
| `subspace_expansion` | `"cbe"` 등 |
| `verb_level` | 로그 verbosity |

> ESPRIT 가 pole 을 안정적으로 잡으려면 `t_max` 가 충분히 길어야 합니다. 현재 테스트 config
> 는 `dt=0.1, max_step=10` (t_max=1) 로 매우 짧음 — smoke test 전용. 실제 spectral function
> 을 뽑을 때는 `max_step=100~500` 권장.

### `[esprit]`

step3 가 읽음. 현재 step1+2 단계에서는 사용되지 않습니다.

| 키 | 의미 |
|---|---|
| `epsilon` | ESPRIT SVD truncation 임계값 |
| `pole_filter_mode` | `"adaptive"` 또는 `"fixed"` |
| `pole_filter_bound` | `"fixed"` 모드일 때 Im[ξ] 컷오프 |

### `[dmft]`

| 키 | 의미 |
|---|---|
| `omega_grid_range` | 출력 ω-grid 의 절반 폭 (`D` 단위). 스칼라면 `[-x·D, x·D]`, 리스트면 `[ω_min/D, ω_max/D]` |
| `omega_grid_points` | ω-grid 점 수 |
| `adaptive_discretization` | true 면 bath 이산화에서 equal-weight bin 사용 |
| `max_iterations` | DMFT 루프 최대 iter (step3 미구현이면 무의미) |
| `mixing_gamma` | bath mixing 계수 — step3 가 사용 |
| `convergence_tol` | δ 수렴 기준 — step3 가 판정 |

### `[parallel]`

| 키 | 의미 |
|---|---|
| `max_concurrent` | step2 task 동시 실행 수 |
| `nodes` | (옵션) SSH 노드 리스트. 비면 로컬 |
| `threaded_blocksparse` | true 면 ITensors blocksparse 멀티스레딩 |
| `[parallel.step1] blas_threads / julia_threads` | step1 자원 |
| `[parallel.step2] blas_threads / julia_threads` | step2 자원 |

`threaded_blocksparse=true` 일 때는 BLAS thread 가 자동으로 1 로 강제되고
Strided thread 도 disable 됩니다 ([cluster_step1_ground_state.jl:48-55](dmft/scripts/cluster_step1_ground_state.jl#L48-L55)).

### `[plotting]`

| 키 | 의미 |
|---|---|
| `enabled` | true 면 step3 가 끝난 뒤 `ClusterDMFTPlots.plot_iteration` 호출 |

step3 가 없으면 무시. step3 가 만든 spectral/hybridization 파일을 보고 그림을 그립니다.

### `[paths]`

| 키 | 의미 |
|---|---|
| `work_dir` | 결과 폴더. `dmft/` 기준 상대 경로 또는 절대 경로. 모든 iteration 폴더가 여기 안에 들어감 |

---

## 5. 아웃풋

### 5.1 work_dir 구조

```
work_dir/
├── dmft_config.toml              # 실행 시점 config 스냅샷
├── initial_bath.dat              # iter1 가 읽는 초기 bath
├── dmft_history.dat              # step3 가 매 iter 마다 append (E0, δ, sum_rule 등)
└── iter{n}/
    ├── step1.log                 # step1 stdout/stderr
    ├── step2_arm{x}_site{j}_{type}.log
    ├── step3.log
    │
    ├── ground_state.h5           # ψ_gs + H (step1 출력, step2 가 읽음)
    ├── energy.dat                # E₀ (한 줄)
    ├── filling.dat               # arm 별 ⟨n₁⟩, ⟨n₂⟩, ⟨n_tot⟩
    ├── bath_params.dat           # 이 iter 가 사용한 bath. step1 직후 input bath 그대로 복사.
    │                             # ※ step3 가 새 bath 를 같은 이름으로 덮어쓰는 것이 표준
    │                             # ※ step3 미구현이면 input bath 만 남아 있음
    │
    ├── G_greater_arm{x}_site{j}.dat   # step2 출력
    ├── G_lesser_arm{x}_site{j}.dat    # ph_symmetric=false 일 때만
    │
    │   (이하 step3 가 만들어야 하는 파일들 — 다음 iter 와 plotting 이 의존)
    ├── esprit_orb{orb}_{spin}_G{ij}.dat       # pole 표현 (학생 자유)
    ├── spectral_orb{orb}_{spin}_A{ij}.dat
    ├── hybridization_matrix_arm{x}.dat
    ├── GR_matrix_orb{orb}_{spin}.dat
    ├── bath_params.dat                         # 새 bath (위 input bath 를 덮어씀)
    ├── convergence.dat                         # iteration, delta, converged, sum_rule
    └── timing.dat                              # step별 wall time
```

### 5.2 step1 출력 상세

- **`ground_state.h5`** — `save_ftns` 와 `save_ftno` 가 같은 HDF5 에 두 그룹으로 저장
- **`energy.dat`** — `E₀` 한 줄 (15자리 소수)
- **`filling.dat`**
  ```
  # backbone  orb  spin  n1  n2  n_tot
  1  1  up  0.500589215653116  0.499211355391093  0.999800571044209
  2  1  dn  0.500592702417592  0.499222391527512  0.999815093945104
  ...
  ```
- **`bath_params.dat`** — 입력 bath 그대로 복사 (step3 가 덮어씀)

### 5.3 step2 출력 상세

- **`G_{greater|lesser}_arm{x}_site{j}.dat`** — 매트릭스 1개 column (`site_j` 번째)
  ```
  # t  Re[G_1j]  Im[G_1j]  Re[G_2j]  Im[G_2j]  max_S  χˣ  χʸ
  0.000000000000  0.000000000000e+00  -4.994107843e-01  0.000e+00  2.774700958e-01  0.0000  40  40
  ...
  ```
  - **j=1 파일**: 첫 두 데이터 열 = G_{1,1}(t), 다음 두 열 = G_{2,1}(t)
  - **j=2 파일**: 첫 두 데이터 열 = G_{1,2}(t), 다음 두 열 = G_{2,2}(t)
  - 즉 두 파일을 합치면 2×2 매트릭스 `G^{greater|lesser}_{i,j}(t)` 완성
- 시간격자 `t = k · δt`, `k=0,...,max_step`

### 5.4 step3 가 만들어야 하는 출력 (인터페이스만)

step3 를 직접 짤 때 다음만 만족하면 됩니다:

1. **`iter{n}/bath_params.dat`** — cluster_bath_v2 포맷. 다음 iter 의 step1 이
   `load_cluster_bath_params` 로 읽습니다.
2. **`iter{n}/convergence.dat`** — `ClusterDMFTLoop` 가 이 파일에서
   `startswith(line, "converged")` 후 `contains(line, "true")` 를 보고 루프 종료를 결정합니다.
   포맷 예:
   ```
   iteration  3
   delta      4.2e-06
   converged  true
   sum_rule   0.998
   ```
3. **`dmft_history.dat`** (work_dir 직속, append) — 선택. 수렴 추적용.
4. plotting 을 켜려면 `ClusterDMFTPlots.plot_iteration` 가 읽는 파일들 — 그 함수의 코드를
   읽어 필요한 파일만 만들면 됩니다.

step2 의 출력 (`G_{greater|lesser}_arm{x}_site{j}.dat`) 를 어떻게 읽고 ESPRIT 를 어떻게
태우는지는 자유. 단 symmetric 옵션 (`spin_symmetric`, `orbital_symmetric`, `ph_symmetric`)
에서 step2 가 만든 파일이 어떤 (arm, site_j, direction) 조합인지는 step3 가 일치시켜야 합니다
([cluster_step2_greens_function.jl:84-108](dmft/scripts/cluster_step2_greens_function.jl#L84-L108) `build_run_list`
로직 동일하게 미러링).

---

## 6. 빠른 sanity check 권장

config 를 바꿔가며 한 iter 만 돌려 보면서 확인:

1. **`t_cluster = 0.0`** — 두 cluster site 가 분리되어 single-site DMFT 와 같은 결과가
   나와야 합니다 (filling, E₀ 등). 두 site 의 `⟨n⟩` 가 동일하게 0.5 근처.
2. **symmetry off → 결과 동일** — `spin_symmetric=false` 로 두면 ↓ arm 도 직접 풀고,
   ↑ arm 결과와 일치하는지 비교 (PM 상태에서).
3. **G(0) 값** — `G_greater_arm*_site{j}.dat` 의 `t=0` 행 `Im[G_{j,j}] ≈ -(1−n_j)`
   (half-filling 이면 ≈ −0.5), 비대각은 `Im[G_{i,j}] = ⟨c_i c_j†⟩` (bonding 상관).
4. **DMRG δE → 1e-12 이하로 수렴** — 안 되면 `chi_x`, `chi_y`, `max_iter` 조정.
5. **TDVP `χ` 가 cap 에 닿지 않는지** — log 의 `χˣ`, `χʸ` 컬럼. `dt·max_step` 을 늘리면
   `chi_x`, `chi_y` 도 함께 늘려야 truncation 오차를 막을 수 있음.

---

## 7. 참고 파일 위치

- **모델 (FTNO/FTNS 빌더)**: [models/ClusterAndersonImpurityModel.jl](models/ClusterAndersonImpurityModel.jl)
  - `ftno_cluster_aim_model(model_params)` → `(Ws, phys_idx, aux_x_idx, aux_y_idx)`
  - `ftns_cluster_initial_state(phys_idx, ρ; ...)` → `(Ts, aux_x_idx, aux_y_idx)`
- **2SiteFermion sitetype**: [models/TwoSiteFermion.jl](models/TwoSiteFermion.jl)
- **Bath 헬퍼**: [dmft/Bath.jl](dmft/Bath.jl)
  - `bethe_hybridization(D)` → callable `Γ(ω)`
  - `initial_cluster_bath(Γ, ω_min, ω_max; N_bath, N_orb, ε_d, ...)` → `(εₖ, Vₖ)`
  - `discretize_hybridization_matrix(ω_grid, Δ_grid; N_bath, adaptive)` → `(ε_levels, V_levels)`
  - `save_cluster_bath_params`, `load_cluster_bath_params`
- **Green's function / ESPRIT**: [dmft/GreensFunction.jl](dmft/GreensFunction.jl)
  - `ComplexPoleGF`, `complex_time_to_spectral`, `analytic_continuation_esprit`
  - `scale_poles`, `mix_poles`, `save_pole_gf`, `load_pole_gf`
- **단일 사이트 reference**: [dmft/solvers/ComplexTimeSolver.jl](dmft/solvers/ComplexTimeSolver.jl) `run_step3` — 알고리즘 패턴이 거의 같이 가집니다 (scalar → 2×2 matrix 로 확장).
- **참고 문서**:
  - [dmft/docs/plans/2026-03-15-cdmft-2site-formulation.md](dmft/docs/plans/2026-03-15-cdmft-2site-formulation.md) — 2-site CDMFT 수식
  - [dmft/docs/plans/2026-03-19-grid-free-refactoring.md](dmft/docs/plans/2026-03-19-grid-free-refactoring.md) — 단일 사이트의 pole-form / Γ 클로저 mixing 설계 (matrix 화 시 참조)
  - [dmft/references/FTN_DMFT.pdf](dmft/references/FTN_DMFT.pdf) Eq. 39-42, 58-61 — cluster bath/impurity tensor 구조
