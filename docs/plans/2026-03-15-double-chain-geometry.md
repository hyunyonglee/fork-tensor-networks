# Double Chain Geometry Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable runtime selection of star or double chain geometry in the DMFT framework.

**Architecture:** Bath.jl gains a `star_to_double_chain` function (Lanczos tridiagonalization). `discretize` and `discretize_from_grid` accept a `geometry` kwarg. ComplexTimeSolver reads geometry from config and passes it through. The AIM model already handles both geometries.

**Tech Stack:** Julia, LinearAlgebra, QuadGK, TOML

**Design doc:** `docs/plans/2026-03-15-double-chain-geometry-design.md`

---

### Task 1: Add `_lanczos_tridiag` to Bath.jl

**Files:**
- Modify: `dmft/Bath.jl` (add before `_build_bath_matrices`, around line 102)

**Step 1: Add LinearAlgebra import**

At the top of Bath.jl, after `using QuadGK`, add:

```julia
using LinearAlgebra
```

**Step 2: Add `_lanczos_tridiag` function**

Insert before `_build_bath_matrices` (before line 103):

```julia
"""
    _lanczos_tridiag(ε, V) -> (E, J)

Lanczos tridiagonalization of diagonal bath Hamiltonian H = diag(ε)
with initial vector proportional to V.
Returns on-site energies E[n] and hoppings J[n], n=0 from impurity.
Ref: Kohn & Santoro, PRB 104, 014303 (2021), Eq.(9)
"""
function _lanczos_tridiag(ε::Vector{Float64}, V::Vector{Float64})
    N = length(ε)
    N == 0 && return (Float64[], Float64[])

    H = diagm(ε)

    J₀ = sqrt(sum(abs2, V))
    q_curr = V ./ J₀

    E = [dot(q_curr, H * q_curr)]
    J = [J₀]
    Q = [copy(q_curr)]

    q_prev = zeros(Float64, N)
    r_prev = 0.0

    for n in 1:(N - 1)
        w = H * q_curr .- E[end] .* q_curr .- r_prev .* q_prev

        for q in Q
            w .-= dot(q, w) .* q
        end

        r = sqrt(dot(w, w))
        if r < 1e-14
            break
        end

        q_next = w ./ r
        push!(J, r)
        push!(E, dot(q_next, H * q_next))
        push!(Q, copy(q_next))

        q_prev = q_curr
        q_curr = q_next
        r_prev = r
    end

    return (E, J)
end
```

**Step 3: Verify syntax**

Run: `julia -e 'include("dmft/Bath.jl"); using .Bath'`
Expected: No errors

**Step 4: Commit**

```bash
git add dmft/Bath.jl
git commit -m "feat(bath): add _lanczos_tridiag for star-to-DC mapping"
```

---

### Task 2: Add `star_to_double_chain` and `_build_bath_matrices_dc` to Bath.jl

**Files:**
- Modify: `dmft/Bath.jl`

**Step 1: Add `star_to_double_chain` after `_lanczos_tridiag`**

```julia
"""
    star_to_double_chain(εₖ, Vₖ) -> NamedTuple

Star geometry bath → double chain mapping via Lanczos tridiagonalization.
Splits bath into occupied (ε ≤ 0) and empty (ε > 0) parts, then
tridiagonalizes each independently.

Returns (E_emp, J_emp, E_occ, J_occ) in Lanczos natural order
(n=0 closest to impurity).

Ref: Kohn & Santoro, PRB 104, 014303 (2021), Eq.(9)
  i=1: empty chain (εₖ > 0), i=2: occupied chain (εₖ ≤ 0)
"""
function star_to_double_chain(εₖ::Vector{Float64}, Vₖ::Vector{Float64})
    N = length(εₖ)
    occ_mask = εₖ .≤ 0
    N_occ = sum(occ_mask)
    N_emp = N - N_occ

    if N_occ + N_emp != N
        error("star_to_double_chain: occ ($N_occ) + emp ($N_emp) ≠ N ($N)")
    end
    if isodd(N)
        @warn "star_to_double_chain: odd N_bath=$N — occupied chain will have $(N_occ) sites, empty chain $(N_emp) sites"
    end

    E_occ, J_occ = _lanczos_tridiag(εₖ[occ_mask], Vₖ[occ_mask])
    E_emp, J_emp = _lanczos_tridiag(εₖ[.!occ_mask], Vₖ[.!occ_mask])
    return (E_emp=E_emp, J_emp=J_emp, E_occ=E_occ, J_occ=J_occ)
end
```

**Step 2: Add `_build_bath_matrices_dc` after `_build_bath_matrices`**

```julia
"""
Build εₖ/Vₖ matrices for double chain geometry.
Column layout: [impurity, occ₁, emp₁, occ₂, emp₂, ...].
Even columns (2,4,...): occupied chain. Odd columns (3,5,...): empty chain.
"""
function _build_bath_matrices_dc(dc, N_orb, N_bath, ε_d)
    Lx = 2 * N_orb
    εₖ = zeros(Float64, Lx, N_bath + 1)
    Vₖ = zeros(Float64, Lx, N_bath + 1)

    for m in 1:Lx
        εₖ[m, 1] = ε_d
    end

    N_occ = length(dc.E_occ)
    N_emp = length(dc.E_emp)

    for i in 1:max(N_occ, N_emp)
        col_occ = 2 * i
        col_emp = 2 * i + 1
        if i <= N_occ && col_occ <= N_bath + 1
            for m in 1:Lx
                εₖ[m, col_occ] = dc.E_occ[i]
                Vₖ[m, col_occ] = dc.J_occ[i]
            end
        end
        if i <= N_emp && col_emp <= N_bath + 1
            for m in 1:Lx
                εₖ[m, col_emp] = dc.E_emp[i]
                Vₖ[m, col_emp] = dc.J_emp[i]
            end
        end
    end
    return εₖ, Vₖ
end
```

**Step 3: Add `star_to_double_chain` to the export line (line 16)**

Change:
```julia
export bethe_hybridization, discretize, discretize_from_grid
```
To:
```julia
export bethe_hybridization, discretize, discretize_from_grid, star_to_double_chain
```

**Step 4: Verify syntax**

Run: `julia -e 'include("dmft/Bath.jl"); using .Bath'`
Expected: No errors

**Step 5: Commit**

```bash
git add dmft/Bath.jl
git commit -m "feat(bath): add star_to_double_chain and DC matrix builder"
```

---

### Task 3: Add `geometry` kwarg to `discretize` and `discretize_from_grid`

**Files:**
- Modify: `dmft/Bath.jl` — `discretize` (line 39) and `discretize_from_grid` (line 68)

**Step 1: Modify `discretize` signature and return logic**

Change the function signature (line 39-41) from:
```julia
function discretize(Γ, ω_min::Real, ω_max::Real;
    N_bath::Int, N_orb::Int, ε_d::Float64=0.0,
    atol::Float64=1e-14, rtol::Float64=1e-10)
```
To:
```julia
function discretize(Γ, ω_min::Real, ω_max::Real;
    N_bath::Int, N_orb::Int, ε_d::Float64=0.0,
    geometry::String="star",
    atol::Float64=1e-14, rtol::Float64=1e-10)
```

Change the return line (line 59) from:
```julia
    return _build_bath_matrices(ε_list, V_list, N_orb, N_bath, ε_d)
```
To:
```julia
    if geometry == "double chain"
        dc = star_to_double_chain(ε_list, V_list)
        return _build_bath_matrices_dc(dc, N_orb, N_bath, ε_d)
    end
    return _build_bath_matrices(ε_list, V_list, N_orb, N_bath, ε_d)
```

**Step 2: Modify `discretize_from_grid` signature and return logic**

Change the function signature (line 68-71) from:
```julia
function discretize_from_grid(Δ_imag::AbstractVector{<:Real},
    ω_grid::AbstractVector{<:Real},
    ω_min::Real, ω_max::Real;
    N_bath::Int, N_orb::Int, ε_d::Float64=0.0)
```
To:
```julia
function discretize_from_grid(Δ_imag::AbstractVector{<:Real},
    ω_grid::AbstractVector{<:Real},
    ω_min::Real, ω_max::Real;
    N_bath::Int, N_orb::Int, ε_d::Float64=0.0,
    geometry::String="star")
```

Change the return line (line 100) from:
```julia
    return _build_bath_matrices(ε_list, V_list, N_orb, N_bath, ε_d)
```
To:
```julia
    if geometry == "double chain"
        dc = star_to_double_chain(ε_list, V_list)
        return _build_bath_matrices_dc(dc, N_orb, N_bath, ε_d)
    end
    return _build_bath_matrices(ε_list, V_list, N_orb, N_bath, ε_d)
```

**Step 3: Verify syntax**

Run: `julia -e 'include("dmft/Bath.jl"); using .Bath'`
Expected: No errors

**Step 4: Commit**

```bash
git add dmft/Bath.jl
git commit -m "feat(bath): add geometry kwarg to discretize functions"
```

---

### Task 4: Pass geometry through ComplexTimeSolver

**Files:**
- Modify: `dmft/solvers/ComplexTimeSolver.jl` — `run_step1` (line 86) and `run_step3` (line 357)

**Step 1: Update `run_step1` geometry parameter (line 86)**

Change:
```julia
        "Geometry" => "star",
```
To:
```julia
        "Geometry" => get(model, "geometry", "star"),
```

**Step 2: Update `run_step3` discretize call (lines 357-359)**

Change:
```julia
    εₖ, Vₖ = bath_update_mod.discretize_from_grid(
        Δ_imag, ω_grid, -D, D;
        N_bath=N_bath, N_orb=N_orb, ε_d=ε_d)
```
To:
```julia
    geometry = get(model, "geometry", "star")
    εₖ, Vₖ = bath_update_mod.discretize_from_grid(
        Δ_imag, ω_grid, -D, D;
        N_bath=N_bath, N_orb=N_orb, ε_d=ε_d, geometry=geometry)
```

**Step 3: Commit**

```bash
git add dmft/solvers/ComplexTimeSolver.jl
git commit -m "feat(solver): read geometry from config instead of hardcoding"
```

---

### Task 5: Pass geometry through DMFTLoop initial bath

**Files:**
- Modify: `dmft/DMFTLoop.jl` — `initialize_bath` (line 157)

**Step 1: Update `initialize_bath` to pass geometry**

Change line 157:
```julia
    εₖ, Vₖ = discretize(Γ, -D, D; N_bath=N_bath, N_orb=N_orb, ε_d=ε_d)
```
To:
```julia
    geometry = get(model, "geometry", "star")
    εₖ, Vₖ = discretize(Γ, -D, D; N_bath=N_bath, N_orb=N_orb, ε_d=ε_d, geometry=geometry)
```

**Step 2: Commit**

```bash
git add dmft/DMFTLoop.jl
git commit -m "feat(dmft): pass geometry to initial bath discretization"
```

---

### Task 6: Add geometry to config

**Files:**
- Modify: `dmft/dmft_config.toml` (line 8, after N_bath)

**Step 1: Add geometry field**

After line 8 (`N_bath = 99`), add:
```toml
geometry = "star"   # "star" or "double chain"
```

**Step 2: Commit**

```bash
git add dmft/dmft_config.toml
git commit -m "config: add geometry option to dmft_config.toml"
```

---

### Task 7: Validation — unit test for star_to_double_chain

**Files:**
- Create: `dmft/test_bath_dc.jl`

**Step 1: Write validation script**

```julia
# dmft/test_bath_dc.jl
# Validate star_to_double_chain against known results from test/star_to_DC.jl

using LinearAlgebra

include(joinpath(@__DIR__, "Bath.jl"))
using .Bath

include(joinpath(@__DIR__, "..", "test", "bethe_lattice.jl"))
include(joinpath(@__DIR__, "..", "test", "star_to_DC.jl"))

println("=== Test 1: Even N_bath ===")
N_bath = 20
D = 1
params = get_bethe_params(N_bath, D)
xn = params["xn"]
vn = params["vn"]

# Old function
old = get_DC_params(xn, vn)

# New function
new = star_to_double_chain(xn, vn)

# Compare (old returns reversed order)
for i in 1:length(new.E_occ)
    old_idx = length(old["xi_occ"]) - i + 1
    @assert isapprox(new.E_occ[i], old["xi_occ"][old_idx]; atol=1e-12) "E_occ mismatch at $i"
    @assert isapprox(new.J_occ[i], old["r_occ"][old_idx]; atol=1e-12) "J_occ mismatch at $i"
end
for i in 1:length(new.E_emp)
    old_idx = length(old["xi_emp"]) - i + 1
    @assert isapprox(new.E_emp[i], old["xi_emp"][old_idx]; atol=1e-12) "E_emp mismatch at $i"
    @assert isapprox(new.J_emp[i], old["r_emp"][old_idx]; atol=1e-12) "J_emp mismatch at $i"
end
println("  PASS: matches old get_DC_params (reversed order)")

println("\n=== Test 2: Odd N_bath (ε=0 handling) ===")
N_bath_odd = 19
params_odd = get_bethe_params(N_bath_odd, D)
xn_odd = params_odd["xn"]
vn_odd = params_odd["vn"]

dc_odd = star_to_double_chain(xn_odd, vn_odd)
N_occ = length(dc_odd.E_occ)
N_emp = length(dc_odd.E_emp)
@assert N_occ + N_emp == N_bath_odd "Total sites mismatch: $N_occ + $N_emp ≠ $N_bath_odd"
@assert N_occ == 10 "Expected 10 occupied (ε≤0), got $N_occ"
@assert N_emp == 9 "Expected 9 empty (ε>0), got $N_emp"
println("  PASS: odd N_bath handled correctly (occ=$N_occ, emp=$N_emp)")

println("\n=== Test 3: Tridiagonal reconstruction ===")
# Verify: reconstructed tridiag matrix has same eigenvalues as original diagonal
N_bath = 20
params = get_bethe_params(N_bath, D)
xn = params["xn"]
vn = params["vn"]
occ_xn = xn[xn .≤ 0]
dc = star_to_double_chain(xn, vn)

# Build tridiagonal matrix from DC params
N_occ = length(dc.E_occ)
T_mat = diagm(0 => dc.E_occ)
for i in 1:(N_occ - 1)
    T_mat[i, i+1] = dc.J_occ[i+1]
    T_mat[i+1, i] = dc.J_occ[i+1]
end
eig_tridiag = sort(eigvals(T_mat))
eig_orig = sort(occ_xn)
@assert isapprox(eig_tridiag, eig_orig; atol=1e-10) "Eigenvalue mismatch in occupied chain"
println("  PASS: tridiag eigenvalues match original bath energies")

println("\n=== All tests passed ===")
```

**Step 2: Run validation**

Run: `cd /Users/hylee/Dropbox/Programs/TENSOR_NETWORK/ITENSOR/ForkTensorNetworks && julia dmft/test_bath_dc.jl`
Expected: All tests pass

**Step 3: Commit**

```bash
git add dmft/test_bath_dc.jl
git commit -m "test: add validation for star_to_double_chain"
```

---

## Task Dependency Graph

```
Task 1 (_lanczos_tridiag)
  └─ Task 2 (star_to_double_chain + _build_bath_matrices_dc)
       ├─ Task 3 (geometry kwarg in discretize)
       │    ├─ Task 4 (ComplexTimeSolver passthrough)
       │    └─ Task 5 (DMFTLoop passthrough)
       ├─ Task 6 (config)
       └─ Task 7 (validation)
```

Tasks 4, 5, 6 are independent of each other (can be parallelized).
Task 7 depends on Tasks 1-2 only.
