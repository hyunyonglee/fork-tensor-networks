# Double Chain Geometry Integration Design

## Background

The DMFT framework currently hardcodes `"Geometry" => "star"` in `ComplexTimeSolver.jl`.
The AIM model (`AndersonImpurityModel.jl`) already supports both `"star"` and `"double chain"` Hamiltonians.
A star-to-double-chain mapping exists in `test/star_to_DC.jl` but is not integrated into the framework.

Reference: Kohn & Santoro, PRB 104, 014303 (2021)

## Design

### Approach: Bath module integration (Approach B)

Bath parameter format conversion belongs in `Bath.jl`. The solver passes geometry as config, Bath handles the mapping internally.

### Flow

```
dmft_config.toml: geometry = "star" | "double chain"
        |
        v
ComplexTimeSolver.run_step1():
  load bath params (always star format in .dat files)
        |
        v  if geometry == "double chain"
  Bath.star_to_double_chain(ε_list, V_list)
        |
        v
  _build_bath_matrices() with DC params
        |
        v
  model_params["Geometry"] = geometry
        |
        v
  AIM Hamiltonian (already supports both)
```

### 1. New function: `star_to_double_chain` in Bath.jl

Following paper Eq.(9) notation:
- i=1 (empty chain, εₖ > 0): on-site `E₁,ₙ`, hopping `J₁,ₙ`
- i=2 (occupied chain, εₖ ≤ 0): on-site `E₂,ₙ`, hopping `J₂,ₙ`

```julia
"""
    star_to_double_chain(εₖ, Vₖ) -> NamedTuple

Star geometry → double chain mapping via Lanczos tridiagonalization.
Ref: Kohn & Santoro, PRB 104, 014303 (2021), Eq.(9)

Returns (E_emp, J_emp, E_occ, J_occ) where indices follow Lanczos
natural order: n=0 (closest to impurity) to n=N-1.
"""
function star_to_double_chain(εₖ::Vector{Float64}, Vₖ::Vector{Float64})
    occ_mask = εₖ .≤ 0
    E_occ, J_occ = _lanczos_tridiag(εₖ[occ_mask], Vₖ[occ_mask])
    E_emp, J_emp = _lanczos_tridiag(εₖ[.!occ_mask], Vₖ[.!occ_mask])
    return (E_emp=E_emp, J_emp=J_emp, E_occ=E_occ, J_occ=J_occ)
end
```

Internal Lanczos function (extracts the duplicated occ/emp code):

```julia
"""
    _lanczos_tridiag(ε, V) -> (E, J)

Lanczos tridiagonalization of star-geometry bath Hamiltonian H = diag(ε)
with initial vector proportional to V.
Returns on-site energies E[n] and hoppings J[n] (n=0 from impurity).
"""
function _lanczos_tridiag(ε::Vector{Float64}, V::Vector{Float64})
    N = length(ε)
    N == 0 && return (Float64[], Float64[])

    H = diagm(ε)

    # Initial Lanczos vector: q₁ = V / ||V||
    J₀ = sqrt(sum(abs2, V))
    q_curr = V ./ J₀

    E = [dot(q_curr, H * q_curr)]
    J = [J₀]
    Q = [q_curr]  # for full reorthogonalization

    q_prev = zeros(Float64, N)
    r_prev = 0.0

    for n in 1:(N - 1)
        # Three-term recurrence: w = H*q - E*q - J_prev*q_prev
        w = H * q_curr .- E[end] .* q_curr .- r_prev .* q_prev

        # Full reorthogonalization
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
        push!(Q, q_next)

        q_prev = q_curr
        q_curr = q_next
        r_prev = r
    end

    return (E, J)
end
```

### 2. Modify `_build_bath_matrices` or add DC variant

For double chain geometry, the εₖ/Vₖ matrix layout follows the notebook convention:
- Column 1: impurity site
- Even columns (2,4,...): occupied chain sites (E₂,ₙ / J₂,ₙ)
- Odd columns (3,5,...): empty chain sites (E₁,ₙ / J₁,ₙ)

```julia
function _build_bath_matrices_dc(dc_params, N_orb, N_bath, ε_d)
    Lx = 2 * N_orb
    N_chain = length(dc_params.E_emp)  # sites per chain
    εₖ = zeros(Float64, Lx, N_bath + 1)
    Vₖ = zeros(Float64, Lx, N_bath + 1)

    for m in 1:Lx
        εₖ[m, 1] = ε_d
    end

    for i in 1:N_chain
        col_occ = 2 * i      # even columns: occupied
        col_emp = 2 * i + 1  # odd columns: empty
        for m in 1:Lx
            εₖ[m, col_occ] = dc_params.E_occ[i]
            Vₖ[m, col_occ] = dc_params.J_occ[i]
            if col_emp <= N_bath + 1
                εₖ[m, col_emp] = dc_params.E_emp[i]
                Vₖ[m, col_emp] = dc_params.J_emp[i]
            end
        end
    end
    return εₖ, Vₖ
end
```

### 3. Modify `discretize` and `discretize_from_grid`

Add `geometry` keyword argument:

```julia
function discretize(Γ, ω_min, ω_max; N_bath, N_orb, ε_d=0.0, geometry="star", ...)
    # ... existing star discretization → ε_list, V_list ...
    if geometry == "double chain"
        dc = star_to_double_chain(ε_list, V_list)
        return _build_bath_matrices_dc(dc, N_orb, N_bath, ε_d)
    end
    return _build_bath_matrices(ε_list, V_list, N_orb, N_bath, ε_d)
end
```

Same pattern for `discretize_from_grid`.

### 4. Modify `ComplexTimeSolver.run_step1` (line 86)

```julia
"Geometry" => get(model, "geometry", "star"),
```

### 5. Modify `ComplexTimeSolver.run_step3` (line 357-358)

Pass geometry to `discretize_from_grid`:

```julia
geometry = get(model, "geometry", "star")
εₖ, Vₖ = bath_update_mod.discretize_from_grid(
    Δ_imag, ω_grid, -D, D;
    N_bath=N_bath, N_orb=N_orb, ε_d=ε_d, geometry=geometry)
```

### 6. Add to `dmft_config.toml`

```toml
[model]
geometry = "star"   # "star" or "double chain"
```

### 7. N_bath constraint

- Star geometry: no constraint on N_bath
- Double chain: even N_bath recommended (both chains equal length)
- If odd N_bath + double chain: ε ≤ 0 → occupied (chains differ by 1 site)
- Add warning in `star_to_double_chain` if N_bath is odd

## Changes Summary

| File | Change |
|------|--------|
| `dmft/Bath.jl` | Add `star_to_double_chain`, `_lanczos_tridiag`, `_build_bath_matrices_dc`; modify `discretize`, `discretize_from_grid` |
| `dmft/solvers/ComplexTimeSolver.jl` | Read geometry from config in `run_step1` (line 86) and `run_step3` (line 357) |
| `dmft/dmft_config.toml` | Add `geometry` field under `[model]` |
| `test/star_to_DC.jl` | Deprecated (functionality moved to Bath.jl) |

## Validation

- Run existing notebook `test/example_aim_simulation_double_chain.ipynb` to verify energy match
- Star geometry: E = -20.1586 (baseline, should not change)
- Double chain: E = -20.1586 (must match star)
- DMFT loop: run 1-2 iterations with each geometry, verify bath params are consistent
