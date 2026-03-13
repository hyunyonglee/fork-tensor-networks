# dmft/GreensFunction.jl
module GreensFunction

using LinearAlgebra

export SpectralRepresentation, ComplexPoleGF, GridGF
export spectral_function

abstract type SpectralRepresentation end

spectral_function(g::SpectralRepresentation, ω::Real) = -imag(g(ω)) / π

struct ComplexPoleGF <: SpectralRepresentation
    A::Vector{ComplexF64}
    ξ::Vector{ComplexF64}
    η::Float64
end

function (g::ComplexPoleGF)(ω::Real)
    return sum(g.A ./ (ω + im * g.η .- g.ξ))
end

struct GridGF <: SpectralRepresentation
    ω_grid::Vector{Float64}
    G_ω::Vector{ComplexF64}
end

function (g::GridGF)(ω::Real)
    grid = g.ω_grid
    vals = g.G_ω
    if ω <= grid[1]
        return vals[1]
    elseif ω >= grid[end]
        return vals[end]
    end
    idx = searchsortedlast(grid, ω)
    idx = clamp(idx, 1, length(grid) - 1)
    t = (ω - grid[idx]) / (grid[idx+1] - grid[idx])
    return (1 - t) * vals[idx] + t * vals[idx+1]
end

include("ESPRIT.jl")
using .ESPRIT

export analytic_continuation_esprit
export complex_time_to_spectral

"""
    analytic_continuation_esprit(G_R_t, Δt; ε=1e-6, η=1e-10, pole_filter_bound=0.0) -> ComplexPoleGF

Apply ESPRIT to retarded Green's function G^R(t) to extract complex pole representation.
This is Pass 2 of the 2-pass pipeline.

G^R(t) ≈ -iθ(t) ∑ Aₗ exp(-iξₗ t)  →  G^R(ω) = ∑ Aₗ / (ω + iη - ξₗ)

Parameters:
- G_R_t: Retarded Green's function time series (t ≥ 0)
- Δt: Time step
- ε: ESPRIT SVD cutoff
- η: Broadening for G^R(ω)
- pole_filter_bound: Remove poles with Im(ξₗ) > pole_filter_bound (upper half plane = unphysical)
"""
function analytic_continuation_esprit(G_R_t::AbstractVector{<:Number}, Δt::Real;
                                       ε::Real=1e-6, η::Float64=1e-10,
                                       pole_filter_bound::Float64=0.0)
    # ESPRIT on G^R(t): G^R(t) = ∑ Rₗ exp(sₗ t)
    result = esprit(ComplexF64.(G_R_t), Δt; ε=ε)

    # Convert to pole representation:
    # G^R(t) = -i ∑ Aₗ exp(-i ξₗ t) = ∑ Rₗ exp(sₗ t)
    # → sₗ = -i ξₗ → ξₗ = i sₗ
    # → Rₗ = -i Aₗ → Aₗ = i Rₗ
    ξ_all = im .* result.s

    # Filter: keep poles in lower half plane (Im(ξₗ) < pole_filter_bound)
    keep = imag.(ξ_all) .< pole_filter_bound
    if !any(keep)
        @warn "All poles filtered out! Returning empty ComplexPoleGF."
        return ComplexPoleGF(ComplexF64[], ComplexF64[], η)
    end

    ξ_kept = ξ_all[keep]

    # Refit amplitudes with kept poles only
    s_kept = -im .* ξ_kept
    A_kept = im .* solve_vandermonde(ComplexF64.(G_R_t), s_kept, Δt)

    return ComplexPoleGF(A_kept, ξ_kept, η)
end

"""
    complex_time_to_spectral(G_greater_ct, G_lesser_ct, Δt, α;
                             ε=1e-6, η=1e-10, pole_filter_bound=0.0) -> ComplexPoleGF

Full ESPRIT 2-pass pipeline: complex-time G^≷(t;α) → real-frequency G^R(ω).

Pass 1: Fit G^>(t;α), G^<(t;α) with ESPRIT, filter divergent modes,
        reconstruct real-time G^>(t), G^<(t).
Pass 2: Form G^R(t), apply ESPRIT again → complex pole representation.

Reference: Yu et al. 2025, Section II.C
"""
function complex_time_to_spectral(G_greater_ct::AbstractVector{<:Number},
                                   G_lesser_ct::AbstractVector{<:Number},
                                   Δt::Real, α::Real;
                                   ε::Real=1e-6, η::Float64=1e-10,
                                   pole_filter_bound::Float64=0.0)
    N = length(G_greater_ct)
    t_grid = collect(0:N-1) .* Δt

    # ── Pass 1: ESPRIT on complex-time data ──
    res_greater = esprit(ComplexF64.(G_greater_ct), Δt; ε=ε)
    res_lesser = esprit(ComplexF64.(G_lesser_ct), Δt; ε=ε)

    # Filter: remove modes that diverge on real-time axis
    # ESPRIT gives s_ct where G(t_ct) = R exp(s_ct · t_ct).
    # Since t_ct = e^{-iα} t_real, we have s_real = s_ct · e^{iα}.
    # Check |exp(s_real · Δt)| = |exp(s_ct · e^{iα} · Δt)| ≤ 1
    rotation = exp(im * α)

    keep_gt = abs.(exp.(res_greater.s .* rotation .* Δt)) .<= 1.0 + 1e-10
    filtered_gt = filter_and_refit(ComplexF64.(G_greater_ct), res_greater.s, Δt, keep_gt)

    keep_lt = abs.(exp.(res_lesser.s .* rotation .* Δt)) .<= 1.0 + 1e-10
    filtered_lt = filter_and_refit(ComplexF64.(G_lesser_ct), res_lesser.s, Δt, keep_lt)

    # Reconstruct real-time Green's functions (Eq. 10)
    G_greater_rt = zeros(ComplexF64, N)
    for l in eachindex(filtered_gt.s)
        G_greater_rt .+= filtered_gt.R[l] .* exp.(filtered_gt.s[l] .* rotation .* t_grid)
    end

    G_lesser_rt = zeros(ComplexF64, N)
    for l in eachindex(filtered_lt.s)
        G_lesser_rt .+= filtered_lt.R[l] .* exp.(filtered_lt.s[l] .* rotation .* t_grid)
    end

    # Form G^R(t) = θ(t)[G^>(t) - G^<(t)], θ(t)=1 since t≥0
    G_R_t = G_greater_rt .- G_lesser_rt

    # ── Pass 2: ESPRIT on G^R(t) → ComplexPoleGF ──
    return analytic_continuation_esprit(G_R_t, Δt; ε=ε, η=η,
                                         pole_filter_bound=pole_filter_bound)
end

export compute_greens_function

"""
    compute_greens_function(H, ψ₀, E₀, site_x, site_y, tdvp_constructor, tdvp_params_constructor, run_tdvp_fn, overlap_fn, apply_ops_fn;
        α=0.0, δt=0.1, max_step=100, χˣ=20, χʸ=20, Ncut=10,
        tdvp_method=:two_site, subspace_expansion=:none, cbe_δ=0.1,
        ε=1e-6, η=1e-10, pole_filter_bound=0.0, verb_level=0)

Compute retarded Green's function from ground state using complex-time TDVP.

1. Creates particle (c†|ψ₀⟩) and hole (c|ψ₀⟩) excitations
2. Evolves each with TDVP using complex time steps δt·e^{∓iα}
3. Collects G^>(t;α) and G^<(t;α)
4. If α > 0: ESPRIT 2-pass → ComplexPoleGF
5. If α ≈ 0: direct Fourier transform → GridGF

Returns: (SpectralRepresentation, t_grid, G_greater, G_lesser)

The FTN functions are passed as arguments to avoid hard dependency on ForkTensorNetworks.
"""
function compute_greens_function(H, ψ₀, E₀::Real, site_x::Int, site_y::Int,
        tdvp_constructor, tdvp_params_constructor, run_tdvp_fn, overlap_fn, apply_ops_fn;
        α::Real=0.0, δt::Real=0.1, max_step::Int=100,
        χˣ::Int=20, χʸ::Int=20, Ncut::Int=10,
        tdvp_method::Symbol=:two_site,
        subspace_expansion::Symbol=:none, cbe_δ::Float64=0.1,
        ε::Real=1e-6, η::Float64=1e-10, pole_filter_bound::Float64=0.0,
        verb_level::Int=0, ω_max::Float64=10.0, broadening::Float64=0.05)

    t_grid = collect(0:max_step) .* δt

    # ── Create excitations ──

    # Particle excitation: c†|ψ₀⟩
    ψ_plus = deepcopy(ψ₀)
    apply_ops_fn(ψ_plus, [(site_x, site_y, "Cdag")])
    n_plus_sq = overlap_fn(ψ_plus, ψ_plus)
    n_plus = sqrt(abs(n_plus_sq))
    if n_plus > 1e-15
        for x in 1:ψ_plus.Lx, y in 1:ψ_plus.Ly
            ψ_plus.Ts[x, y] ./= n_plus
        end
    end

    # Hole excitation: c|ψ₀⟩
    ψ_minus = deepcopy(ψ₀)
    apply_ops_fn(ψ_minus, [(site_x, site_y, "C")])
    n_minus_sq = overlap_fn(ψ_minus, ψ_minus)
    n_minus = sqrt(abs(n_minus_sq))
    if n_minus > 1e-15
        for x in 1:ψ_minus.Lx, y in 1:ψ_minus.Ly
            ψ_minus.Ts[x, y] ./= n_minus
        end
    end

    # ── Save initial states for overlap ──
    ψ_plus_0  = deepcopy(ψ_plus)
    ψ_minus_0 = deepcopy(ψ_minus)

    # ── Complex time steps ──
    δt_greater = ComplexF64(δt * exp(-im * α))   # G^>: z₁(t) = e^{-iα}t
    δt_lesser  = ComplexF64(δt * exp(im * α))    # G^<: z₂(t) = e^{iα}t

    # ── Setup TDVP instances ──
    tdvp_gt = tdvp_constructor(tdvp_params_constructor(;
        χˣ=χˣ, χʸ=χʸ, method=tdvp_method, δt=δt_greater,
        Ncut=Ncut, subspace_expansion=subspace_expansion, δ=cbe_δ,
        verb_level=verb_level
    ))

    tdvp_lt = tdvp_constructor(tdvp_params_constructor(;
        χˣ=χˣ, χʸ=χʸ, method=tdvp_method, δt=δt_lesser,
        Ncut=Ncut, subspace_expansion=subspace_expansion, δ=cbe_δ,
        verb_level=verb_level
    ))

    # ── Collect Green's functions ──
    G_greater = Vector{ComplexF64}(undef, max_step + 1)
    G_lesser  = Vector{ComplexF64}(undef, max_step + 1)

    # t = 0
    G_greater[1] = -im * n_plus^2
    G_lesser[1]  = im * n_minus^2

    for step in 1:max_step
        run_tdvp_fn(tdvp_gt, H, ψ_plus, 1)
        run_tdvp_fn(tdvp_lt, H, ψ_minus, 1)

        t = step * δt
        z1 = exp(-im * α) * t
        z2 = exp(im * α) * t

        ovlp_gt = overlap_fn(ψ_plus_0, ψ_plus)
        ovlp_lt = overlap_fn(ψ_minus_0, ψ_minus)

        G_greater[step + 1] = -im * exp(im * E₀ * z1) * n_plus^2 * ovlp_gt
        G_lesser[step + 1]  = im * exp(-im * E₀ * z2) * n_minus^2 * conj(ovlp_lt)
    end

    # ── Analytic continuation ──
    if abs(α) < 1e-12
        # Real-time: direct Fourier transform → GridGF
        G_R_t = G_greater .- G_lesser
        ω_grid = collect(range(-ω_max, ω_max, length=2001))
        G_R_ω = zeros(ComplexF64, length(ω_grid))
        for (i, ω) in enumerate(ω_grid)
            for (j, t) in enumerate(t_grid)
                G_R_ω[i] += G_R_t[j] * exp(im * ω * t) * exp(-broadening * t) * δt
            end
        end
        gf = GridGF(ω_grid, G_R_ω)
    else
        # Complex-time: ESPRIT 2-pass pipeline → ComplexPoleGF
        gf = complex_time_to_spectral(G_greater, G_lesser, δt, α;
                                       ε=ε, η=η, pole_filter_bound=pole_filter_bound)
    end

    return gf, t_grid, G_greater, G_lesser
end

end # module
