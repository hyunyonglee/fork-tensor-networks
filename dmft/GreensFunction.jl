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

end # module
