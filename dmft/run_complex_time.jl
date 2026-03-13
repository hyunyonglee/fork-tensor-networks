# dmft/run_complex_time.jl
#
# Integration script: Complex-time Green's function + ESPRIT analytic continuation
# for Single-Orbital Anderson Impurity Model on Fork Tensor Network
#
# Usage: julia run_complex_time.jl

using Printf

# Load ForkTensorNetworks
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using ForkTensorNetworks

# Load model
include(joinpath(@__DIR__, "..", "models", "AndersonImpurityModel.jl"))
using .AndersonImpurityModel

# Load DMFT modules
include(joinpath(@__DIR__, "GreensFunction.jl"))
using .GreensFunction

# ═══════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════

# Model parameters
N_orb = 1
N_bath = 19
U = 4.0       # Hubbard U (= 2D for D=2)
D = 2.0       # Half-bandwidth

# Complex-time parameters
α = 0.2       # Contour angle
δt = 0.05     # D·Δt = 0.1 → Δt = 0.05 for D=2
max_step = 400  # Dt_max = 20

# DMRG parameters
χˣ_dmrg = 40
χʸ_dmrg = 40

# TDVP parameters
χˣ_tdvp = 40
χʸ_tdvp = 40

# ESPRIT parameters
ε_esprit = 1e-6

# ═══════════════════════════════════════════
# 1. Build Model
# ═══════════════════════════════════════════
println("[1] Building Anderson Impurity Model...")

# TODO: Fill in model construction based on your AIM setup:
# model_params = Dict(...)
# Ws, phys_idx, aux_x_idx, aux_y_idx = ftno_aim_model(model_params)
# H = ForkTensorNetworkOperator(Lx, Ly, phys_idx, aux_x_idx, aux_y_idx, Ws)
# Ts, _, _ , _ = ftns_initial_state(phys_idx, ρ)
# ψ₀ = ForkTensorNetworkState(Lx, Ly, phys_idx, aux_x_idx, aux_y_idx, Ts; χˣ=χˣ_dmrg, χʸ=χʸ_dmrg)

# ═══════════════════════════════════════════
# 2. Ground State via DMRG
# ═══════════════════════════════════════════
println("[2] Running DMRG for ground state...")

# dmrg_params = DMRGParams(; χˣ=χˣ_dmrg, χʸ=χʸ_dmrg, method=:two_site,
#                            max_iter=20, convergence_tol=1e-8)
# dmrg = DMRG(dmrg_params)
# E₀, ψ₀ = run_dmrg!(dmrg, H, ψ₀)
# println("  Ground state energy: E₀ = $E₀")

# ═══════════════════════════════════════════
# 3. Complex-Time Green's Function
# ═══════════════════════════════════════════
println("[3] Computing complex-time Green's function (α = $α)...")

# imp_x, imp_y = 1, 1  # Impurity site position in FTN

# gf, t_grid, G_gt, G_lt = compute_greens_function(
#     H, ψ₀, E₀, imp_x, imp_y,
#     TDVP, TDVPParams, run_tdvp!, overlap_ftn, applying_local_operators!;
#     α=α, δt=δt, max_step=max_step,
#     χˣ=χˣ_tdvp, χʸ=χʸ_tdvp,
#     ε=ε_esprit
# )

# ═══════════════════════════════════════════
# 4. Spectral Function
# ═══════════════════════════════════════════
println("[4] Computing spectral function...")

# ω_grid = collect(range(-6, 6, length=2000))
# A_ω = [spectral_function(gf, ω) for ω in ω_grid]

# ═══════════════════════════════════════════
# 5. Save Results
# ═══════════════════════════════════════════
println("[5] Saving results...")

# mkpath(joinpath(@__DIR__, "results"))
# open(joinpath(@__DIR__, "results", "spectral_alpha$(α).dat"), "w") do f
#     @printf(f, "# ω  A(ω)\n")
#     for (ω, A) in zip(ω_grid, A_ω)
#         @printf(f, "%.8f  %.12f\n", ω, A)
#     end
# end

# # Save Green's function time series
# open(joinpath(@__DIR__, "results", "greens_function_alpha$(α).dat"), "w") do f
#     @printf(f, "# t  Re[G>]  Im[G>]  Re[G<]  Im[G<]\n")
#     for (i, t) in enumerate(t_grid)
#         @printf(f, "%.8f  %.12f  %.12f  %.12f  %.12f\n",
#                 t, real(G_gt[i]), imag(G_gt[i]), real(G_lt[i]), imag(G_lt[i]))
#     end
# end

println("Done.")
