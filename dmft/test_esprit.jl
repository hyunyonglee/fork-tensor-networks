# dmft/test_esprit.jl
include("ESPRIT.jl")
using .ESPRIT
using Test

@testset "ESPRIT single exponential" begin
    Δt = 0.1
    N = 200
    t_grid = collect(0:N-1) .* Δt
    s_true = -0.1 + 3.0im
    R_true = 2.0 + 0.0im
    f = R_true .* exp.(s_true .* t_grid)

    result = esprit(f, Δt; ε=1e-10)

    @test length(result.s) == 1
    @test isapprox(result.s[1], s_true; atol=1e-6)
    @test isapprox(result.R[1], R_true; atol=1e-6)
end

@testset "ESPRIT multi-exponential" begin
    Δt = 0.1
    N = 300
    t_grid = collect(0:N-1) .* Δt

    s_true = [-0.05 + 2.0im, -0.1 - 1.5im, -0.2 + 4.0im]
    R_true = [1.0 + 0.5im, -0.5 + 0.3im, 0.8 - 0.2im]
    f = sum(R_true[l] .* exp.(s_true[l] .* t_grid) for l in 1:3)

    result = esprit(f, Δt; ε=1e-10)

    @test length(result.s) == 3

    # Match exponents (order may differ)
    for s_t in s_true
        dists = abs.(result.s .- s_t)
        @test minimum(dists) < 1e-4
    end

    # Verify reconstruction
    f_recon = sum(result.R[l] .* exp.(result.s[l] .* t_grid) for l in 1:length(result.s))
    @test maximum(abs.(f .- f_recon)) < 1e-6
end

@testset "ESPRIT with noise" begin
    Δt = 0.1
    N = 200
    t_grid = collect(0:N-1) .* Δt

    s_true = [-0.1 + 3.0im]
    R_true = [2.0 + 0.0im]
    f_clean = R_true[1] .* exp.(s_true[1] .* t_grid)
    noise = 1e-8 .* (randn(N) .+ im .* randn(N))
    f = f_clean .+ noise

    result = esprit(f, Δt; ε=1e-6)

    @test length(result.s) >= 1
    dists = abs.(result.s .- s_true[1])
    @test minimum(dists) < 1e-3
end

@testset "filter_and_refit" begin
    Δt = 0.1
    N = 200
    t_grid = collect(0:N-1) .* Δt

    # Two modes: one stable (decaying), one unstable (growing)
    s_stable = -0.1 + 2.0im
    s_unstable = 0.05 + 1.0im   # positive real part → growing
    R1, R2 = 1.0+0im, 0.5+0im
    f = R1 .* exp.(s_stable .* t_grid) .+ R2 .* exp.(s_unstable .* t_grid)

    result = esprit(f, Δt; ε=1e-10)

    # Filter: keep only modes with |exp(s * Δt)| ≤ 1 (non-growing)
    keep = abs.(exp.(result.s .* Δt)) .<= 1.0 + 1e-10
    filtered = ESPRIT.filter_and_refit(f, result.s, Δt, keep)

    @test length(filtered.s) == 1
    @test isapprox(filtered.s[1], s_stable; atol=1e-3)
end
