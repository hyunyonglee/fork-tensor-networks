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
