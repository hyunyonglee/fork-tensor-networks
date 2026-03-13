# dmft/test_greens_function.jl
include("GreensFunction.jl")
using .GreensFunction
using Test

@testset "ComplexPoleGF callable" begin
    A = [1.0 + 0.0im]
    ξ = [-1.0 - 0.5im]
    η = 1e-10
    gf = ComplexPoleGF(A, ξ, η)

    val = gf(-1.0)
    @test imag(val) < 0  # retarded GF has negative imaginary part

    A_ω = spectral_function(gf, -1.0)
    @test A_ω > 0
end

@testset "ComplexPoleGF multi-pole" begin
    A = [0.5 + 0.0im, 0.5 + 0.0im]
    ξ = [-1.0 - 0.1im, 1.0 - 0.1im]
    gf = ComplexPoleGF(A, ξ, 1e-10)

    A_m1 = spectral_function(gf, -1.0)
    A_0  = spectral_function(gf, 0.0)
    A_p1 = spectral_function(gf, 1.0)

    @test A_m1 > A_0
    @test A_p1 > A_0
end

@testset "GridGF interpolation" begin
    ω_grid = collect(-5.0:0.1:5.0)
    # Lorentzian: A(ω) = (1/π) · Γ / ((ω-ω₀)² + Γ²)
    Γ = 0.3
    ω₀ = 0.0
    G_ω = @. -1.0 / (ω_grid + im*Γ - ω₀)  # simple single pole on grid
    gf = GridGF(ω_grid, G_ω)

    # Test interpolation at grid point
    idx = findfirst(x -> x ≈ 0.0, ω_grid)
    @test isapprox(gf(0.0), G_ω[idx]; atol=1e-10)

    # Test interpolation between grid points
    val = gf(0.05)
    @test !isnan(real(val))
    @test !isnan(imag(val))
end
