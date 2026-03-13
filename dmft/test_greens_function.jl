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

@testset "analytic_continuation_esprit" begin
    # Known G^R(t) = -i ∑ Aₗ exp(-iξₗ t) with two poles
    ξ_true = [-1.0 - 0.2im, 1.0 - 0.3im]
    A_true = [0.6 + 0.0im, 0.4 + 0.0im]

    Δt = 0.1
    N = 400
    t_grid = collect(0:N-1) .* Δt

    G_R_t = -im .* sum(A_true[l] .* exp.(-im .* ξ_true[l] .* t_grid) for l in 1:2)

    gf = analytic_continuation_esprit(G_R_t, Δt; ε=1e-10, pole_filter_bound=0.0)

    # Check spectral function has peaks near ω = -1.0 and ω = 1.0
    A_m1 = spectral_function(gf, -1.0)
    A_p1 = spectral_function(gf, 1.0)
    A_0  = spectral_function(gf, 0.0)

    @test A_m1 > A_0
    @test A_p1 > A_0
    @test A_m1 > 0.1
    @test A_p1 > 0.1
end

@testset "complex_time_to_spectral full pipeline" begin
    # Known poles
    ξ_true = [-1.0 - 0.2im, 1.0 - 0.3im]
    A_true = [0.6 + 0.0im, 0.4 + 0.0im]

    Δt = 0.1
    N = 500
    t_grid = collect(0:N-1) .* Δt
    α = 0.2

    # G^>(t) comes from particle excitation, G^<(t) from hole excitation
    # For testing: construct them so G^R = G^> - G^<
    # G^>(t) = -i A₁ exp(-i ξ₁ t), G^<(t) = -(-i A₂ exp(-i ξ₂ t)) = i A₂ exp(-i ξ₂ t)
    # Then G^R = G^> - G^< = -i[A₁ exp(-iξ₁t) + A₂ exp(-iξ₂t)] ✓

    # Complex-time versions: G^≷(t;α) = G^≷(e^{-iα}t)
    # Both use the same rotation since time evolution is along same contour
    rot = exp(-im * α)
    G_greater_ct = -im .* A_true[1] .* exp.(-im .* ξ_true[1] .* rot .* t_grid)
    G_lesser_ct  = im .* A_true[2] .* exp.(-im .* ξ_true[2] .* rot .* t_grid)

    gf = complex_time_to_spectral(G_greater_ct, G_lesser_ct, Δt, α;
                                   ε=1e-10, pole_filter_bound=0.0)

    # Spectral function should have peaks near ω = -1.0 and ω = 1.0
    A_m1 = spectral_function(gf, -1.0)
    A_p1 = spectral_function(gf, 1.0)
    A_0  = spectral_function(gf, 0.0)

    @test A_m1 > 0.05
    @test A_p1 > 0.05
    @test A_m1 > A_0
    @test A_p1 > A_0
end
