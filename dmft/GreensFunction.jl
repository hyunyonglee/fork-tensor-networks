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

end # module
