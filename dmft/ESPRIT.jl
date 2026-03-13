# dmft/ESPRIT.jl
module ESPRIT

using LinearAlgebra

export ESPRITResult, esprit, solve_vandermonde

struct ESPRITResult
    R::Vector{ComplexF64}   # weights
    s::Vector{ComplexF64}   # exponents
end

function build_hankel(f::AbstractVector{<:Number}, L::Int)
    N = length(f)
    rows = N - L
    H = Matrix{ComplexF64}(undef, rows, L + 1)
    for j in 1:(L+1), i in 1:rows
        H[i, j] = f[i + j - 1]
    end
    return H
end

function esprit(f::AbstractVector{<:Number}, Δt::Real; ε::Real=1e-6)
    N = length(f)
    L = round(Int, 2N / 5)

    # Build Hankel matrix (Yu et al. Eq. 13)
    H = build_hankel(f, L)

    # SVD
    F_svd = svd(H)
    σ = F_svd.S

    # Determine M — smallest index where σ_M < ε * σ_1
    threshold = ε * σ[1]
    M = findfirst(s -> s < threshold, σ)
    if M === nothing
        M = length(σ)
    else
        M = M - 1
    end
    M = max(M, 1)

    # Extract exponents via shift-invariance
    W = Matrix(transpose(F_svd.Vt)[:, 1:M])  # (L+1) × M
    W₀ = W[1:end-1, :]
    W₁ = W[2:end, :]
    F_mat = pinv(W₀) * W₁
    z = eigvals(F_mat)
    s = log.(z) ./ Δt

    # Solve Vandermonde system for weights (Eq. 14)
    R = solve_vandermonde(f, s, Δt)

    return ESPRITResult(R, s)
end

function solve_vandermonde(f::AbstractVector{<:Number}, s::AbstractVector{<:Number}, Δt::Real)
    N = length(f)
    M = length(s)
    x = exp.(s .* Δt)

    V = Matrix{ComplexF64}(undef, N, M)
    for l in 1:M, i in 1:N
        V[i, l] = x[l]^(i - 1)
    end

    R = V \ ComplexF64.(f)
    return R
end

end # module
