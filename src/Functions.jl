"""
    exp_taylor_sum(A, v₀, Ncut) -> ITensor

Compute the Taylor expansion of exp(A)|v₀⟩ truncated at order `Ncut`.
`A` is represented as a tuple of ITensors (and scalars) applied via contraction.
"""
function exp_taylor_sum(A::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, v₀::ITensor, Ncut::Integer)

    vₙ::ITensor = copy(v₀)
    v::ITensor = copy(v₀)
    for n = 1:Ncut
        vₙ .= noprime(reduce(*, A, init=vₙ))
        v .+= vₙ .* (1.0 / factorial(n))
    end

    return v
end


"""
    krylov_expm(H, x0; max_iter=10, tol=1e-8, verbose=false) -> ITensor

Compute exp(H)|x0⟩ using the Krylov subspace (Lanczos) method.
Builds a tridiagonal representation in the Krylov basis and exponentiates it.

# Arguments
- `H`: Effective Hamiltonian as a tuple of ITensors/scalars (includes the prefactor, e.g., -iδt).
- `x0`: Initial ITensor vector.
- `max_iter`: Maximum Krylov subspace dimension.
- `tol`: Convergence tolerance for the matrix exponential.
- `verbose`: Print convergence information.
"""
function krylov_expm(H::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, x0::ITensor; max_iter=10, tol=1.0E-8, shift::ComplexF64=0.0+0.0im, verbose=false)

    Vs = Vector{ITensor}(undef, max_iter + 1)
    T = Matrix{Complex}(undef, max_iter, max_iter)
    fill!(T, 0.0 + 0.0im)

    norm_x = norm(x0)
    if norm_x < 1e-14
        return x0
    end
    Vs[1] = x0 / norm_x
    exp_Av = similar(x0)
    δexp_Av = similar(x0)
    exp_Av_current = similar(x0)
    fill!(exp_Av, 0.0 + 0.0im)

    w = similar(x0)
    residual = 0.0

    for j = 1:max_iter

        w .= noprime(reduce(*, H, init=Vs[j]))
        for i = 1:j
            T[i, j] = scalar(dag(Vs[i]) * w)
            w .-= T[i, j] * Vs[i]
        end
        T[j, j] += shift

        if j < max_iter
            T[j+1, j] = norm(w)
            if abs(T[j+1, j]) < 1e-14
                exp_T = exp(T[1:j, 1:j])
                fill!(exp_Av_current, 0.0 + 0.0im)
                for i = 1:j
                    exp_Av_current .+= Vs[i] * exp_T[i, 1] * norm_x
                end
                verbose && println("--- Krylov exponentiation: invariant subspace found at iteration $j.")
                return exp_Av_current
            end
            Vs[j+1] = w / T[j+1, j]
        end

        exp_T = exp(T[1:j, 1:j])
        fill!(exp_Av_current, 0.0 + 0.0im)
        for i = 1:j
            exp_Av_current .+= Vs[i] * exp_T[i, 1] * norm_x
        end

        if j > 1
            δexp_Av .= exp_Av_current - exp_Av
            residual = abs(norm(δexp_Av) / norm(exp_Av_current))
            if residual < tol
                verbose && println("--- Krylov exponentiation converged after $j iterations with the residual $residual.")
                return exp_Av_current
            end
        end

        exp_Av .= exp_Av_current

    end

    verbose && println("--- Krylov exponentiation NOT converged after $max_iter iterations with the residual $residual.")

    return exp_Av
end


"""
    lanczos(H, x0; max_iter=10, E_tol=1e-14, R_tol=1e-8, prt=false) -> (ITensor, Float64)

Find the lowest eigenvalue and eigenvector of `H` using the Lanczos algorithm.
`H` is an effective Hamiltonian represented as a tuple of ITensors applied via contraction.

Returns `(Ritz_vec, E)` where `Ritz_vec` is the ground-state approximation and
`E` is the corresponding eigenvalue in the Krylov subspace.

# Arguments
- `H`: Effective Hamiltonian as a tuple of ITensors.
- `x0`: Initial guess ITensor.
- `max_iter`: Maximum number of Lanczos iterations.
- `E_tol`: Convergence tolerance for energy (relative change).
- `R_tol`: Convergence tolerance for Ritz residual.
- `prt`: Print convergence diagnostics.
"""
function lanczos(H::Tuple{Vararg{Union{ITensor,Float64}}}, x0::ITensor; max_iter=10, E_tol=1.0E-14, R_tol=1.0E-8, prt=false)

    Lan_vecs::Vector{ITensor} = ITensor[]
    Ritz_vec::ITensor = similar(x0)

    a = Float64[]
    b = Float64[]

    N_ortho::Int64 = 2
    E_old::Float64 = 10.0
    E_new::Float64 = 0.0
    ΔE::Float64 = 10.0
    Residual::Float64 = 10.0
    E_shift::Float64 = 0.0

    x2::ITensor = copy(x0)
    x1::ITensor = similar(x0)
    H_x::ITensor = similar(x0)
    fill!(H_x, 0.0 + 0.0im)
    β::Float64 = norm(x2)
    for i in 1:max_iter

        x1 .= x2 / β
        push!(Lan_vecs, copy(x1))

        H_x .= noprime(reduce(*, H, init=x1))
        x2 .= H_x - E_shift * x1
        α = real(scalar(dag(x1) * x2))
        append!(a, α)

        N_ortho = (i > 1) ? 2 : 1

        for j in 1:N_ortho
            x2 .-= (dag(Lan_vecs[end-j+1]) * x2) * Lan_vecs[end-j+1]
        end

        H_kryl = SymTridiagonal(copy(a), copy(b))

        F = eigen(H_kryl)
        E_new = F.values[1]
        kryl_eigenvec = F.vectors[:, 1]

        ΔE = (i > 1) ? norm(1.0 - E_new / E_old) : 10.0

        E_old = copy(E_new)
        E_kryl = E_new + E_shift
        Residual = norm(x2) * abs(kryl_eigenvec[end])

        if (ΔE < E_tol || Residual < R_tol || β < 1.0E-12)
            fill!(Ritz_vec, 0.0 + 0.0im)
            for ik in 1:length(a)
                Ritz_vec += kryl_eigenvec[ik] * Lan_vecs[ik]
            end
            if prt
                println("       [Lanczos] Converged after ", i, " iterations.")
                println("       [Lanczos] ΔE(kryl) = ", ΔE, ", E(kryl) = ", E_kryl)
                println("       [Lanczos] Ritz residual |β| = ", Residual)
            end
            break

        elseif (i == max_iter)
            fill!(Ritz_vec, 0.0 + 0.0im)
            for ik in 1:length(a)
                Ritz_vec += kryl_eigenvec[ik] * Lan_vecs[ik]
            end
            if prt
                println("       [Lanczos] Not Converged after ", i, " iterations.")
                println("       [Lanczos] ΔE(kryl) = ", ΔE, ", E(kryl) = ", E_kryl)
                println("       [Lanczos] Ritz residual |β| = ", Residual)
            end
            break
        end

        β = norm(x2)
        append!(b, β)
    end
    return Ritz_vec, E_new
end


"""
    print_dict(dict, title)

Pretty-print a `Dict` with a title header. The `"model"` key is printed first if present.
"""
function print_dict(dict::Dict, title::String)
    println("$title:")
    if haskey(dict, "model")
        println("*  model: $(dict["model"])")
    end
    for (key, value) in dict
        key != "model" && println("*  $key: $value")
    end
    println()
end


"""
    von_neumann_entropy(S::ITensor) -> Float64

Compute von Neumann entanglement entropy from singular value diagonal tensor S.
S_vN = -Σ pᵢ log(pᵢ), where pᵢ = sᵢ² / Σ sⱼ².
"""
function von_neumann_entropy(S::ITensor)
    s_vals = [S[i, i] for i in 1:minimum(size(S))]
    p = s_vals .^ 2
    p_sum = sum(p)
    p_sum == 0 && return 0.0
    p ./= p_sum
    return -sum(pᵢ > 0 ? pᵢ * log(pᵢ) : 0.0 for pᵢ in p)
end
