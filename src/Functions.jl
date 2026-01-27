"""
    General functions
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""


"""
    exp_taylor_sum(A::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, v₀::ITensor, Ncut::Integer) -> ITensor
    return the Taylor sum of the exponential operator A acting on the input vector in ITensor
"""
function exp_taylor_sum(A::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, v₀::ITensor, Ncut::Integer)

    vₙ::ITensor = deepcopy(v₀)
    v::ITensor = deepcopy(v₀)
    for n = 1:Ncut
        vₙ .= noprime(reduce(*, A, init=vₙ))
        v .+= vₙ .* (1.0 / factorial(n))
    end

    return v
end



function krylov_expm(H::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, x0::ITensor; max_iter=10, tol=1.0E-8, verbose=false)

    Vs = Vector{ITensor}(undef, max_iter + 1) # Lanczos vectors
    T = Matrix{Complex}(undef, max_iter, max_iter) # Lanczos tridiagonal matrix
    fill!(T, 0.0 + 0.0im)

    norm_x = norm(x0)
    Vs[1] = deepcopy(x0) / norm_x
    exp_Av = deepcopy(x0)
    δexp_Av = deepcopy(x0)
    exp_Av_current = deepcopy(x0)
    fill!(exp_Av, 0.0 + 0.0im)

    w = deepcopy(x0)
    residual = 0.0

    for j = 1:max_iter

        w .= noprime(reduce(*, H, init=Vs[j]))
        for i = 1:j
            T[i, j] = scalar(dag(Vs[i]) * w)
            w .-= T[i, j] * Vs[i]
        end

        if j < max_iter
            T[j+1, j] = norm(w)
            Vs[j+1] = deepcopy(w) / T[j+1, j]
        end

        # Calculate current approximation
        exp_T = exp(T[1:j, 1:j])
        fill!(exp_Av_current, 0.0 + 0.0im)
        for i = 1:j
            exp_Av_current .+= Vs[i] * exp_T[i, 1] * norm_x
        end

        # Convergence check
        if j > 1
            δexp_Av .= exp_Av_current - exp_Av
            residual = abs(norm(δexp_Av) / norm(exp_Av_current))
            if residual < tol
                if verbose
                    println("--- Krylov exponentiation converged after $j iterations with the residual $residual.")
                end
                return exp_Av_current
            end
        end

        # Update for next iteration
        exp_Av .= exp_Av_current

    end # end for j

    if verbose
        println("--- Krylov exponentiation NOT converged after $max_iter iterations with the residual $residual.")
    end

    return exp_Av
end



function lanczos(H::Tuple{Vararg{Union{ITensor,Float64}}}, x0::ITensor; max_iter=3, E_tol=1.0E-14, R_tol=1.0E-8, prt=false)

    Lan_vecs::Vector{ITensor} = ITensor[]
    Ritz_vec::ITensor = deepcopy(x0) #ITensor(ComplexF64)

    a = Float64[]
    b = Float64[]

    N_ortho::Int64 = 2
    E_old::Float64 = 10.0
    E_new::Float64 = 0.0
    ΔE::Float64 = 10.0
    Residual::Float64 = 10.0
    E_shift::Float64 = 0.0

    x2::ITensor = deepcopy(x0)
    x1::ITensor = deepcopy(x0)
    H_x::ITensor = deepcopy(x0)
    fill!(H_x, 0.0 + 0.0im)
    β::Float64 = norm(x2)
    for i in 1:max_iter

        x1 .= deepcopy(x2) / β
        push!(Lan_vecs, deepcopy(x1))

        H_x .= noprime(reduce(*, H, init=x1))
        x2 .= H_x - E_shift * x1
        α = real(scalar(dag(x1) * x2))
        append!(a, deepcopy(α))

        if (i > 1)
            N_ortho = 2
        else
            N_ortho = 1
        end

        for j in 1:N_ortho
            # Gram-schmidt orthogonalization: |i> - Σⱼ|j><j|i>
            x2 .-= (dag(Lan_vecs[end-j+1]) * x2) * Lan_vecs[end-j+1]
        end

        # krylov Hamiltonian construction
        if (i != 1 && size(b)[1] > 0)
            H_kryl = diagm([a[i] for i ∈ 1:size(a)[1]]) + diagm(1 => [b[i] for i ∈ 1:size(b)[1]]) + diagm(-1 => [b[i] for i ∈ 1:size(b)[1]])

        elseif (i == 1)
            H_kryl = diagm([a[i] for i ∈ 1:size(a)[1]])
            # E_old = real(a[i] + 0.0);  # No matter to exit condition.
        end

        F = eigen(SymTridiagonal(H_kryl))
        E_new = F.values[1]
        kryl_eigenvec = F.vectors[:, 1]

        if (i > 1)
            #ΔE = norm( 1.0 - E_new / E_old )/abs(E_old);
            ΔE = norm(1.0 - E_new / E_old)
        else
            ΔE = 10.0
        end

        E_old = copy(E_new)
        E_kryl = E_new + E_shift
        Residual = norm(x2) * abs(kryl_eigenvec[end])

        if (ΔE < E_tol || Residual < R_tol || β < 1.0E-12)
            fill!(Ritz_vec, 0.0 + 0.0im)#Initilization
            for ik in 1:size(a)[1]
                Ritz_vec += kryl_eigenvec[ik] * Lan_vecs[ik]
            end
            if (prt == true)
                println("       [Lanczos] Converged after ", i, " iterations.")
                println("       [Lanczos] ΔE(kryl) = ", ΔE, ", E(kryl) = ", E_kryl)
                println("       [Lanczos] Ritz residual |β| = ", Residual)
            end
            break

        elseif (i == max_iter)
            fill!(Ritz_vec, 0.0 + 0.0im)#Initilization
            for ik in 1:size(a)[1]
                Ritz_vec += kryl_eigenvec[ik] * Lan_vecs[ik]
            end
            if (prt == true)
                println("       [Lanczos] Not Converged after ", i, " iterations.")
                println("       [Lanczos] ΔE(kryl) = ", ΔE, ", E(kryl) = ", E_kryl)
                println("       [Lanczos] Ritz residual |β| = ", Residual)
            end
            break
        end

        β = norm(x2)
        append!(b, deepcopy(β))
    end
    return Ritz_vec, E_new
end


function print_dict(dict::Dict, title::String)
    println("$title:")
    # "model" 키가 있다면 먼저 출력
    if haskey(dict, "model")
        println("*  model: $(dict["model"])")
    end

    # 나머지 키들을 출력
    for (key, value) in dict
        if key != "model"
            println("*  $key: $value")
        end
    end
    println()  # 줄바꿈을 위해 추가
end


function alter_overlap_ftn(ψ1::ForkTensorNetworkState, ψ2::ForkTensorNetworkState)
    tot = 1
    Tx = 1
    for x = 1:ψ1.Lx
        GC.gc()
        println("current x is = $x")
        flush(stdout)
        Ty = 1
        if x == 1
            for y = ψ1.Ly:-1:1
                Ty *= dag(replaceind(ψ1.Ts[x, y], ψ1.phys_idx[x, y], ψ2.phys_idx[x, y])) * noprime(prime(ψ2.Ts[x, y]); tags="Site")
                # println("Ty",size(Ty.tensor))
            end
            Tx *= Ty
        else#if x < ψ1.Lx
            alt = dag(ψ1.Ts[x,1]) * Tx
            alt2 = noprime(alt) * noprime(prime(ψ2.Ts[x, 1]); tags="Backbone,FTNS,x=($(x-1)-$(x)),y=1")
            zipped = replaceind(replaceind(alt2,inds(alt2)[findall(inds(alt2),"Site")[1]],dag(setprime(ψ1.phys_idx[x,1],4))),inds(alt2)[findall(inds(alt2),"Site")[2]],setprime(ψ2.phys_idx[x,1],5)) * delta(dag(setprime(ψ1.phys_idx[x,1],5)),setprime(ψ1.phys_idx[x,1],4))
            for y = ψ1.Ly:-1:2
                Ty *= dag(replaceind(ψ1.Ts[x, y], ψ1.phys_idx[x, y], ψ2.phys_idx[x, y])) * noprime(prime(ψ2.Ts[x, y]); tags="Site")
                # println("Ty",size(Ty.tensor))
            end
            Tx = zipped * Ty
        end

    end
        println("Tx",size(Tx.tensor))
        return scalar(Tx)
end

    

