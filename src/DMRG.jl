"""
    Struct: DMRG
    Finding a ground state in FTNS for a given FTNO Hamiltonian using the DMRG algorithm
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
mutable struct DMRG

    params::Dict{String,Any}                                     # DMRG parameters
    εl::Union{Matrix{Union{ITensor,AbstractFloat}},Nothing}      # Left environments
    εr::Union{Matrix{Union{ITensor,AbstractFloat}},Nothing}      # Right environments
    εu::Union{Vector{Union{ITensor,AbstractFloat}},Nothing}      # Up environments
    εd::Union{Vector{Union{ITensor,AbstractFloat}},Nothing}      # Down environments
    E::Float64                                                   # Energy


    function DMRG(params::Dict{String,Any})

        if params["verbose"]
            print_dict(params, "* DMRG Parameters")
        end

        dmrg = new(
            params,
            nothing,
            nothing,
            nothing,
            nothing,
            0.0
        )

        return dmrg
    end

end # struct DMRG


"""
    run_dmrg!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ₀::ForkTensorNetworkState) -> AbstractFloat, ForkTensorNetworkState
    
    Run the DMRG algorithm with the Ĥamiltonian Ĥ in FTNO for a given the initial FTNS ψ₀.
    The function returns the ground-state energy E and the optimized FTNS ψ.
        # Example
        ```julia
        dmrg = DMRG(params)
        E, ψ = run_dmrg!(dmrg, Ĥ, ψ₀)

"""
function run_dmrg!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ₀::ForkTensorNetworkState)

    ψ = deepcopy(ψ₀)
    ψ.χˣ = dmrg.params["χˣ"]
    ψ.χʸ = dmrg.params["χʸ"]

    E₀ = 0.0
    δE = 1.0

    set_initial_environments!(dmrg, Ĥ, ψ)
    for i = 1:dmrg.params["max_iter"]

        if dmrg.params["method"] == "single-site"

            single_site_sweep!(dmrg, Ĥ, ψ)

            if dmrg.params["subspace_expansion"]
                dmrg.params["α"] *= dmrg.params["α_decay"]
            end

        elseif dmrg.params["method"] == "two-site"

            two_site_sweep!(dmrg, Ĥ, ψ)

        end

        δE = abs(dmrg.E - E₀) / abs(dmrg.E)
        E₀ = dmrg.E

        dmrg.params["verbose"] && println("Iteration: ", i, ", Energy: ", dmrg.E, ", δE: ", δE)

        δE < dmrg.params["convergence_tol"] && break
    end

    return dmrg.E, ψ

end # function run_dmrg!


"""
    set_initial_environments!(dmrg::DMRG)
"""
function set_initial_environments!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    dmrg.εl = Matrix{Union{ITensor,AbstractFloat,Nothing}}(undef, ψ.Lx, ψ.Ly)
    dmrg.εr = Matrix{Union{ITensor,AbstractFloat,Nothing}}(undef, ψ.Lx, ψ.Ly)
    dmrg.εu = Vector{Union{ITensor,AbstractFloat,Nothing}}(undef, ψ.Lx)
    dmrg.εd = Vector{Union{ITensor,AbstractFloat,Nothing}}(undef, ψ.Lx)

    #  matrices for the canonical center at (1, 1)
    canonical_form!(ψ, 1, 1)

    for x = 1:ψ.Lx
        dmrg.εr[x, ψ.Ly] = 1.0
        for y = (ψ.Ly-1):-1:1
            dmrg.εr[x, y] = dmrg.εr[x, y+1] * ψ.Ts[x, y+1] * Ĥ.Ws[x, y+1] * prime(dag(ψ.Ts[x, y+1]))
        end
    end

    dmrg.εd[ψ.Lx] = 1.0
    for x = (ψ.Lx-1):-1:1
        dmrg.εd[x] = ψ.Ts[x+1, 1] * dmrg.εd[x+1] * dmrg.εr[x+1, 1] * Ĥ.Ws[x+1, 1] * prime(dag(ψ.Ts[x+1, 1]))
    end

    dmrg.εu[1] = 1.0
end # function set_initial_environments!


"""
    single_site_sweep!(dmrg::DMRG)
"""
function single_site_sweep!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    for i = 1:(ψ.Lx-1)

        # dmrg.params["verbose"] && println(" -- Sweeping arm right at x = ", ψ.canonical_center[1])
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, Ĥ, ψ, "right")
        end

        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, Ĥ, ψ, "left")
        end

        # dmrg.params["verbose"] && println(" -- Updating backbone down")
        single_site_update_direction!(dmrg, Ĥ, ψ, "down")
    end


    for i = 1:(ψ.Lx-1)

        # dmrg.params["verbose"] && println(" -- Sweeping arm right at x = ", ψ.canonical_center[1])
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, Ĥ, ψ, "right")
        end

        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, Ĥ, ψ, "left")
        end

        # dmrg.params["verbose"] && println(" -- Updating backbone up")
        single_site_update_direction!(dmrg, Ĥ, ψ, "up")
    end

end # function single_site_sweep!


"""
    single_site_update_direction!(dmrg::DMRG, dir::String)
"""
function single_site_update_direction!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, dir::String)

    (x, y) = ψ.canonical_center
    error_check_update_direction(ψ.Lx, ψ.Ly, x, y, dir)

    # single-site update
    if y == 1
        Env = (dmrg.εu[x], dmrg.εd[x], Ĥ.Ws[x, y], dmrg.εr[x, y])
    else
        Env = (dmrg.εl[x, y], Ĥ.Ws[x, y], dmrg.εr[x, y])
    end
    ψ.Ts[x, y], dmrg.E = lanczos(Env, ψ.Ts[x, y];)

    # subspace expansion
    if dmrg.params["subspace_expansion"]
        single_site_subspace_expansion!(dmrg, Ĥ, ψ, dir)
    end

    # canonical center move along the direction
    canonical_center_move!(ψ, dir)

    # update the environment
    if dir == "right"

        if y == 1
            dmrg.εl[x, y+1] = ψ.Ts[x, y] * dmrg.εu[x] * Ĥ.Ws[x, y] * dmrg.εd[x] * prime(dag(ψ.Ts[x, y]))
        else
            dmrg.εl[x, y+1] = ψ.Ts[x, y] * dmrg.εl[x, y] * Ĥ.Ws[x, y] * prime(dag(ψ.Ts[x, y]))
        end

    elseif dir == "left"

        dmrg.εr[x, y-1] = ψ.Ts[x, y] * dmrg.εr[x, y] * Ĥ.Ws[x, y] * prime(dag(ψ.Ts[x, y]))

    elseif dir == "down"

        dmrg.εu[x+1] = ψ.Ts[x, y] * dmrg.εu[x] * Ĥ.Ws[x, 1] * dmrg.εr[x, 1] * prime(dag(ψ.Ts[x, y]))

    elseif dir == "up"

        dmrg.εd[x-1] = ψ.Ts[x, y] * dmrg.εd[x] * Ĥ.Ws[x, 1] * dmrg.εr[x, 1] * prime(dag(ψ.Ts[x, y]))

    end

end # function single_site_update_direction!


"""
    error_check_update_direction(dmrg::DMRG, dir::String)
"""
function error_check_update_direction(Lx::Integer, Ly::Integer, x::Integer, y::Integer, dir::String)

    if dir == "right"

        y == Ly && throw(ArgumentError("Canonical center should not be at the rightmost site of the system when updating the arm right."))

    elseif dir == "left"

        y == 1 && throw(ArgumentError("Canonical center should not be at the leftmost site of the system when updating the arm left."))

    elseif dir == "down"

        x == Lx && throw(ArgumentError("xc should not be at the Lx when updating the backbone down."))
        y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone."))

    elseif dir == "up"

        x == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
        y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone."))

    else
        throw(ArgumentError("Invalid direction."))
    end

end # function error_check_update_direction!


""" 
    bond_expansion(Env::Tuple{Vararg{Union{ITensor,AbstractFloat}}}, T::ITensor, idx_T::Index, idx_W::Index, α::AbstractFloat)
"""
function bond_expansion(Env::Tuple{Vararg{Union{ITensor,AbstractFloat}}}, T1::ITensor, T2::ITensor, idx_T::Index, idx_W::Index)

    P = noprime(reduce(*, Env, init=T1))
    C = combiner(idx_T, idx_W; dir=dir(idx_T))
    idx_exp = directsum(idx_T, combinedind(C); tags=tags(idx_T))

    T1_exp = directsum(idx_exp, T1 => idx_T, P * C => combinedind(C))
    T2_exp = T2 * delta(dag(idx_exp), idx_T)

    return T1_exp, T2_exp, idx_exp
end # function bond_expansion!


"""
    single_site_subspace_expansion!(dmrg::DMRG, dir::String)
"""
function single_site_subspace_expansion!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, dir::String)

    x = ψ.canonical_center[1]
    y = ψ.canonical_center[2]

    if dir == "right"

        if y == 1
            Env = (dmrg.params["α"], dmrg.εu[x], dmrg.εd[x], Ĥ.Ws[x, y])
        else
            Env = (dmrg.params["α"], dmrg.εl[x, y], Ĥ.Ws[x, y])
        end

        ψ.Ts[x, y], ψ.Ts[x, y+1], ψ.aux_y_idx[x, y] = bond_expansion(Env, ψ.Ts[x, y], ψ.Ts[x, y+1], ψ.aux_y_idx[x, y], Ĥ.aux_y_idx[x, y])

    elseif dir == "left"

        Env = (dmrg.params["α"], dmrg.εr[x, y], Ĥ.Ws[x, y])
        ψ.Ts[x, y], ψ.Ts[x, y-1], ψ.aux_y_idx[x, y-1] = bond_expansion(Env, ψ.Ts[x, y], ψ.Ts[x, y-1], dag(ψ.aux_y_idx[x, y-1]), dag(Ĥ.aux_y_idx[x, y-1]))

    elseif dir == "down"

        Env = (dmrg.params["α"], dmrg.εu[x], dmrg.εr[x, 1], Ĥ.Ws[x, 1])
        ψ.Ts[x, 1], ψ.Ts[x+1, 1], ψ.aux_x_idx[x] = bond_expansion(Env, ψ.Ts[x, 1], ψ.Ts[x+1, 1], ψ.aux_x_idx[x], Ĥ.aux_x_idx[x])

    elseif dir == "up"

        Env = (dmrg.params["α"], dmrg.εd[x], dmrg.εr[x, 1], Ĥ.Ws[x, 1])
        ψ.Ts[x, 1], ψ.Ts[x-1, 1], ψ.aux_x_idx[x-1] = bond_expansion(Env, ψ.Ts[x, 1], ψ.Ts[x-1, 1], dag(ψ.aux_x_idx[x-1]), dag(Ĥ.aux_x_idx[x-1]))

    end

end # function single_site_subspace_expansion!


"""
    two_site_sweep!(dmrg::DMRG)
"""
function two_site_sweep!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    for i = 1:(ψ.Lx-1)

        # dmrg.params["verbose"] && println("Sweeping arm at x = ", ψ.canonical_center[1])
        two_site_sweep_arm_right!(dmrg, Ĥ, ψ)
        two_site_sweep_arm_left!(dmrg, Ĥ, ψ)

        # dmrg.params["verbose"] && println("Updating backbone down")
        two_site_update_backbone_down!(dmrg, Ĥ, ψ)
    end

    for i = 1:(ψ.Lx-1)

        # dmrg.params["verbose"] && println("Sweeping arm at x = ", ψ.canonical_center[1])
        two_site_sweep_arm_right!(dmrg, Ĥ, ψ)
        two_site_sweep_arm_left!(dmrg, Ĥ, ψ)

        # dmrg.params["verbose"] && println("Updating backbone up")
        two_site_update_backbone_up!(dmrg, Ĥ, ψ)
    end

end # function two_site_sweep!



function two_site_sweep_arm_right!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the arm right."))

    x = ψ.canonical_center[1]
    χʸ = dmrg.params["χʸ"]

    for y = 1:(ψ.Ly-1)

        if y == 1

            T, dmrg.E = lanczos((dmrg.εu[x], dmrg.εd[x], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1], dmrg.εr[x, y+1]), ψ.Ts[x, y] * ψ.Ts[x, y+1];)

            if dmrg.params["subspace_expansion"]
                Env = (dmrg.params["α"], dmrg.εu[x], dmrg.εd[x], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1])
                T, ψ.Ts[x, y+2], ψ.aux_y_idx[x, y+1] = bond_expansion(Env, T, ψ.Ts[x, y+2], ψ.aux_y_idx[x, y+1], Ĥ.aux_y_idx[x, y+1])
            end

            V, S, U = svd(T, (ψ.aux_y_idx[x, y+1], ψ.phys_idx[x, y+1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y]))
            dmrg.εl[x, y+1] = U * dmrg.εu[x] * dmrg.εd[x] * Ĥ.Ws[x, y] * prime(dag(U))

        else

            T, dmrg.E = lanczos((dmrg.εl[x, y], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1], dmrg.εr[x, y+1]), ψ.Ts[x, y] * ψ.Ts[x, y+1];)

            if dmrg.params["subspace_expansion"] && y < (ψ.Ly - 1)
                Env = (dmrg.params["α"], dmrg.εl[x, y], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1])
                T, ψ.Ts[x, y+2], ψ.aux_y_idx[x, y+1] = bond_expansion(Env, T, ψ.Ts[x, y+2], ψ.aux_y_idx[x, y+1], Ĥ.aux_y_idx[x, y+1])
            end

            U, S, V = svd(T, (ψ.aux_y_idx[x, y-1], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y]))
            dmrg.εl[x, y+1] = U * dmrg.εl[x, y] * Ĥ.Ws[x, y] * prime(dag(U))

        end

        ψ.Ts[x, y] = U
        ψ.Ts[x, y+1] = S * V
        ψ.aux_y_idx[x, y] = commonind(U, S)

        network_update!(ψ, "right")

    end

end # function sweep_arm_right!


function two_site_sweep_arm_left!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    ψ.canonical_center[2] != ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site of the system when updating the arm left."))

    x = ψ.canonical_center[1]
    χʸ = dmrg.params["χʸ"]

    for y = ψ.Ly:-1:2

        if y == 2
            T, dmrg.E = lanczos((dmrg.εu[x], dmrg.εd[x], Ĥ.Ws[x, y-1], Ĥ.Ws[x, y], dmrg.εr[x, y]), ψ.Ts[x, y-1] * ψ.Ts[x, y];)
            V, S, U = svd(T, (ψ.aux_y_idx[x, y], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y-1]))
        else
            T, dmrg.E = lanczos((dmrg.εl[x, y-1], Ĥ.Ws[x, y-1], Ĥ.Ws[x, y], dmrg.εr[x, y]), ψ.Ts[x, y-1] * ψ.Ts[x, y];)

            if dmrg.params["subspace_expansion"]
                Env = (dmrg.params["α"], dmrg.εr[x, y], Ĥ.Ws[x, y], Ĥ.Ws[x, y-1])
                T, ψ.Ts[x, y-2], ψ.aux_y_idx[x, y-2] = bond_expansion(Env, T, ψ.Ts[x, y-2], dag(ψ.aux_y_idx[x, y-2]), dag(Ĥ.aux_y_idx[x, y-2]))
            end

            U, S, V = svd(T, (ψ.aux_y_idx[x, y-2], ψ.phys_idx[x, y-1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y-1]))
        end
        dmrg.εr[x, y-1] = V * dmrg.εr[x, y] * Ĥ.Ws[x, y] * prime(dag(V))

        ψ.Ts[x, y] = V
        ψ.Ts[x, y-1] = U * S
        ψ.aux_y_idx[x, y-1] = commonind(S, V) # 순서 주의

        network_update!(ψ, "left")

    end

end # function sweep_arm_left!


function two_site_update_backbone_down!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    ψ.canonical_center[1] == ψ.Lx && throw(ArgumentError("xc should not be at the Lx when updating the backbone down."))
    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    x = ψ.canonical_center[1]
    χˣ = dmrg.params["χˣ"]

    T, dmrg.E = lanczos((dmrg.εu[x], dmrg.εr[x, 1], Ĥ.Ws[x, 1], Ĥ.Ws[x+1, 1], dmrg.εr[x+1, 1], dmrg.εd[x+1]), ψ.Ts[x, 1] * ψ.Ts[x+1, 1];)

    if dmrg.params["subspace_expansion"] && x < (ψ.Lx - 1)
        Env = (dmrg.params["α"], dmrg.εu[x], dmrg.εr[x, 1], Ĥ.Ws[x, 1], Ĥ.Ws[x+1, 1], dmrg.εd[x+1])
        T, ψ.Ts[x+1, 2], ψ.aux_y_idx[x+1, 1] = bond_expansion(Env, T, ψ.Ts[x+1, 2], ψ.aux_y_idx[x+1, 1], Ĥ.aux_y_idx[x+1, 1])
    end

    if x == 1
        U, S, V = svd(T, (ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    else
        U, S, V = svd(T, (ψ.aux_x_idx[x-1], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    end
    dmrg.εu[x+1] = U * dmrg.εu[x] * dmrg.εr[x, 1] * Ĥ.Ws[x, 1] * prime(dag(U))

    ψ.Ts[x, 1] = U
    ψ.Ts[x+1, 1] = S * V
    ψ.aux_x_idx[x] = commonind(U, S)

    network_update!(ψ, "down")

end # function update_backbone_down!


function two_site_update_backbone_up!(dmrg::DMRG, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    ψ.canonical_center[1] == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    x = ψ.canonical_center[1]
    χˣ = dmrg.params["χˣ"]

    T, dmrg.E = lanczos((dmrg.εu[x-1], dmrg.εr[x-1, 1], Ĥ.Ws[x-1, 1], Ĥ.Ws[x, 1], dmrg.εr[x, 1], dmrg.εd[x]), ψ.Ts[x-1, 1] * ψ.Ts[x, 1];)

    if dmrg.params["subspace_expansion"] && x > 2
        Env = (dmrg.params["α"], dmrg.εd[x], dmrg.εr[x, 1], Ĥ.Ws[x, 1], Ĥ.Ws[x-1, 1], dmrg.εu[x-1])
        T, ψ.Ts[x-1, 2], ψ.aux_y_idx[x-1, 1] = bond_expansion(Env, T, ψ.Ts[x-1, 2], ψ.aux_y_idx[x-1, 1], Ĥ.aux_y_idx[x-1, 1])
    end

    if x == ψ.Lx
        U, S, V = svd(T, (ψ.aux_x_idx[x-2], ψ.aux_y_idx[x-1, 1], ψ.phys_idx[x-1, 1]); cutoff=1e-10, maxdim=χˣ, righttags=tags(ψ.aux_x_idx[x-1]))
    else
        V, S, U = svd(T, (ψ.aux_x_idx[x], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x-1]))
    end
    dmrg.εd[x-1] = V * dmrg.εd[x] * dmrg.εr[x, 1] * Ĥ.Ws[x, 1] * prime(dag(V))

    ψ.Ts[x, 1] = V
    ψ.Ts[x-1, 1] = S * U
    ψ.aux_x_idx[x-1] = commonind(S, V) # 순서 주의

    network_update!(ψ, "up")

end # function update_backbone_up!
