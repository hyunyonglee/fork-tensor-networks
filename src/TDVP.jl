"""
    Struct: TDVP
    Time-evolving a FTN state for a given FTNO Hamiltonian using the TDVP algorithm
    Reference: SciPost Phys. 8, 024 (2020)
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
mutable struct TDVP

    params::Dict{String,Any}                               # TDVP parameters
    εl::Union{Matrix{Union{ITensor,Float64}},Nothing}      # Left environments
    εr::Union{Matrix{Union{ITensor,Float64}},Nothing}      # Right environments
    εu::Union{Vector{Union{ITensor,Float64}},Nothing}      # Up environments
    εd::Union{Vector{Union{ITensor,Float64}},Nothing}      # Down environments
    time::Union{Float64,Complex}                           # Time
    env_set::Bool                                          # Flag for the initial environments


    function TDVP(params::Dict{String,Any})

        print_dict(params, "* TDVP Parameters")

        tdvp = new(
            params,
            nothing,
            nothing,
            nothing,
            nothing,
            0.0,
            false
        )

        return tdvp
    end

end # struct TDVP



function run_tdvp!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, max_step::Integer)

    if !tdvp.env_set
        set_initial_environments!(tdvp, Ĥ, ψ)
    end

    # @printf("* TDVP starts at time %.2f...\n", tdvp.time)

    for i = 1:max_step

        if tdvp.params["verb_level"] > 0
            @printf("*  running at time %.2f...\n", tdvp.time)
        end

        if tdvp.params["method"] == "single-site"

            single_site_time_evolution_sweep_direction!(tdvp, Ĥ, ψ; direction="down", half_step=true)
            single_site_time_evolution_sweep_direction!(tdvp, Ĥ, ψ; direction="up", half_step=true)

        elseif tdvp.params["method"] == "two-site"

            two_site_time_evolution_sweep_direction!(tdvp, Ĥ, ψ; direction="down", half_step=true)
            two_site_time_evolution_sweep_direction!(tdvp, Ĥ, ψ; direction="up", half_step=true)

        else
            throw(ArgumentError("Invalid method"))
        end
        tdvp.time += tdvp.params["δt"]

    end

    # @printf("* TDVP ends at time %.2f...\n", tdvp.time)

end # function run_tdvp!



function set_initial_environments!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    tdvp.εl = Matrix{Union{ITensor,Float64}}(undef, ψ.Lx, ψ.Ly)
    tdvp.εr = Matrix{Union{ITensor,Float64}}(undef, ψ.Lx, ψ.Ly)
    tdvp.εu = Vector{Union{ITensor,Float64}}(undef, ψ.Lx)
    tdvp.εd = Vector{Union{ITensor,Float64}}(undef, ψ.Lx)

    #  matrices for the canonical center at (1, Ly)
    canonical_form!(ψ, 1, ψ.Ly)

    tdvp.εr[1, ψ.Ly] = 1.0
    for x = 2:ψ.Lx
        tdvp.εr[x, ψ.Ly] = 1.0
        for y = ψ.Ly:-1:2
            update_environment!(tdvp, Ĥ, ψ, x, y, "left")
        end
    end

    tdvp.εd[ψ.Lx] = 1.0
    for x = ψ.Lx:-1:2
        update_environment!(tdvp, Ĥ, ψ, x, 1, "up")
    end

    tdvp.εu[1] = 1.0
    for y = 1:(ψ.Ly-1)
        update_environment!(tdvp, Ĥ, ψ, 1, y, "right")
    end

    tdvp.env_set = true

end # function set_initial_environments!



function update_environment!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, x::Integer, y::Integer, direction::String="left")

    # update the environment
    if direction == "right"

        if y == 1
            tdvp.εl[x, y+1] = ψ.Ts[x, y] * tdvp.εu[x] * Ĥ.Ws[x, y] * tdvp.εd[x] * prime(dag(ψ.Ts[x, y]))
        else
            tdvp.εl[x, y+1] = ψ.Ts[x, y] * tdvp.εl[x, y] * Ĥ.Ws[x, y] * prime(dag(ψ.Ts[x, y]))
        end

    elseif direction == "left"

        tdvp.εr[x, y-1] = ψ.Ts[x, y] * tdvp.εr[x, y] * Ĥ.Ws[x, y] * prime(dag(ψ.Ts[x, y]))

    elseif direction == "down"

        y !== 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the backbone down."))
        tdvp.εu[x+1] = ψ.Ts[x, y] * tdvp.εu[x] * Ĥ.Ws[x, y] * tdvp.εr[x, y] * prime(dag(ψ.Ts[x, y]))

    elseif direction == "up"

        y !== 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the backbone up."))
        tdvp.εd[x-1] = ψ.Ts[x, y] * tdvp.εd[x] * Ĥ.Ws[x, y] * tdvp.εr[x, y] * prime(dag(ψ.Ts[x, y]))

    end

end



function single_site_time_evolution_on_arm!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, direction::String, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center

    if direction == "right"
        yc !== 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the arm right."))
        (yi, yf, dy, dyₗ, dyᵣ, a) = (yc, ψ.Ly, 1, 1, 0, 0)
    elseif direction == "left"
        yc !== ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site of the system when time-evolving the arm left."))
        (yi, yf, dy, dyₗ, dyᵣ, a) = (yc, 2, -1, 0, -1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end


    for y = yi:dy:yf

        if tdvp.params["verb_level"] > 1
            println("** single-site TDVP running on the arm $(direction)ward at (x, y) = ($x, $y).")
        end

        if y == 1
            Ĥₑ = (-1im * δt, tdvp.εu[x], Ĥ.Ws[x, y], tdvp.εd[x], tdvp.εr[x, y])
        else
            Ĥₑ = (-1im * δt, tdvp.εl[x, y], Ĥ.Ws[x, y], tdvp.εr[x, y])
        end

        ψ.Ts[x, y] .= local_time_evolution(Ĥₑ, ψ.Ts[x, y], tdvp.params["Ncut"], tdvp.params["verb_level"])

        # Time-evolution stops at the rightmost site of the system w/o the backward time-evolution
        if y == ψ.Ly && direction == "right"
            break
        end

        U, S, ψ.Ts[x, y] = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-10, righttags=tags(ψ.aux_y_idx[x, y+a]))
        update_environment!(tdvp, Ĥ, ψ, x, y, direction)

        Kₑ = (+1im * δt, tdvp.εl[x, y+dyₗ], tdvp.εr[x, y+dyᵣ])
        C = local_time_evolution(Kₑ, S * U, tdvp.params["Ncut"], tdvp.params["verb_level"])

        ψ.Ts[x, y+dy] *= C
        ψ.aux_y_idx[x, y+a] = commonind(S, ψ.Ts[x, y])

        if direction == "right"
            ψ.aux_y_idx[x, y+a] = dag(ψ.aux_y_idx[x, y+a])
        end # to make the direction of bond consistent

        network_update!(ψ, direction)

    end

end


function single_site_time_evolution_on_backbone!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, direction::String, δt::Union{AbstractFloat,Complex})

    (x, y) = ψ.canonical_center
    y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone."))

    if direction == "down"
        (dx, dxₑ, dxᵤ, a) = (1, 0, 1, 0)
    elseif direction == "up"
        (dx, dxₑ, dxᵤ, a) = (-1, -1, 0, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    if tdvp.params["verb_level"] > 1
        println("** single-site TDVP running on the backbone $(direction)ward at (x, y) = ($x, $y).")
    end

    Ĥₑ = (-1im * δt, tdvp.εu[x], Ĥ.Ws[x, 1], tdvp.εr[x, 1], tdvp.εd[x])
    ψ.Ts[x, 1] .= local_time_evolution(Ĥₑ, ψ.Ts[x, 1], tdvp.params["Ncut"], tdvp.params["verb_level"])
    U, S, ψ.Ts[x, 1] = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-10, righttags=tags(ψ.aux_x_idx[x+a]))

    update_environment!(tdvp, Ĥ, ψ, x, 1, direction)

    Kₑ = (+1im * δt, tdvp.εu[x+dxᵤ], tdvp.εd[x+dxₑ])
    C = local_time_evolution(Kₑ, S * U, tdvp.params["Ncut"], tdvp.params["verb_level"])

    ψ.Ts[x+dx, 1] *= C
    ψ.aux_x_idx[x+a] = commonind(S, ψ.Ts[x, 1])

    if direction == "down"
        ψ.aux_x_idx[x+a] = dag(ψ.aux_x_idx[x+a])
    end # to make the direction of bond consistent

    network_update!(ψ, direction)

end



function single_site_time_evolution_sweep_direction!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState; direction::String="down", half_step::Bool=false)

    if direction == "down"
        ψ.canonical_center != (1, ψ.Ly) && throw(ArgumentError(" The time-evolution downward should start at the rightmost & upmost bath site."))
        (xi, dx, xf) = (1, 1, ψ.Lx - 1)
    elseif direction == "up"
        ψ.canonical_center != (ψ.Lx, ψ.Ly) && throw(ArgumentError(" The time-evolution upward should start at the rightmost & bottommost bath site."))
        (xi, dx, xf) = (ψ.Lx, -1, 2)
    else
        throw(ArgumentError("Invalid direction"))
    end

    δt = tdvp.params["δt"] / (half_step ? 2.0 : 1.0)

    for x = xi:dx:xf

        if (direction == "down" && x > 1) || (direction == "up" && x < ψ.Lx)
            for y = 1:(ψ.Ly-1)
                canonical_center_move!(ψ, "right")
                update_environment!(tdvp, Ĥ, ψ, x, y, "right")
            end
        end

        single_site_time_evolution_on_arm!(tdvp, Ĥ, ψ, "left", δt)
        single_site_time_evolution_on_backbone!(tdvp, Ĥ, ψ, direction, δt)

    end

    single_site_time_evolution_on_arm!(tdvp, Ĥ, ψ, "right", δt)

end



function two_site_time_evolution_arm_right!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the arm right."))

    x = ψ.canonical_center[1]
    χʸ = tdvp.params["χʸ"]

    for y = 1:(ψ.Ly-1)

        if tdvp.params["verb_level"] > 1
            println("** 2-site TDVP running on the arm rightward at (x, y) = ($x, $y).")
        end

        # Forward time-evolution
        if y == 1

            Ĥₑ = (-1im * δt, tdvp.εu[x], tdvp.εd[x], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1], tdvp.εr[x, y+1])
            T = local_time_evolution(Ĥₑ, ψ.Ts[x, y] * ψ.Ts[x, y+1], tdvp.params["Ncut"], tdvp.params["verb_level"])
            V, S, ψ.Ts[x, y] = svd(T, (ψ.aux_y_idx[x, y+1], ψ.phys_idx[x, y+1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y]))

        else

            Ĥₑ = (-1im * δt, tdvp.εl[x, y], Ĥ.Ws[x, y], Ĥ.Ws[x, y+1], tdvp.εr[x, y+1])
            T = local_time_evolution(Ĥₑ, ψ.Ts[x, y] * ψ.Ts[x, y+1], tdvp.params["Ncut"], tdvp.params["verb_level"])
            ψ.Ts[x, y], S, V = svd(T, (ψ.aux_y_idx[x, y-1], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y]))

        end
        update_environment!(tdvp, Ĥ, ψ, x, y, "right")

        if y == ψ.Ly - 1
            # No backward time-evolution at the rightmost site of the system
            ψ.Ts[x, y+1] = S * V
        else
            # Backward time-evolution
            Kₑ = (+1im * δt, tdvp.εl[x, y+1], Ĥ.Ws[x, y+1], tdvp.εr[x, y+1])
            ψ.Ts[x, y+1] = local_time_evolution(Kₑ, S * V, tdvp.params["Ncut"], tdvp.params["verb_level"])
        end

        # Update index & network matrix
        ψ.aux_y_idx[x, y] = commonind(ψ.Ts[x, y], S)
        network_update!(ψ, "right")

    end

end


function two_site_time_evolution_arm_left!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    ψ.canonical_center[2] != ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site of the system when time-evolving the arm left."))

    x = ψ.canonical_center[1]
    χʸ = tdvp.params["χʸ"]

    for y = ψ.Ly:-1:2

        if tdvp.params["verb_level"] > 1
            println("** 2-site TDVP running on the arm leftward at (x, y) = ($x, $y).")
        end

        # Forward time-evolution
        if y == 2

            Ĥₑ = (-1im * δt, tdvp.εu[x], tdvp.εd[x], Ĥ.Ws[x, y-1], Ĥ.Ws[x, y], tdvp.εr[x, y])
            T = local_time_evolution(Ĥₑ, ψ.Ts[x, y-1] * ψ.Ts[x, y], tdvp.params["Ncut"], tdvp.params["verb_level"])
            ψ.Ts[x, y], S, U = svd(T, (ψ.aux_y_idx[x, y], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y-1]))

        else

            Ĥₑ = (-1im * δt, tdvp.εl[x, y-1], Ĥ.Ws[x, y-1], Ĥ.Ws[x, y], tdvp.εr[x, y])
            T = local_time_evolution(Ĥₑ, ψ.Ts[x, y-1] * ψ.Ts[x, y], tdvp.params["Ncut"], tdvp.params["verb_level"])
            U, S, ψ.Ts[x, y] = svd(T, (ψ.aux_y_idx[x, y-2], ψ.phys_idx[x, y-1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y-1]))

        end
        update_environment!(tdvp, Ĥ, ψ, x, y, "left")

        # Backward time-evolution
        if y == 2
            Kₑ = (+1im * δt, tdvp.εu[x], tdvp.εd[x], Ĥ.Ws[x, y-1], tdvp.εr[x, y-1])
        else
            Kₑ = (+1im * δt, tdvp.εl[x, y-1], Ĥ.Ws[x, y-1], tdvp.εr[x, y-1])
        end
        ψ.Ts[x, y-1] = local_time_evolution(Kₑ, U * S, tdvp.params["Ncut"], tdvp.params["verb_level"])

        # Update index & network matrix
        ψ.aux_y_idx[x, y-1] = commonind(S, ψ.Ts[x, y]) # 순서 주의
        network_update!(ψ, "left")

    end

end


function two_site_time_evolution_backbone_down!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    ψ.canonical_center[1] == ψ.Lx && throw(ArgumentError("xc should not be at the Lx when updating the backbone down."))
    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    x = ψ.canonical_center[1]
    χˣ = tdvp.params["χˣ"]

    if tdvp.params["verb_level"] > 1
        println("** 2-site TDVP running on the backbone downward at (x, y) = ($x, 1).")
    end

    Ĥₑ = (-1im * δt, tdvp.εu[x], tdvp.εr[x, 1], Ĥ.Ws[x, 1], Ĥ.Ws[x+1, 1], tdvp.εr[x+1, 1], tdvp.εd[x+1])
    T = local_time_evolution(Ĥₑ, ψ.Ts[x, 1] * ψ.Ts[x+1, 1], tdvp.params["Ncut"], tdvp.params["verb_level"])

    if x == 1
        ψ.Ts[x, 1], S, V = svd(T, (ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    else
        ψ.Ts[x, 1], S, V = svd(T, (ψ.aux_x_idx[x-1], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    end
    update_environment!(tdvp, Ĥ, ψ, x, 1, "down")

    # Backward time-evolution
    Kₑ = (+1im * δt, tdvp.εu[x+1], tdvp.εr[x+1, 1], Ĥ.Ws[x+1, 1], tdvp.εd[x+1])
    ψ.Ts[x+1, 1] = local_time_evolution(Kₑ, S * V, tdvp.params["Ncut"], tdvp.params["verb_level"])

    # Update index & network matrix
    ψ.aux_x_idx[x] = commonind(ψ.Ts[x, 1], S)
    network_update!(ψ, "down")

end


function two_site_time_evolution_backbone_up!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    ψ.canonical_center[1] == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
    ψ.canonical_center[2] != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    x = ψ.canonical_center[1]
    χˣ = tdvp.params["χˣ"]

    if tdvp.params["verb_level"] > 1
        println("** 2-site TDVP running on the backbone upward at (x, y) = ($x, 1).")
    end

    Ĥₑ = (-1im * δt, tdvp.εu[x-1], tdvp.εr[x-1, 1], Ĥ.Ws[x-1, 1], Ĥ.Ws[x, 1], tdvp.εr[x, 1], tdvp.εd[x])
    T = local_time_evolution(Ĥₑ, ψ.Ts[x-1, 1] * ψ.Ts[x, 1], tdvp.params["Ncut"], tdvp.params["verb_level"])

    if x == ψ.Lx
        U, S, V = svd(T, (ψ.aux_x_idx[x-2], ψ.aux_y_idx[x-1, 1], ψ.phys_idx[x-1, 1]); cutoff=1e-10, maxdim=χˣ, righttags=tags(ψ.aux_x_idx[x-1]))
    else
        V, S, U = svd(T, (ψ.aux_x_idx[x], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x-1]))
    end
    ψ.Ts[x, 1] = V
    update_environment!(tdvp, Ĥ, ψ, x, 1, "up")

    # Backward time-evolution
    Kₑ = (+1im * δt, tdvp.εu[x-1], tdvp.εr[x-1, 1], Ĥ.Ws[x-1, 1], tdvp.εd[x-1])
    ψ.Ts[x-1, 1] = local_time_evolution(Kₑ, S * U, tdvp.params["Ncut"], tdvp.params["verb_level"])

    # Update index & network matrix
    ψ.aux_x_idx[x-1] = commonind(S, ψ.Ts[x, 1]) # 순서 주의
    network_update!(ψ, "up")

end


function two_site_time_evolution_sweep_direction!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState; direction::String="down", half_step::Bool=false)

    if direction == "down"
        ψ.canonical_center != (1, ψ.Ly) && throw(ArgumentError(" The time-evolution downward should start at the rightmost & upmost bath site."))
        (xi, dx, xf) = (1, 1, ψ.Lx - 1)
    elseif direction == "up"
        ψ.canonical_center != (ψ.Lx, ψ.Ly) && throw(ArgumentError(" The time-evolution upward should start at the rightmost & bottommost bath site."))
        (xi, dx, xf) = (ψ.Lx, -1, 2)
    else
        throw(ArgumentError("Invalid direction"))
    end

    δt = tdvp.params["δt"] / (half_step ? 2.0 : 1.0)

    for x = xi:dx:xf

        if (direction == "down" && x > 1) || (direction == "up" && x < ψ.Lx)
            for y = 1:(ψ.Ly-1)
                canonical_center_move!(ψ, "right")
                update_environment!(tdvp, Ĥ, ψ, x, y, "right")
            end
        end

        two_site_time_evolution_arm_left!(tdvp, Ĥ, ψ, δt)

        if direction == "down"
            two_site_time_evolution_backbone_down!(tdvp, Ĥ, ψ, δt)
        else
            two_site_time_evolution_backbone_up!(tdvp, Ĥ, ψ, δt)
        end

    end

    two_site_time_evolution_arm_right!(tdvp, Ĥ, ψ, δt)

end


function local_time_evolution(Ĥₑ::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, T₀::ITensor, Ncut::Integer, verb_level::Integer)

    if verb_level > 2
        return krylov_expm(Ĥₑ, T₀; max_iter=Ncut, tol=1.0E-6, verbose=true)
        # return exp_taylor_sum(Ĥₑ, T₀, 10)
    else
        return krylov_expm(Ĥₑ, T₀; max_iter=Ncut, tol=1.0E-6, verbose=false)
        # return exp_taylor_sum(Ĥₑ, T₀, 10)
    end

end



