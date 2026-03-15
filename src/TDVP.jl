"""
    TDVPParams

Typed parameter struct for TDVP algorithm with type stability.

Fields:
- `χˣ::Int`: Maximum bond dimension along the x-direction
- `χʸ::Int`: Maximum bond dimension along the y-direction
- `method::Symbol`: Evolution method (:single_site or :two_site)
- `δt::ComplexF64`: Time step (can be real or complex)
- `Ncut::Int`: Krylov subspace cutoff
- `subspace_expansion::Symbol`: `:none` or `:cbe` (CBE before forward evolution in single-site)
- `δ::Float64`: CBE expansion ratio
- `verb_level::Int`: Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)

Reference: SciPost Phys. 8, 024 (2020)
"""
struct TDVPParams
    χˣ::Int
    χʸ::Int
    method::Symbol
    δt::ComplexF64
    Ncut::Int
    subspace_expansion::Symbol
    δ::Float64
    verb_level::Int
    energy_shift::Float64
end


"""
    TDVPParams(; χˣ, χʸ, method, δt, Ncut, subspace_expansion, δ, verb_level)

Constructor for TDVPParams with keyword arguments.
"""
function TDVPParams(; χˣ::Int, χʸ::Int, method::Symbol=:single_site,
    δt::Union{Float64,ComplexF64}=0.1, Ncut::Int=10,
    subspace_expansion::Symbol=:none, δ::Float64=0.1, verb_level::Int=0,
    energy_shift::Float64=0.0)
    return TDVPParams(χˣ, χʸ, method, ComplexF64(δt), Ncut, subspace_expansion, δ, verb_level, energy_shift)
end


"""
    TDVPParams(d::Dict{String,Any})

Construct TDVPParams from a dictionary for backward compatibility.
"""
function TDVPParams(d::Dict{String,Any})
    method_str = get(d, "method", "single-site")
    method = method_str == "single-site" ? :single_site :
             method_str == "two-site" ? :two_site : Symbol(replace(method_str, "-" => "_"))

    se_str = get(d, "subspace_expansion", "none")
    se = se_str == "cbe" ? :cbe : :none

    return TDVPParams(
        get(d, "χˣ", 10),
        get(d, "χʸ", 10),
        method,
        ComplexF64(get(d, "δt", 0.1)),
        get(d, "Ncut", 10),
        se,
        get(d, "δ", 0.1),
        get(d, "verb_level", 0),
        get(d, "energy_shift", 0.0),
    )
end


"""
    show(io::IO, p::TDVPParams)

Pretty-print TDVP parameters.
"""
function Base.show(io::IO, p::TDVPParams)
    println(io, "* TDVP Parameters:")
    for fname in fieldnames(TDVPParams)
        println(io, "*  ", fname, ": ", getfield(p, fname))
    end
end


"""
    TDVP

Time-evolving a fork tensor network state using the TDVP algorithm.

Fields:
- `params::TDVPParams`: TDVP parameters
- `envs::FTNEnvironments`: Boundary environment tensors (shared struct)
- `time::ComplexF64`: Current evolution time
- `env_set::Bool`: Flag indicating whether initial environments are set

Reference: SciPost Phys. 8, 024 (2020)
"""
mutable struct TDVP

    params::TDVPParams
    envs::FTNEnvironments
    time::ComplexF64
    env_set::Bool


    """
        TDVP(params::TDVPParams)

    Construct a TDVP instance with initialized parameters.
    """
    function TDVP(params::TDVPParams)

        show(stdout, params)

        tdvp = new(
            params,
            FTNEnvironments(),
            0.0,
            false
        )

        return tdvp
    end

    """
        TDVP(params::Dict{String,Any})

    Backward-compatible constructor from dictionary.
    """
    function TDVP(params::Dict{String,Any})
        return TDVP(TDVPParams(params))
    end

end





"""
    run_tdvp!(tdvp, H, ψ, max_step)

Execute TDVP time evolution for the specified number of steps.
"""
function run_tdvp!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, max_step::Integer)

    if !tdvp.env_set
        set_initial_environments!(tdvp, H, ψ)
    end

    for i = 1:max_step

        ψ.max_S = 0.0  # reset per step
        if tdvp.params.verb_level > 0
            if isa(tdvp.time, Real) || abs(imag(tdvp.time)) < 1e-15
                @printf("*  step %d, t=%.2f", i, real(tdvp.time))
            else
                @printf("*  step %d, t=%.2f%+.2fi", i, real(tdvp.time), imag(tdvp.time))
            end
        end

        if tdvp.params.method == :single_site

            single_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:down, half_step=true)
            single_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:up, half_step=true)

        elseif tdvp.params.method == :two_site

            two_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:down, half_step=true)
            two_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:up, half_step=true)

        elseif tdvp.params.method == :hybrid

            hybrid_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:down, half_step=true)
            hybrid_time_evolution_sweep_direction!(tdvp, H, ψ; direction=:up, half_step=true)

        else
            throw(ArgumentError("Invalid method"))
        end
        tdvp.time += tdvp.params.δt

        if tdvp.params.verb_level > 0
            @printf(", max_S: %.4f\n", ψ.max_S)
        end

    end

end


"""
    set_initial_environments!(tdvp, H, ψ)

Initialize all boundary environments for the TDVP algorithm.
"""
function set_initial_environments!(tdvp::TDVP, Ĥ::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    allocate!(tdvp.envs, ψ.Lx, ψ.Ly)

    canonical_form!(ψ, 1, ψ.Ly)

    tdvp.envs.er[1, ψ.Ly] = ITensor(1.0)
    for x = 2:ψ.Lx
        tdvp.envs.er[x, ψ.Ly] = ITensor(1.0)
        for y = ψ.Ly:-1:2
            update_environment!(tdvp.envs, Ĥ, ψ, x, y, :left)
        end
    end

    tdvp.envs.ed[ψ.Lx] = ITensor(1.0)
    for x = ψ.Lx:-1:2
        update_environment!(tdvp.envs, Ĥ, ψ, x, 1, :up)
    end

    tdvp.envs.eu[1] = ITensor(1.0)
    for y = 1:(ψ.Ly-1)
        update_environment!(tdvp.envs, Ĥ, ψ, 1, y, :right)
    end

    tdvp.env_set = true

end


"""
    single_site_time_evolution_on_arm!(tdvp, H, ψ, direction, δt)

Perform single-site TDVP time evolution along the arm in specified direction.
"""
function single_site_time_evolution_on_arm!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, direction::Symbol, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center

    if direction == :right
        yc !== 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the arm right."))
        (yi, yf, dy, dyₗ, dyᵣ, a) = (yc, ψ.Ly, 1, 1, 0, 0)
    elseif direction == :left
        yc !== ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site of the system when time-evolving the arm left."))
        (yi, yf, dy, dyₗ, dyᵣ, a) = (yc, 2, -1, 0, -1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level
    do_cbe = (tdvp.params.subspace_expansion == :cbe)
    χʸ = tdvp.params.χʸ

    for y = yi:dy:yf

        if verb_level > 1
            println("** single-site TDVP running on the arm $(direction)ward at (x, y) = ($x, $y).")
        end

        # CBE: expand bond before forward evolution (skip at boundary where no move follows)
        can_expand = !(direction == :right && y == ψ.Ly)
        if do_cbe && can_expand
            controlled_bond_expansion!(tdvp.envs, ψ, H, direction, tdvp.params.δ)
        end

        if y == 1
            Hₑ = (-1im * δt, tdvp.envs.eu[x], H.Ws[x, y], tdvp.envs.ed[x], tdvp.envs.er[x, y])
        else
            Hₑ = (-1im * δt, tdvp.envs.el[x, y], H.Ws[x, y], tdvp.envs.er[x, y])
        end

        ψ.Ts[x, y] .= local_time_evolution(Hₑ, ψ.Ts[x, y], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

        if y == ψ.Ly && direction == :right
            break
        end

        U, S, ψ.Ts[x, y] = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-16, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y+a]))
        ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
        update_environment!(tdvp.envs, H, ψ, x, y, direction)

        Kₑ = (+1im * δt, tdvp.envs.el[x, y+dyₗ], tdvp.envs.er[x, y+dyᵣ])
        C = local_time_evolution(Kₑ, S * U, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

        ψ.Ts[x, y+dy] *= C
        ψ.aux_y_idx[x, y+a] = commonind(S, ψ.Ts[x, y])

        if direction == :right
            ψ.aux_y_idx[x, y+a] = dag(ψ.aux_y_idx[x, y+a])
        end

        network_update!(ψ, direction)

    end

end


"""
    single_site_time_evolution_on_backbone!(tdvp, H, ψ, direction, δt)

Perform single-site TDVP time evolution along the backbone in specified direction.
"""
function single_site_time_evolution_on_backbone!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, direction::Symbol, δt::Union{AbstractFloat,Complex})

    (x, y) = ψ.canonical_center
    y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone."))

    if direction == :down
        (dx, dxₑ, dxᵤ, a) = (1, 0, 1, 0)
    elseif direction == :up
        (dx, dxₑ, dxᵤ, a) = (-1, -1, 0, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level
    do_cbe = (tdvp.params.subspace_expansion == :cbe)
    χˣ = tdvp.params.χˣ

    if verb_level > 1
        println("** single-site TDVP running on the backbone $(direction)ward at (x, y) = ($x, $y).")
    end

    # CBE: expand bond before forward evolution
    if do_cbe
        controlled_bond_expansion!(tdvp.envs, ψ, H, direction, tdvp.params.δ)
    end

    Hₑ = (-1im * δt, tdvp.envs.eu[x], H.Ws[x, 1], tdvp.envs.er[x, 1], tdvp.envs.ed[x])
    ψ.Ts[x, 1] .= local_time_evolution(Hₑ, ψ.Ts[x, 1], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    U, S, ψ.Ts[x, 1] = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-16, maxdim=χˣ, righttags=tags(ψ.aux_x_idx[x+a]))
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))

    update_environment!(tdvp.envs, H, ψ, x, 1, direction)

    Kₑ = (+1im * δt, tdvp.envs.eu[x+dxᵤ], tdvp.envs.ed[x+dxₑ])
    C = local_time_evolution(Kₑ, S * U, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.Ts[x+dx, 1] *= C
    ψ.aux_x_idx[x+a] = commonind(S, ψ.Ts[x, 1])

    if direction == :down
        ψ.aux_x_idx[x+a] = dag(ψ.aux_x_idx[x+a])
    end

    network_update!(ψ, direction)

end


"""
    single_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction, half_step)

Execute single-site TDVP sweep in specified direction over the entire network.
"""
function single_site_time_evolution_sweep_direction!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState; direction::Symbol=:down, half_step::Bool=false)

    if direction == :down
        ψ.canonical_center != (1, ψ.Ly) && throw(ArgumentError(" The time-evolution downward should start at the rightmost & upmost bath site."))
        (xi, dx, xf) = (1, 1, ψ.Lx - 1)
    elseif direction == :up
        ψ.canonical_center != (ψ.Lx, ψ.Ly) && throw(ArgumentError(" The time-evolution upward should start at the rightmost & bottommost bath site."))
        (xi, dx, xf) = (ψ.Lx, -1, 2)
    else
        throw(ArgumentError("Invalid direction"))
    end

    δt = tdvp.params.δt / (half_step ? 2.0 : 1.0)

    for x = xi:dx:xf

        if (direction == :down && x > 1) || (direction == :up && x < ψ.Lx)
            for y = 1:(ψ.Ly-1)
                canonical_center_move!(ψ, :right)
                update_environment!(tdvp.envs, H, ψ, x, y, :right)
            end
        end

        single_site_time_evolution_on_arm!(tdvp, H, ψ, :left, δt)
        single_site_time_evolution_on_backbone!(tdvp, H, ψ, direction, δt)

    end

    single_site_time_evolution_on_arm!(tdvp, H, ψ, :right, δt)

end


"""
    two_site_step_arm_right!(tdvp, H, ψ, x, y, δt; skip_backward=false)

Local two-site TDVP step on the arm moving rightward at position (x, y).
Merges sites (x,y) and (x,y+1), time-evolves, SVD-splits, and optionally
performs backward evolution on the remaining site tensor.

- `y=1`: backbone-arm boundary (environment uses eu, ed)
- `y≥2`: pure arm bond (environment uses el)
- `skip_backward=true`: skip backward evolution (used at the last bond y=Ly-1)
"""
function two_site_step_arm_right!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState,
    x::Int, y::Int, δt::Union{AbstractFloat,Complex}; skip_backward::Bool=false)

    χʸ = tdvp.params.χʸ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    if verb_level > 1
        println("** 2-site TDVP running on the arm rightward at (x, y) = ($x, $y).")
    end

    # Forward evolution: merge two sites and time-evolve
    if y == 1
        Hₑ = (-1im * δt, tdvp.envs.eu[x], tdvp.envs.ed[x], H.Ws[x, y], H.Ws[x, y+1], tdvp.envs.er[x, y+1])
    else
        Hₑ = (-1im * δt, tdvp.envs.el[x, y], H.Ws[x, y], H.Ws[x, y+1], tdvp.envs.er[x, y+1])
    end
    T = local_time_evolution(Hₑ, ψ.Ts[x, y] * ψ.Ts[x, y+1], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    # SVD: split the two-site tensor
    if y == 1
        V, S, ψ.Ts[x, y] = svd(T, (ψ.aux_y_idx[x, y+1], ψ.phys_idx[x, y+1]); cutoff=1e-16, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y]))
    else
        ψ.Ts[x, y], S, V = svd(T, (ψ.aux_y_idx[x, y-1], ψ.phys_idx[x, y]); cutoff=1e-16, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    update_environment!(tdvp.envs, H, ψ, x, y, :right)

    # Backward evolution on the remaining site tensor (or just assign if last bond)
    if skip_backward
        ψ.Ts[x, y+1] = S * V
    else
        Kₑ = (+1im * δt, tdvp.envs.el[x, y+1], H.Ws[x, y+1], tdvp.envs.er[x, y+1])
        ψ.Ts[x, y+1] = local_time_evolution(Kₑ, S * V, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)
    end

    ψ.aux_y_idx[x, y] = commonind(ψ.Ts[x, y], S)
    network_update!(ψ, :right)

end


"""
    two_site_step_arm_left!(tdvp, H, ψ, x, y, δt)

Local two-site TDVP step on the arm moving leftward at position (x, y).
Merges sites (x,y-1) and (x,y), time-evolves, SVD-splits, and performs
backward evolution on the remaining site tensor.

- `y=2`: backbone-arm boundary (environment uses eu, ed for backward evolution)
- `y≥3`: pure arm bond (environment uses el for backward evolution)
"""
function two_site_step_arm_left!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState,
    x::Int, y::Int, δt::Union{AbstractFloat,Complex})

    χʸ = tdvp.params.χʸ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    if verb_level > 1
        println("** 2-site TDVP running on the arm leftward at (x, y) = ($x, $y).")
    end

    # Forward evolution: merge two sites and time-evolve
    if y == 2
        Hₑ = (-1im * δt, tdvp.envs.eu[x], tdvp.envs.ed[x], H.Ws[x, y-1], H.Ws[x, y], tdvp.envs.er[x, y])
    else
        Hₑ = (-1im * δt, tdvp.envs.el[x, y-1], H.Ws[x, y-1], H.Ws[x, y], tdvp.envs.er[x, y])
    end
    T = local_time_evolution(Hₑ, ψ.Ts[x, y-1] * ψ.Ts[x, y], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    # SVD: split the two-site tensor
    if y == 2
        ψ.Ts[x, y], S, U = svd(T, (ψ.aux_y_idx[x, y], ψ.phys_idx[x, y]); cutoff=1e-16, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y-1]))
    else
        U, S, ψ.Ts[x, y] = svd(T, (ψ.aux_y_idx[x, y-2], ψ.phys_idx[x, y-1]); cutoff=1e-16, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y-1]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    update_environment!(tdvp.envs, H, ψ, x, y, :left)

    # Backward evolution on the remaining site tensor
    if y == 2
        Kₑ = (+1im * δt, tdvp.envs.eu[x], tdvp.envs.ed[x], H.Ws[x, y-1], tdvp.envs.er[x, y-1])
    else
        Kₑ = (+1im * δt, tdvp.envs.el[x, y-1], H.Ws[x, y-1], tdvp.envs.er[x, y-1])
    end
    ψ.Ts[x, y-1] = local_time_evolution(Kₑ, U * S, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.aux_y_idx[x, y-1] = commonind(S, ψ.Ts[x, y])
    network_update!(ψ, :left)

end


"""
    two_site_time_evolution_arm_right!(tdvp, H, ψ, δt)

Perform two-site TDVP time evolution along the arm moving rightward.
"""
function two_site_time_evolution_arm_right!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center
    yc != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when time-evolving the arm right."))

    for y = 1:(ψ.Ly-1)
        two_site_step_arm_right!(tdvp, H, ψ, x, y, δt; skip_backward=(y == ψ.Ly - 1))
    end

end


"""
    two_site_time_evolution_arm_left!(tdvp, H, ψ, δt)

Perform two-site TDVP time evolution along the arm moving leftward.
"""
function two_site_time_evolution_arm_left!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center
    yc != ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site of the system when time-evolving the arm left."))

    for y = ψ.Ly:-1:2
        two_site_step_arm_left!(tdvp, H, ψ, x, y, δt)
    end

end


"""
    two_site_time_evolution_backbone_down!(tdvp, H, ψ, δt)

Perform two-site TDVP time evolution along the backbone moving downward.
"""
function two_site_time_evolution_backbone_down!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, y) = ψ.canonical_center
    x == ψ.Lx && throw(ArgumentError("xc should not be at the Lx when updating the backbone down."))
    y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    χˣ = tdvp.params.χˣ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    if verb_level > 1
        println("** 2-site TDVP running on the backbone downward at (x, y) = ($x, 1).")
    end

    Hₑ = (-1im * δt, tdvp.envs.eu[x], tdvp.envs.er[x, 1], H.Ws[x, 1], H.Ws[x+1, 1], tdvp.envs.er[x+1, 1], tdvp.envs.ed[x+1])
    T = local_time_evolution(Hₑ, ψ.Ts[x, 1] * ψ.Ts[x+1, 1], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    if x == 1
        ψ.Ts[x, 1], S, V = svd(T, (ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-16, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    else
        ψ.Ts[x, 1], S, V = svd(T, (ψ.aux_x_idx[x-1], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-16, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    update_environment!(tdvp.envs, H, ψ, x, 1, :down)

    Kₑ = (+1im * δt, tdvp.envs.eu[x+1], tdvp.envs.er[x+1, 1], H.Ws[x+1, 1], tdvp.envs.ed[x+1])
    ψ.Ts[x+1, 1] = local_time_evolution(Kₑ, S * V, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.aux_x_idx[x] = commonind(ψ.Ts[x, 1], S)
    network_update!(ψ, :down)

end


"""
    two_site_time_evolution_backbone_up!(tdvp, H, ψ, δt)

Perform two-site TDVP time evolution along the backbone moving upward.
"""
function two_site_time_evolution_backbone_up!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, y) = ψ.canonical_center
    x == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
    y != 1 && throw(ArgumentError("Canonical center should be at the leftmost site of the system when updating the backbone down."))

    χˣ = tdvp.params.χˣ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    if verb_level > 1
        println("** 2-site TDVP running on the backbone upward at (x, y) = ($x, 1).")
    end

    Hₑ = (-1im * δt, tdvp.envs.eu[x-1], tdvp.envs.er[x-1, 1], H.Ws[x-1, 1], H.Ws[x, 1], tdvp.envs.er[x, 1], tdvp.envs.ed[x])
    T = local_time_evolution(Hₑ, ψ.Ts[x-1, 1] * ψ.Ts[x, 1], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    if x == ψ.Lx
        U, S, V = svd(T, (ψ.aux_x_idx[x-2], ψ.aux_y_idx[x-1, 1], ψ.phys_idx[x-1, 1]); cutoff=1e-16, maxdim=χˣ, righttags=tags(ψ.aux_x_idx[x-1]))
    else
        V, S, U = svd(T, (ψ.aux_x_idx[x], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-16, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x-1]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    ψ.Ts[x, 1] = V
    update_environment!(tdvp.envs, H, ψ, x, 1, :up)

    Kₑ = (+1im * δt, tdvp.envs.eu[x-1], tdvp.envs.er[x-1, 1], H.Ws[x-1, 1], tdvp.envs.ed[x-1])
    ψ.Ts[x-1, 1] = local_time_evolution(Kₑ, S * U, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.aux_x_idx[x-1] = commonind(S, ψ.Ts[x, 1])
    network_update!(ψ, :up)

end


"""
    two_site_time_evolution_sweep_direction!(tdvp, H, ψ; direction, half_step)

Execute two-site TDVP sweep in specified direction over the entire network.
"""
function two_site_time_evolution_sweep_direction!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState; direction::Symbol=:down, half_step::Bool=false)

    if direction == :down
        ψ.canonical_center != (1, ψ.Ly) && throw(ArgumentError(" The time-evolution downward should start at the rightmost & upmost bath site."))
        (xi, dx, xf) = (1, 1, ψ.Lx - 1)
    elseif direction == :up
        ψ.canonical_center != (ψ.Lx, ψ.Ly) && throw(ArgumentError(" The time-evolution upward should start at the rightmost & bottommost bath site."))
        (xi, dx, xf) = (ψ.Lx, -1, 2)
    else
        throw(ArgumentError("Invalid direction"))
    end

    δt = tdvp.params.δt / (half_step ? 2.0 : 1.0)

    for x = xi:dx:xf

        if (direction == :down && x > 1) || (direction == :up && x < ψ.Lx)
            for y = 1:(ψ.Ly-1)
                canonical_center_move!(ψ, :right)
                update_environment!(tdvp.envs, H, ψ, x, y, :right)
            end
        end

        two_site_time_evolution_arm_left!(tdvp, H, ψ, δt)

        if direction == :down
            two_site_time_evolution_backbone_down!(tdvp, H, ψ, δt)
        else
            two_site_time_evolution_backbone_up!(tdvp, H, ψ, δt)
        end

    end

    two_site_time_evolution_arm_right!(tdvp, H, ψ, δt)

end


"""
    hybrid_time_evolution_arm_left!(tdvp, H, ψ, δt)

Hybrid arm sweep leftward: two-site for pure arm bonds (y=Ly → 3),
then single-site + CBE for the backbone-arm boundary (y=2 → 1).
"""
function hybrid_time_evolution_arm_left!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center
    yc != ψ.Ly && throw(ArgumentError("Canonical center should be at the rightmost site when sweeping arm left."))

    χʸ = tdvp.params.χʸ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    # Phase 1: Two-site updates for pure arm bonds (y = Ly down to 3)
    for y = ψ.Ly:-1:3
        two_site_step_arm_left!(tdvp, H, ψ, x, y, δt)
    end

    # Phase 2: Single-site + CBE for backbone-arm boundary (y=2 → y=1)
    # Now canonical center is at (x, 2)
    if verb_level > 1
        println("** hybrid (1-site+CBE) arm leftward at (x, y) = ($x, 2).")
    end

    controlled_bond_expansion!(tdvp.envs, ψ, H, :left, tdvp.params.δ)

    Hₑ = (-1im * δt, tdvp.envs.el[x, 2], H.Ws[x, 2], tdvp.envs.er[x, 2])
    ψ.Ts[x, 2] .= local_time_evolution(Hₑ, ψ.Ts[x, 2], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    U, S, ψ.Ts[x, 2] = svd(ψ.Ts[x, 2], ψ.aux_y_idx[x, 1]; cutoff=1e-16, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, 1]))
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    update_environment!(tdvp.envs, H, ψ, x, 2, :left)

    Kₑ = (+1im * δt, tdvp.envs.el[x, 2], tdvp.envs.er[x, 1])
    C = local_time_evolution(Kₑ, S * U, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.Ts[x, 1] *= C
    ψ.aux_y_idx[x, 1] = commonind(S, ψ.Ts[x, 2])
    network_update!(ψ, :left)

end


"""
    hybrid_time_evolution_arm_right!(tdvp, H, ψ, δt)

Hybrid arm sweep rightward: single-site + CBE for the backbone-arm boundary (y=1),
then two-site for pure arm bonds (y=2 → Ly).
"""
function hybrid_time_evolution_arm_right!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, δt::Union{AbstractFloat,Complex})

    (x, yc) = ψ.canonical_center
    yc != 1 && throw(ArgumentError("Canonical center should be at y=1 when sweeping arm right."))

    χʸ = tdvp.params.χʸ
    Ncut = tdvp.params.Ncut
    verb_level = tdvp.params.verb_level

    # Phase 1: Single-site + CBE for backbone-arm boundary (y=1)
    if verb_level > 1
        println("** hybrid (1-site+CBE) arm rightward at (x, y) = ($x, 1).")
    end

    controlled_bond_expansion!(tdvp.envs, ψ, H, :right, tdvp.params.δ)

    Hₑ = (-1im * δt, tdvp.envs.eu[x], H.Ws[x, 1], tdvp.envs.ed[x], tdvp.envs.er[x, 1])
    ψ.Ts[x, 1] .= local_time_evolution(Hₑ, ψ.Ts[x, 1], Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    U, S, ψ.Ts[x, 1] = svd(ψ.Ts[x, 1], ψ.aux_y_idx[x, 1]; cutoff=1e-16, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, 1]))
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    update_environment!(tdvp.envs, H, ψ, x, 1, :right)

    Kₑ = (+1im * δt, tdvp.envs.el[x, 2], tdvp.envs.er[x, 1])
    C = local_time_evolution(Kₑ, S * U, Ncut, verb_level; energy_shift=tdvp.params.energy_shift)

    ψ.Ts[x, 2] *= C
    ψ.aux_y_idx[x, 1] = commonind(S, ψ.Ts[x, 1])
    ψ.aux_y_idx[x, 1] = dag(ψ.aux_y_idx[x, 1])
    network_update!(ψ, :right)

    # Phase 2: Two-site updates for pure arm bonds (y=2 → Ly)
    for y = 2:(ψ.Ly-1)
        two_site_step_arm_right!(tdvp, H, ψ, x, y, δt; skip_backward=(y == ψ.Ly - 1))
    end

end


"""
    hybrid_time_evolution_sweep_direction!(tdvp, H, ψ; direction, half_step)

Hybrid sweep: two-site on pure arm bonds, single-site + CBE on backbone and backbone-arm boundary.
"""
function hybrid_time_evolution_sweep_direction!(tdvp::TDVP, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState; direction::Symbol=:down, half_step::Bool=false)

    if direction == :down
        ψ.canonical_center != (1, ψ.Ly) && throw(ArgumentError("The time-evolution downward should start at the rightmost & upmost bath site."))
        (xi, dx, xf) = (1, 1, ψ.Lx - 1)
    elseif direction == :up
        ψ.canonical_center != (ψ.Lx, ψ.Ly) && throw(ArgumentError("The time-evolution upward should start at the rightmost & bottommost bath site."))
        (xi, dx, xf) = (ψ.Lx, -1, 2)
    else
        throw(ArgumentError("Invalid direction"))
    end

    δt = tdvp.params.δt / (half_step ? 2.0 : 1.0)

    for x = xi:dx:xf

        if (direction == :down && x > 1) || (direction == :up && x < ψ.Lx)
            for y = 1:(ψ.Ly-1)
                canonical_center_move!(ψ, :right)
                update_environment!(tdvp.envs, H, ψ, x, y, :right)
            end
        end

        hybrid_time_evolution_arm_left!(tdvp, H, ψ, δt)

        # Backbone: single-site + CBE
        single_site_time_evolution_on_backbone!(tdvp, H, ψ, direction, δt)

    end

    hybrid_time_evolution_arm_right!(tdvp, H, ψ, δt)

end


"""
    local_time_evolution(Hₑ, T₀, Ncut, verb_level; energy_shift=0.0)

Compute local time evolution using Krylov exponential method.
"""
function local_time_evolution(Hₑ::Tuple{Vararg{Union{ITensor,Complex,AbstractFloat}}}, T₀::ITensor, Ncut::Integer, verb_level::Integer;
                              energy_shift::Float64=0.0)

    gen_shift = -ComplexF64(Hₑ[1]) * energy_shift
    return krylov_expm(Hₑ, T₀; max_iter=Ncut, tol=1.0e-6, shift=gen_shift, verbose=(verb_level > 2))

end
