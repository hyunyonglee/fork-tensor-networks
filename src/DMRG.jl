"""
    DMRGParams

Typed parameter struct for DMRG, replacing `Dict{String,Any}` for type stability.

# Fields
- `χˣ::Int`: Maximum bond dimension along the backbone (x-direction).
- `χʸ::Int`: Maximum bond dimension along the arms (y-direction).
- `χˣ_schedule::Vector{Pair{Int,Int}}`: Per-sweep schedule for χˣ, e.g., `[5 => 10, 10 => 20]`
  means sweep 1–5 uses χˣ=10, sweep 6–10 uses χˣ=20. Empty means constant `χˣ`.
- `χʸ_schedule::Vector{Pair{Int,Int}}`: Per-sweep schedule for χʸ (same format).
- `max_iter::Int`: Maximum number of DMRG sweeps.
- `convergence_tol::Float64`: Convergence tolerance for relative energy change.
- `method::Symbol`: `:single_site`, `:two_site`, or `:hybrid`.
- `subspace_expansion::Symbol`: `:none`, `:cbe`, `:subspace_3s`, or `:cbe_3s`.
- `α::Float64`: Mixing parameter for 3S subspace expansion.
- `α_decay::Float64`: Per-sweep decay rate for `α`.
- `δ::Float64`: CBE expansion ratio.
- `max_3s_sweeps::Int`: Maximum number of sweeps to apply 3S expansion (0 = unlimited).
- `verbose::Bool`: Enable verbose output.
"""
mutable struct DMRGParams
    χˣ::Int
    χʸ::Int
    χˣ_schedule::Vector{Pair{Int,Int}}
    χʸ_schedule::Vector{Pair{Int,Int}}
    max_iter::Int
    convergence_tol::Float64
    method::Symbol
    subspace_expansion::Symbol
    α::Float64
    α_decay::Float64
    δ::Float64
    max_3s_sweeps::Int
    verbose::Bool
end


"""Keyword constructor with default values."""
function DMRGParams(; χˣ::Int, χʸ::Int,
    χˣ_schedule::Vector{Pair{Int,Int}}=Pair{Int,Int}[],
    χʸ_schedule::Vector{Pair{Int,Int}}=Pair{Int,Int}[],
    max_iter::Int=100, convergence_tol::Float64=1e-8,
    method::Symbol=:single_site, subspace_expansion::Symbol=:none,
    α::Float64=1e-3, α_decay::Float64=1.0, δ::Float64=0.1, max_3s_sweeps::Int=0,
    verbose::Bool=false)
    return DMRGParams(χˣ, χʸ, χˣ_schedule, χʸ_schedule, max_iter, convergence_tol, method, subspace_expansion, α, α_decay, δ, max_3s_sweeps, verbose)
end


"""Construct `DMRGParams` from a `Dict{String,Any}` for backward compatibility."""
function DMRGParams(d::Dict{String,Any})
    method_str = get(d, "method", "single-site")
    method = method_str == "single-site" ? :single_site :
             method_str == "two-site" ? :two_site : Symbol(replace(method_str, "-" => "_"))

    se_str = get(d, "subspace_expansion", "none")
    se = se_str == "3s" ? :subspace_3s :
         se_str == "cbe" ? :cbe :
         se_str == "cbe+3s" ? :cbe_3s : :none

    return DMRGParams(
        get(d, "χˣ", 10),
        get(d, "χʸ", 10),
        get(d, "χˣ_schedule", Pair{Int,Int}[]),
        get(d, "χʸ_schedule", Pair{Int,Int}[]),
        get(d, "max_iter", 100),
        get(d, "convergence_tol", 1e-8),
        method,
        se,
        get(d, "α", 1e-3),
        get(d, "α_decay", 1.0),
        get(d, "δ", 0.1),
        get(d, "max_3s_sweeps", 0),
        get(d, "verbose", false),
    )
end


function Base.show(io::IO, p::DMRGParams)
    println(io, "* DMRG Parameters:")
    for fname in fieldnames(DMRGParams)
        println(io, "*  ", fname, ": ", getfield(p, fname))
    end
end


"""
    resolve_χ(schedule, default, sweep) -> Int

Return the bond dimension for the given `sweep` number. If `schedule` is empty,
return `default`. Otherwise, find the first entry `(max_sweep => χ)` where
`sweep <= max_sweep`.
"""
function resolve_χ(schedule::Vector{Pair{Int,Int}}, default::Int, sweep::Int)
    isempty(schedule) && return default
    for (max_sweep, χ) in schedule
        sweep <= max_sweep && return χ
    end
    return default
end


"""Return true if the schedule is empty or sweep has passed the last scheduled sweep."""
_schedule_complete(schedule::Vector{Pair{Int,Int}}, sweep::Int) =
    isempty(schedule) || sweep >= last(schedule).first


"""
    DMRG

DMRG solver for finding the ground state of a Fork Tensor Network Hamiltonian.

# Fields
- `params::DMRGParams`: Algorithm parameters.
- `envs::FTNEnvironments`: Boundary environment tensors (shared struct).
- `E::Float64`: Current ground-state energy estimate.
- `cbe_stats::Dict{Symbol,Int}`: Per-sweep CBE skip/success statistics.
- `current_sweep::Int`: Current sweep number (used for 3S sweep cutoff).
"""
mutable struct DMRG

    params::DMRGParams
    envs::FTNEnvironments
    E::Float64
    cbe_stats::Dict{Symbol,Int}
    current_sweep::Int

    function DMRG(params::DMRGParams)
        params.verbose && show(stdout, params)
        return new(params, FTNEnvironments(), 0.0, Dict{Symbol,Int}(), 0)
    end

    """Backward-compatible constructor from `Dict{String,Any}`."""
    function DMRG(params::Dict{String,Any})
        return DMRG(DMRGParams(params))
    end

end



"""
    run_dmrg!(dmrg, H, ψ₀) -> (E, ψ)

Run the DMRG algorithm to find the ground state of Hamiltonian `H` starting from
initial state `ψ₀`. Returns the ground-state energy `E` and the optimized state `ψ`.

# Example
```julia
dmrg = DMRG(DMRGParams(χˣ=20, χʸ=20))
E, ψ = run_dmrg!(dmrg, H, ψ₀)
```
"""
function run_dmrg!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ₀::ForkTensorNetworkState)

    ψ = deepcopy(ψ₀)
    ψ.χˣ = dmrg.params.χˣ
    ψ.χʸ = dmrg.params.χʸ

    E₀ = 0.0
    δE = 1.0

    se = dmrg.params.subspace_expansion

    set_initial_environments!(dmrg, H, ψ)
    χˣ_max = dmrg.params.χˣ
    χʸ_max = dmrg.params.χʸ
    for i = 1:dmrg.params.max_iter

        dmrg.current_sweep = i
        empty!(dmrg.cbe_stats)
        ψ.max_S = 0.0

        # Resolve χ for this sweep (schedule or constant)
        dmrg.params.χˣ = resolve_χ(dmrg.params.χˣ_schedule, χˣ_max, i)
        dmrg.params.χʸ = resolve_χ(dmrg.params.χʸ_schedule, χʸ_max, i)
        ψ.χˣ = dmrg.params.χˣ
        ψ.χʸ = dmrg.params.χʸ

        if dmrg.params.method == :single_site

            single_site_sweep!(dmrg, H, ψ)

            if se in (:subspace_3s, :cbe_3s)
                dmrg.params.α *= dmrg.params.α_decay
            end

        elseif dmrg.params.method == :two_site

            two_site_sweep!(dmrg, H, ψ)

        elseif dmrg.params.method == :hybrid

            hybrid_sweep!(dmrg, H, ψ)

            if se in (:subspace_3s, :cbe_3s)
                dmrg.params.α *= dmrg.params.α_decay
            end

        end

        δE = abs(dmrg.E - E₀) / abs(dmrg.E)
        E₀ = dmrg.E

        if dmrg.params.verbose
            actual_χˣ = maximum(dim(ψ.aux_x_idx[x]) for x in 1:ψ.Lx-1)
            actual_χʸ = maximum(dim(ψ.aux_y_idx[x,y]) for x in 1:ψ.Lx for y in 1:ψ.Ly-1)
            @printf("Iteration: %d, Energy: %.12f, δE: %.4e, χˣ: %d(%d), χʸ: %d(%d), max_S: %.4f\n",
                i, dmrg.E, δE, dmrg.params.χˣ, actual_χˣ, dmrg.params.χʸ, actual_χʸ, ψ.max_S)
            if se in (:cbe, :cbe_3s) && !isempty(dmrg.cbe_stats)
                total = sum(values(dmrg.cbe_stats))
                n_success = get(dmrg.cbe_stats, :success, 0)
                n_skip = total - n_success
                print("  CBE: $n_success/$total expanded")
                if n_skip > 0
                    skip_parts = String[]
                    for (k, v) in dmrg.cbe_stats
                        k == :success && continue
                        push!(skip_parts, "$(k)=$v")
                    end
                    print(" (skipped: ", join(skip_parts, ", "), ")")
                end
                println()
            end
            if se in (:subspace_3s, :cbe_3s) && dmrg.params.max_3s_sweeps > 0 && i == dmrg.params.max_3s_sweeps
                println("  3S subspace expansion disabled after sweep $i (max_3s_sweeps reached)")
            end
        end

        schedule_done = _schedule_complete(dmrg.params.χˣ_schedule, i) &&
                        _schedule_complete(dmrg.params.χʸ_schedule, i)
        schedule_done && δE < dmrg.params.convergence_tol && break
    end

    return dmrg.E, ψ

end


"""
    set_initial_environments!(dmrg, H, ψ)

Initialize the left, right, up, and down environment tensors for DMRG.
Sets the canonical center at (1, 1) and contracts all boundary environments.
"""
function set_initial_environments!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    allocate!(dmrg.envs, ψ.Lx, ψ.Ly)

    canonical_form!(ψ, 1, 1)

    for x = 1:ψ.Lx
        dmrg.envs.er[x, ψ.Ly] = ITensor(1.0)
        for y = (ψ.Ly-1):-1:1
            dmrg.envs.er[x, y] = dmrg.envs.er[x, y+1] * ψ.Ts[x, y+1] * H.Ws[x, y+1] * prime(dag(ψ.Ts[x, y+1]))
        end
    end

    dmrg.envs.ed[ψ.Lx] = ITensor(1.0)
    for x = (ψ.Lx-1):-1:1
        dmrg.envs.ed[x] = ψ.Ts[x+1, 1] * dmrg.envs.ed[x+1] * dmrg.envs.er[x+1, 1] * H.Ws[x+1, 1] * prime(dag(ψ.Ts[x+1, 1]))
    end

    dmrg.envs.eu[1] = ITensor(1.0)
end


"""
    single_site_sweep!(dmrg, H, ψ)

Perform one full single-site DMRG sweep: down-sweep followed by up-sweep.
Each half-sweep traverses all arms (right then left) before moving along the backbone.
"""
function single_site_sweep!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    for i = 1:(ψ.Lx-1)
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, H, ψ, :right)
        end
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, H, ψ, :left)
        end
        single_site_update_direction!(dmrg, H, ψ, :down)
    end

    for i = 1:(ψ.Lx-1)
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, H, ψ, :right)
        end
        for j = 1:(ψ.Ly-1)
            single_site_update_direction!(dmrg, H, ψ, :left)
        end
        single_site_update_direction!(dmrg, H, ψ, :up)
    end

end


"""
    single_site_update_direction!(dmrg, H, ψ, dir)

Perform a single-site DMRG update at the current canonical center, then move the
canonical center in direction `dir`. Includes optional CBE or 3S subspace expansion.
"""
function single_site_update_direction!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, dir::Symbol)

    (x, y) = ψ.canonical_center
    error_check_update_direction(ψ.Lx, ψ.Ly, x, y, dir)

    se = dmrg.params.subspace_expansion
    do_cbe = (se == :cbe || se == :cbe_3s)
    do_3s = (se == :subspace_3s || se == :cbe_3s) &&
            (dmrg.params.max_3s_sweeps == 0 || dmrg.current_sweep <= dmrg.params.max_3s_sweeps)

    if do_cbe
        cbe_result = controlled_bond_expansion!(dmrg.envs, ψ, H, dir, dmrg.params.δ)
        dmrg.cbe_stats[cbe_result] = get(dmrg.cbe_stats, cbe_result, 0) + 1
    end

    if y == 1
        Env = (dmrg.envs.eu[x], dmrg.envs.ed[x], H.Ws[x, y], dmrg.envs.er[x, y])
    else
        Env = (dmrg.envs.el[x, y], H.Ws[x, y], dmrg.envs.er[x, y])
    end
    ψ.Ts[x, y], dmrg.E = lanczos(Env, ψ.Ts[x, y];)

    if do_3s
        subspace_expansion_3s!(dmrg.envs, H, ψ, dir, dmrg.params.α)
    end

    canonical_center_move!(ψ, dir)

    update_environment!(dmrg.envs, H, ψ, x, y, dir)

end


"""
    error_check_update_direction(Lx, Ly, x, y, dir)

Validate that the canonical center position `(x, y)` is consistent with moving in direction `dir`.
Throws `ArgumentError` if the move is invalid.
"""
function error_check_update_direction(Lx::Integer, Ly::Integer, x::Integer, y::Integer, dir::Symbol)

    if dir == :right
        y == Ly && throw(ArgumentError("Canonical center should not be at the rightmost site when updating the arm right."))
    elseif dir == :left
        y == 1 && throw(ArgumentError("Canonical center should not be at the leftmost site when updating the arm left."))
    elseif dir == :down
        x == Lx && throw(ArgumentError("xc should not be at Lx when updating the backbone down."))
        y != 1 && throw(ArgumentError("Canonical center should be at y=1 when updating the backbone."))
    elseif dir == :up
        x == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
        y != 1 && throw(ArgumentError("Canonical center should be at y=1 when updating the backbone."))
    else
        throw(ArgumentError("Invalid direction."))
    end

end


"""
    two_site_sweep!(dmrg, H, ψ)

Perform one full two-site DMRG sweep: down-sweep followed by up-sweep.
"""
function two_site_sweep!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    for i = 1:(ψ.Lx-1)
        two_site_sweep_arm_right!(dmrg, H, ψ)
        two_site_sweep_arm_left!(dmrg, H, ψ)
        two_site_update_backbone_down!(dmrg, H, ψ)
    end

    for i = 1:(ψ.Lx-1)
        two_site_sweep_arm_right!(dmrg, H, ψ)
        two_site_sweep_arm_left!(dmrg, H, ψ)
        two_site_update_backbone_up!(dmrg, H, ψ)
    end

end


"""
    two_site_step_arm_right!(dmrg, H, ψ, x, y)

Single two-site DMRG step at bond (y, y+1) on arm `x`, sweeping rightward.
Optimizes the two-site tensor via Lanczos, performs SVD to split, and updates the
left environment. Handles the backbone-arm boundary (y=1) and pure arm bonds (y≥2).
"""
function two_site_step_arm_right!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState,
    x::Int, y::Int)

    χʸ = dmrg.params.χʸ

    if y == 1
        T, dmrg.E = lanczos((dmrg.envs.eu[x], dmrg.envs.ed[x], H.Ws[x, y], H.Ws[x, y+1], dmrg.envs.er[x, y+1]), ψ.Ts[x, y] * ψ.Ts[x, y+1];)
        V, S, U = svd(T, (ψ.aux_y_idx[x, y+1], ψ.phys_idx[x, y+1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y]))
        dmrg.envs.el[x, y+1] = U * dmrg.envs.eu[x] * dmrg.envs.ed[x] * H.Ws[x, y] * prime(dag(U))
    else
        T, dmrg.E = lanczos((dmrg.envs.el[x, y], H.Ws[x, y], H.Ws[x, y+1], dmrg.envs.er[x, y+1]), ψ.Ts[x, y] * ψ.Ts[x, y+1];)
        U, S, V = svd(T, (ψ.aux_y_idx[x, y-1], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y]))
        dmrg.envs.el[x, y+1] = U * dmrg.envs.el[x, y] * H.Ws[x, y] * prime(dag(U))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))

    ψ.Ts[x, y] = U
    ψ.Ts[x, y+1] = S * V
    ψ.aux_y_idx[x, y] = commonind(U, S)

    network_update!(ψ, :right)

end


"""
    two_site_step_arm_left!(dmrg, H, ψ, x, y)

Single two-site DMRG step at bond (y-1, y) on arm `x`, sweeping leftward.
Optimizes the two-site tensor via Lanczos, performs SVD to split, and updates the
right environment. Handles the backbone-arm boundary (y=2) and pure arm bonds (y≥3).
"""
function two_site_step_arm_left!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState,
    x::Int, y::Int)

    χʸ = dmrg.params.χʸ

    if y == 2
        T, dmrg.E = lanczos((dmrg.envs.eu[x], dmrg.envs.ed[x], H.Ws[x, y-1], H.Ws[x, y], dmrg.envs.er[x, y]), ψ.Ts[x, y-1] * ψ.Ts[x, y];)
        V, S, U = svd(T, (ψ.aux_y_idx[x, y], ψ.phys_idx[x, y]); cutoff=1e-10, maxdim=χʸ, lefttags=tags(ψ.aux_y_idx[x, y-1]))
    else
        T, dmrg.E = lanczos((dmrg.envs.el[x, y-1], H.Ws[x, y-1], H.Ws[x, y], dmrg.envs.er[x, y]), ψ.Ts[x, y-1] * ψ.Ts[x, y];)
        U, S, V = svd(T, (ψ.aux_y_idx[x, y-2], ψ.phys_idx[x, y-1]); cutoff=1e-10, maxdim=χʸ, righttags=tags(ψ.aux_y_idx[x, y-1]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    dmrg.envs.er[x, y-1] = V * dmrg.envs.er[x, y] * H.Ws[x, y] * prime(dag(V))

    ψ.Ts[x, y] = V
    ψ.Ts[x, y-1] = U * S
    ψ.aux_y_idx[x, y-1] = commonind(S, V)

    network_update!(ψ, :left)

end


"""
    two_site_sweep_arm_right!(dmrg, H, ψ)

Two-site DMRG update sweeping rightward along the current arm.
"""
function two_site_sweep_arm_right!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, yc) = ψ.canonical_center
    yc != 1 && throw(ArgumentError("Canonical center should be at y=1 when sweeping arm right."))

    for y = 1:(ψ.Ly-1)
        two_site_step_arm_right!(dmrg, H, ψ, x, y)
    end

end


"""
    two_site_sweep_arm_left!(dmrg, H, ψ)

Two-site DMRG update sweeping leftward along the current arm.
"""
function two_site_sweep_arm_left!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, yc) = ψ.canonical_center
    yc != ψ.Ly && throw(ArgumentError("Canonical center should be at y=Ly when sweeping arm left."))

    for y = ψ.Ly:-1:2
        two_site_step_arm_left!(dmrg, H, ψ, x, y)
    end

end


"""
    two_site_update_backbone_down!(dmrg, H, ψ)

Two-site DMRG update moving the canonical center one step down the backbone.
"""
function two_site_update_backbone_down!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, y) = ψ.canonical_center
    x == ψ.Lx && throw(ArgumentError("xc should not be at Lx when updating the backbone down."))
    y != 1 && throw(ArgumentError("Canonical center should be at y=1 when updating the backbone."))

    χˣ = dmrg.params.χˣ

    T, dmrg.E = lanczos((dmrg.envs.eu[x], dmrg.envs.er[x, 1], H.Ws[x, 1], H.Ws[x+1, 1], dmrg.envs.er[x+1, 1], dmrg.envs.ed[x+1]), ψ.Ts[x, 1] * ψ.Ts[x+1, 1];)

    if x == 1
        U, S, V = svd(T, (ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    else
        U, S, V = svd(T, (ψ.aux_x_idx[x-1], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    dmrg.envs.eu[x+1] = U * dmrg.envs.eu[x] * dmrg.envs.er[x, 1] * H.Ws[x, 1] * prime(dag(U))

    ψ.Ts[x, 1] = U
    ψ.Ts[x+1, 1] = S * V
    ψ.aux_x_idx[x] = commonind(U, S)

    network_update!(ψ, :down)

end


"""
    two_site_update_backbone_up!(dmrg, H, ψ)

Two-site DMRG update moving the canonical center one step up the backbone.
"""
function two_site_update_backbone_up!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, y) = ψ.canonical_center
    x == 1 && throw(ArgumentError("xc should not be at x=1 when updating the backbone up."))
    y != 1 && throw(ArgumentError("Canonical center should be at y=1 when updating the backbone."))

    χˣ = dmrg.params.χˣ

    T, dmrg.E = lanczos((dmrg.envs.eu[x-1], dmrg.envs.er[x-1, 1], H.Ws[x-1, 1], H.Ws[x, 1], dmrg.envs.er[x, 1], dmrg.envs.ed[x]), ψ.Ts[x-1, 1] * ψ.Ts[x, 1];)

    if x == ψ.Lx
        U, S, V = svd(T, (ψ.aux_x_idx[x-2], ψ.aux_y_idx[x-1, 1], ψ.phys_idx[x-1, 1]); cutoff=1e-10, maxdim=χˣ, righttags=tags(ψ.aux_x_idx[x-1]))
    else
        V, S, U = svd(T, (ψ.aux_x_idx[x], ψ.aux_y_idx[x, 1], ψ.phys_idx[x, 1]); cutoff=1e-10, maxdim=χˣ, lefttags=tags(ψ.aux_x_idx[x-1]))
    end
    ψ.max_S = max(ψ.max_S, von_neumann_entropy(S))
    dmrg.envs.ed[x-1] = V * dmrg.envs.ed[x] * dmrg.envs.er[x, 1] * H.Ws[x, 1] * prime(dag(V))

    ψ.Ts[x, 1] = V
    ψ.Ts[x-1, 1] = S * U
    ψ.aux_x_idx[x-1] = commonind(S, V)

    network_update!(ψ, :up)

end


"""
    hybrid_sweep_arm_right!(dmrg, H, ψ)

Hybrid arm sweep rightward: single-site + CBE at y=1 (backbone-arm boundary),
then two-site for pure arm bonds (y=2 → Ly).
"""
function hybrid_sweep_arm_right!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, yc) = ψ.canonical_center
    yc != 1 && throw(ArgumentError("Canonical center should be at y=1 when sweeping arm right."))

    # Phase 1: Single-site + CBE at y=1 (backbone tensor)
    single_site_update_direction!(dmrg, H, ψ, :right)

    # Phase 2: Two-site for pure arm bonds (y=2 → Ly-1)
    for y = 2:(ψ.Ly-1)
        two_site_step_arm_right!(dmrg, H, ψ, x, y)
    end

end


"""
    hybrid_sweep_arm_left!(dmrg, H, ψ)

Hybrid arm sweep leftward: two-site for pure arm bonds (Ly → 3),
then single-site + CBE at y=2 (backbone-arm boundary).
"""
function hybrid_sweep_arm_left!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    (x, yc) = ψ.canonical_center
    yc != ψ.Ly && throw(ArgumentError("Canonical center should be at y=Ly when sweeping arm left."))

    # Phase 1: Two-site for pure arm bonds (Ly → 3)
    for y = ψ.Ly:-1:3
        two_site_step_arm_left!(dmrg, H, ψ, x, y)
    end

    # Phase 2: Single-site + CBE at y=2 (backbone-arm boundary)
    single_site_update_direction!(dmrg, H, ψ, :left)

end


"""
    hybrid_sweep!(dmrg, H, ψ)

Hybrid DMRG sweep: two-site on pure arm bonds, single-site + CBE on backbone
and backbone-arm boundary. Down-sweep followed by up-sweep.
"""
function hybrid_sweep!(dmrg::DMRG, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState)

    for i = 1:(ψ.Lx-1)
        hybrid_sweep_arm_right!(dmrg, H, ψ)
        hybrid_sweep_arm_left!(dmrg, H, ψ)
        single_site_update_direction!(dmrg, H, ψ, :down)
    end

    for i = 1:(ψ.Lx-1)
        hybrid_sweep_arm_right!(dmrg, H, ψ)
        hybrid_sweep_arm_left!(dmrg, H, ψ)
        single_site_update_direction!(dmrg, H, ψ, :up)
    end

end
