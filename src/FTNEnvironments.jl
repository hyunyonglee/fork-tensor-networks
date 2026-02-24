"""
    FTNEnvironments

Shared environment tensor storage for FTN solvers (DMRG, TDVP, etc.).
Holds left, right, up, and down boundary environments used in sweeping algorithms.

# Fields
- `el::Union{Matrix{ITensor},Nothing}`: Left arm environments, indexed `[x, y]`.
- `er::Union{Matrix{ITensor},Nothing}`: Right arm environments, indexed `[x, y]`.
- `eu::Union{Vector{ITensor},Nothing}`: Up backbone environments, indexed `[x]`.
- `ed::Union{Vector{ITensor},Nothing}`: Down backbone environments, indexed `[x]`.
"""
mutable struct FTNEnvironments
    el::Union{Matrix{ITensor},Nothing}
    er::Union{Matrix{ITensor},Nothing}
    eu::Union{Vector{ITensor},Nothing}
    ed::Union{Vector{ITensor},Nothing}
end

"""
    FTNEnvironments()

Construct an empty `FTNEnvironments` with all fields set to `nothing`.
"""
FTNEnvironments() = FTNEnvironments(nothing, nothing, nothing, nothing)


"""
    allocate!(envs, Lx, Ly)

Allocate environment storage for a Fork Tensor Network of size `Lx × Ly`.
"""
function allocate!(envs::FTNEnvironments, Lx::Integer, Ly::Integer)
    envs.el = Matrix{ITensor}(undef, Lx, Ly)
    envs.er = Matrix{ITensor}(undef, Lx, Ly)
    envs.eu = Vector{ITensor}(undef, Lx)
    envs.ed = Vector{ITensor}(undef, Lx)
    return envs
end


"""
    update_environment!(envs, H, ψ, x, y, direction)

Update a boundary environment tensor in direction `direction` at site `(x, y)`.
Shared by DMRG and TDVP for maintaining environments during sweeps.
"""
function update_environment!(envs::FTNEnvironments, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, x::Integer, y::Integer, direction::Symbol)

    if direction == :right

        if y == 1
            envs.el[x, y+1] = ψ.Ts[x, y] * envs.eu[x] * H.Ws[x, y] * envs.ed[x] * prime(dag(ψ.Ts[x, y]))
        else
            envs.el[x, y+1] = ψ.Ts[x, y] * envs.el[x, y] * H.Ws[x, y] * prime(dag(ψ.Ts[x, y]))
        end

    elseif direction == :left

        envs.er[x, y-1] = ψ.Ts[x, y] * envs.er[x, y] * H.Ws[x, y] * prime(dag(ψ.Ts[x, y]))

    elseif direction == :down

        y !== 1 && throw(ArgumentError("y must be 1 when updating backbone environment downward."))
        envs.eu[x+1] = ψ.Ts[x, y] * envs.eu[x] * H.Ws[x, y] * envs.er[x, y] * prime(dag(ψ.Ts[x, y]))

    elseif direction == :up

        y !== 1 && throw(ArgumentError("y must be 1 when updating backbone environment upward."))
        envs.ed[x-1] = ψ.Ts[x, y] * envs.ed[x] * H.Ws[x, y] * envs.er[x, y] * prime(dag(ψ.Ts[x, y]))

    else
        throw(ArgumentError("Invalid direction: $direction"))
    end

end
