"""
    SubspaceExpansion

Solver-independent subspace expansion methods for Fork Tensor Networks.
Includes Controlled Bond Expansion (CBE) and 3S (three-site) subspace expansion.
Both methods are designed to work with any FTN solver (DMRG, TDVP, etc.)
via the shared `FTNEnvironments` struct.
"""


# ============================================================================
#  3S (Three-Site) Subspace Expansion
# ============================================================================

"""
    bond_expansion(Env, α, T1, T2, idx_T, idx_W) -> (T1_exp, T2_exp, idx_exp)

Perform 3S-style bond expansion by projecting the perturbative correction onto the
null space and enlarging the bond index via `directsum`.

# Arguments
- `Env`: Environment tensors (excluding the bond being expanded).
- `α`: Mixing parameter controlling the expansion magnitude.
- `T1`, `T2`: Tensors on either side of the bond.
- `idx_T`: Bond index to expand.
- `idx_W`: Corresponding MPO auxiliary index.
"""
function bond_expansion(Env::Tuple{Vararg{ITensor}}, α::AbstractFloat, T1::ITensor, T2::ITensor, idx_T::Index, idx_W::Index)

    P = noprime(reduce(*, Env, init=(α * T1)))
    C = combiner(idx_T, idx_W; dir=dir(idx_T))
    idx_exp = directsum(idx_T, combinedind(C); tags=tags(idx_T))

    T1_exp = directsum(idx_exp, T1 => idx_T, P * C => combinedind(C))
    T2_exp = T2 * delta(dag(idx_exp), idx_T)

    return T1_exp, T2_exp, idx_exp
end


"""
    subspace_expansion_3s!(envs, H, ψ, dir, α)

Apply 3S (three-site) subspace expansion at the current canonical center in direction `dir`.
Expands the bond dimension by mixing in perturbative corrections from the environment.

Works with any FTN solver via `FTNEnvironments`.
"""
function subspace_expansion_3s!(envs::FTNEnvironments, H::ForkTensorNetworkOperator, ψ::ForkTensorNetworkState, dir::Symbol, α::AbstractFloat)

    (x, y) = ψ.canonical_center

    if dir == :right

        if y == 1
            Env = (envs.eu[x], envs.ed[x], H.Ws[x, y])
        else
            Env = (envs.el[x, y], H.Ws[x, y])
        end
        ψ.Ts[x, y], ψ.Ts[x, y+1], ψ.aux_y_idx[x, y] = bond_expansion(Env, α, ψ.Ts[x, y], ψ.Ts[x, y+1], ψ.aux_y_idx[x, y], H.aux_y_idx[x, y])

    elseif dir == :left

        Env = (envs.er[x, y], H.Ws[x, y])
        ψ.Ts[x, y], ψ.Ts[x, y-1], ψ.aux_y_idx[x, y-1] = bond_expansion(Env, α, ψ.Ts[x, y], ψ.Ts[x, y-1], dag(ψ.aux_y_idx[x, y-1]), dag(H.aux_y_idx[x, y-1]))

    elseif dir == :down

        Env = (envs.eu[x], envs.er[x, 1], H.Ws[x, 1])
        ψ.Ts[x, 1], ψ.Ts[x+1, 1], ψ.aux_x_idx[x] = bond_expansion(Env, α, ψ.Ts[x, 1], ψ.Ts[x+1, 1], ψ.aux_x_idx[x], H.aux_x_idx[x])

    elseif dir == :up

        Env = (envs.ed[x], envs.er[x, 1], H.Ws[x, 1])
        ψ.Ts[x, 1], ψ.Ts[x-1, 1], ψ.aux_x_idx[x-1] = bond_expansion(Env, α, ψ.Ts[x, 1], ψ.Ts[x-1, 1], dag(ψ.aux_x_idx[x-1]), dag(H.aux_x_idx[x-1]))

    end

end


# ============================================================================
#  Controlled Bond Expansion (CBE)
# ============================================================================

"""
    orth_env(E, T) -> ITensor

Compute the orthogonal complement of the environment `E` with respect to tensor `T`:
    E_orth = E - (E·T†)·T†
"""
function orth_env(E::ITensor, T::ITensor)
    return noprime(E - (E * dag(T')) * T')
end


"""
    shrewd_selection(Env1, Env2, T1, T2, Σ, idx, idx_W, Dp, Dt; tol=1e-8) -> (Aex, idx_ex) or Symbol

Perform the shrewd selection procedure for controlled bond expansion (CBE).
Identifies and constructs the optimal expansion subspace for bond enrichment.

Returns `(Aex, idx_ex)` on success, or a Symbol indicating skip reason:
`:skip_Rorth`, `:skip_Lorth`, `:skip_Apr`.

# Arguments
- `Env1`, `Env2`: Environment tensor tuples for the two sides of the bond.
- `T1`, `T2`: Local tensors on either side of the bond.
- `Σ`: Singular value matrix from the local SVD.
- `idx`: Bond index to be expanded.
- `idx_W`: Corresponding MPO auxiliary index.
- `Dp`, `Dt`: Maximum dimensions for perturbative and truncation subspaces.
- `tol`: Tolerance for skipping negligible expansions.
"""
function shrewd_selection(Env1::Tuple{Vararg{ITensor}}, Env2::Tuple{Vararg{ITensor}}, T1::ITensor, T2::ITensor, Σ::ITensor, idx::Index, idx_W::Index, Dp::Integer, Dt::Integer; tol=1e-8)

    Rtmp = reduce(*, Env1, init=(T1 * Σ))
    Rorth = orth_env(Rtmp, T1)

    if norm(Rorth) < tol
        return :skip_Rorth
    end

    U, S, ~ = svd(Rorth, commoninds(Rorth, Σ))

    Ltmp = reduce(*, Env2, init=(U * S * T2))
    Lorth = orth_env(Ltmp, T2)

    if norm(Lorth) < tol
        return :skip_Lorth
    end

    u, s, ~ = svd(Lorth, uniqueinds(Lorth, S); cutoff=1e-16, maxdim=Dp)

    Q = u * s

    # Guard: SVD requires valid QN blocks on both sides.
    # With QN-conserving tensors, the combined left index (idx_W ⊗ svd_index)
    # can have empty QN sectors when bath-site symmetry causes exact cancellation.
    ci = commoninds(Q, s)
    if length(ci) == 0 || any(i -> dim(i) == 0, ci)
        return :skip_Apr
    end
    ri = uniqueinds(Q, idx_W, ci...)
    if length(ri) == 0 || any(i -> dim(i) == 0, ri)
        return :skip_Apr
    end

    ~, ~, Apr = svd(Q, (idx_W, ci...); cutoff=1e-16)

    if abs(norm(dag(T2) * Apr)) > tol
        return :skip_Apr
    end

    Lpr = reduce(*, Env2, init=T2) * dag(Apr')

    Ctmp = Rtmp * Lpr
    Corth = orth_env(Ctmp, T1)

    if norm(Corth) < tol
        Aex, idx_ex = directsum(T2 => uniqueind(T2, Apr), Apr => uniqueind(Apr, T2); tags=tags(idx))
    else
        u, ~, ~ = svd(Corth, uniqueinds(Corth, Rorth); cutoff=1e-16, maxdim=Dt)
        Atr = u * Apr
        Aex, idx_ex = directsum(T2 => uniqueind(T2, Atr), Atr => uniqueind(Atr, T2); tags=tags(idx))
    end

    return (Aex, idx_ex)

end


"""
    controlled_bond_expansion!(envs, ψ, H, dir, δ; tol=1e-8) -> Symbol

Perform controlled bond expansion (CBE) at the current canonical center along direction `dir`.
Enriches the bond dimension by adding perturbatively selected basis vectors.

Returns `:success` if expansion was applied, or a skip reason Symbol
(`:skip_Rorth`, `:skip_Lorth`, `:skip_Apr`) if the expansion was skipped.

Works with any FTN solver via `FTNEnvironments`.

# Arguments
- `envs`: FTN environment tensors.
- `ψ`: Fork tensor network state (modified in-place on success).
- `H`: Fork tensor network operator (Hamiltonian).
- `dir`: Expansion direction (`:right`, `:left`, `:up`, `:down`).
- `δ`: Expansion ratio controlling the truncation subspace dimension.
"""
function controlled_bond_expansion!(envs::FTNEnvironments, ψ::ForkTensorNetworkState, H::ForkTensorNetworkOperator, dir::Symbol, δ::Real; tol=1e-8)

    (x, y) = ψ.canonical_center

    if dir == :down
        Env1 = (envs.eu[x], H.Ws[x, 1], envs.er[x, 1])
        Env2 = (envs.ed[x+1], H.Ws[x+1, 1], envs.er[x+1, 1])
        bond_index = ψ.aux_x_idx[x]
        idx_W = dag(H.aux_x_idx[x])
        (dx, dy) = (1, 0)
    elseif dir == :up
        Env1 = (envs.ed[x], H.Ws[x, 1], envs.er[x, 1])
        Env2 = (envs.eu[x-1], H.Ws[x-1, 1], envs.er[x-1, 1])
        bond_index = dag(ψ.aux_x_idx[x-1])
        idx_W = dag(H.aux_x_idx[x-1])
        (dx, dy) = (-1, 0)
    elseif dir == :right
        Env1 = y == 1 ? (envs.eu[x], H.Ws[x, y], envs.ed[x]) : (envs.el[x, y], H.Ws[x, y])
        Env2 = (envs.er[x, y+1], H.Ws[x, y+1])
        bond_index = ψ.aux_y_idx[x, y]
        idx_W = dag(H.aux_y_idx[x, y])
        (dx, dy) = (0, 1)
    elseif dir == :left
        Env2 = y == 2 ? (envs.eu[x], H.Ws[x, 1], envs.ed[x]) : (envs.el[x, y-1], H.Ws[x, y-1])
        Env1 = (envs.er[x, y], H.Ws[x, y])
        bond_index = dag(ψ.aux_y_idx[x, y-1])
        idx_W = dag(H.aux_y_idx[x, y-1])
        (dx, dy) = (0, -1)
    else
        error("Invalid direction: $dir")
    end

    T1, Σ, V = svd(ψ.Ts[x, y], uniqueinds(ψ.Ts[x, y], bond_index); cutoff=1e-16)
    T2 = V * ψ.Ts[x+dx, y+dy]

    Dp = ceil(Int, dim(bond_index) / dim(idx_W))
    Dt = ceil(Int, δ * dim(bond_index))
    result = shrewd_selection(Env1, Env2, T1, T2, Σ, bond_index, idx_W, Dp, Dt; tol=tol)

    if result isa Symbol
        return result
    end

    Aex, idx_ex = result

    ψ.Ts[x, y] = (dag(Aex) * T2) * (T1 * Σ)
    ψ.Ts[x+dx, y+dy] = Aex

    if dir == :down
        ψ.aux_x_idx[x] = dag(idx_ex)
        envs.ed[x] = envs.ed[x+1] * Aex * H.Ws[x+1, 1] * dag(Aex') * envs.er[x+1, 1]
    elseif dir == :up
        ψ.aux_x_idx[x-1] = idx_ex
        envs.eu[x] = envs.eu[x-1] * Aex * H.Ws[x-1, 1] * dag(Aex') * envs.er[x-1, 1]
    elseif dir == :right
        ψ.aux_y_idx[x, y] = dag(idx_ex)
        envs.er[x, y] = envs.er[x, y+1] * Aex * H.Ws[x, y+1] * dag(Aex')
    elseif dir == :left
        ψ.aux_y_idx[x, y-1] = idx_ex
        if y == 2
            envs.el[x, y] = envs.eu[x] * Aex * H.Ws[x, 1] * dag(Aex') * envs.ed[x]
        else
            envs.el[x, y] = envs.el[x, y-1] * Aex * H.Ws[x, y-1] * dag(Aex')
        end
    end

    return :success
end
