module ClusterAndersonImpurityModel
using ITensors, ITensorMPS

include("TwoSiteFermion.jl")

export ftno_cluster_aim_model, ftns_cluster_initial_state


# ─── Bath tensors (star geometry, V_jmk extension) ───
# Bath sites use "Fermion" sitetype (dim=2, spinless)
# Spin-up arm: aux dim=7, entries: εn, I, p, V₁c, V₁*c†, V₂c, V₂*c†
# Spin-down arm: aux dim=6, entries: εn, I, V₁c, V₁*c†, V₂c, V₂*c†
# Site-dependent hybridization V_jmk (FTN_DMFT.pdf Eq. 39-42)

function W_cluster_bath_spin_up(s::Index, l::Index, r::Union{Index,Nothing},
    ε::Float64, V1::ComplexF64, V2::ComplexF64; edge::Bool=false)

    if edge
        # Eq. 39: edge (first) bath tensor — column vector (dim 7)
        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("F", s)
        W += onehot(l => 4) * op("c", s) * V1
        W += onehot(l => 5) * op("c†", s) * conj(V1)
        W += onehot(l => 6) * op("c", s) * V2
        W += onehot(l => 7) * op("c†", s) * conj(V2)
    else
        # Eq. 40: bulk bath tensor — 7×7 matrix
        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 5, r => 5) * op("F", s)
        W += onehot(l => 6, r => 6) * op("F", s)
        W += onehot(l => 7, r => 7) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 4, r => 2) * op("c", s) * V1
        W += onehot(l => 5, r => 2) * op("c†", s) * conj(V1)
        W += onehot(l => 6, r => 2) * op("c", s) * V2
        W += onehot(l => 7, r => 2) * op("c†", s) * conj(V2)
    end

    return W
end


function W_cluster_bath_spin_down(s::Index, l::Index, r::Union{Index,Nothing},
    ε::Float64, V1::ComplexF64, V2::ComplexF64; edge::Bool=false)

    if edge
        # Eq. 41: edge (first) bath tensor — column vector (dim 6)
        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("c", s) * V1
        W += onehot(l => 4) * op("c†", s) * conj(V1)
        W += onehot(l => 5) * op("c", s) * V2
        W += onehot(l => 6) * op("c†", s) * conj(V2)
    else
        # Eq. 42: bulk bath tensor — 6×6 matrix
        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 5, r => 5) * op("F", s)
        W += onehot(l => 6, r => 6) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 3, r => 2) * op("c", s) * V1
        W += onehot(l => 4, r => 2) * op("c†", s) * conj(V1)
        W += onehot(l => 5, r => 2) * op("c", s) * V2
        W += onehot(l => 6, r => 2) * op("c†", s) * conj(V2)
    end

    return W
end


# ─── Helper: operator product with correct prime structure ───
oprod(A::String, B::String, s::Index) = swapprime(op(A, s)' * op(B, s), 2 => 1)


# ─── Impurity tensors (star geometry, 2-orbital 2-site cluster, V_jmk) ───
# Physical index s uses "2SiteFermion" sitetype
# FTN_DMFT.pdf Eq. 58-61

# Eq. 58: W^imp_{A↑} — first orbital, spin-up
# backbone d: dim=8, arm r: dim=7 (spin-up bath, V_jmk)
function W_cluster_imp_A_up(s::Index, d::Index, r::Index,
    ε₁::Float64, ε₂::Float64, t::Float64)

    # Row 1: H^IB_{A↑} — connects to bath entries 1,2,4,5,6,7
    W = onehot(d => 1, r => 1) * op("I", s)
    hop = oprod("d1†", "d2", s) + oprod("d2†", "d1", s)
    W += onehot(d => 1, r => 2) * (op("N1", s) * ε₁ + op("N2", s) * ε₂ - t * hop)
    W += onehot(d => 1, r => 4) * oprod("d1†", "F2", s)     # site 1: V₁*c† channel
    W += onehot(d => 1, r => 5) * oprod("d1", "F2", s)      # site 1: V₁c channel
    W += onehot(d => 1, r => 6) * op("d2†", s)               # site 2: V₂*c† channel
    W += onehot(d => 1, r => 7) * op("d2", s)                # site 2: V₂c channel

    # Rows 2-4: identity and number operators → bath index 2
    W += onehot(d => 2, r => 2) * op("I", s)
    W += onehot(d => 3, r => 2) * op("N1", s)
    W += onehot(d => 4, r => 2) * op("N2", s)

    # Rows 5-8: creation/annihilation with Fermi strings → bath index 3
    W += onehot(d => 5, r => 3) * oprod("d1†", "F1F2", s)
    W += onehot(d => 6, r => 3) * oprod("d1", "F1F2", s)
    W += onehot(d => 7, r => 3) * oprod("d2†", "F2", s)
    W += onehot(d => 8, r => 3) * oprod("d2", "F2", s)

    return W
end


# Eq. 59: W^imp_{A↓} — first orbital, spin-down
# u: dim=8 (from A↑), d: dim=14, r: dim=6 (spin-down bath, V_jmk)
function W_cluster_imp_A_down(s::Index, d::Index, u::Index, r::Index,
    ε₁::Float64, ε₂::Float64, t::Float64, U::Float64)

    # Row 1: H^IB_{A↓} + Coulomb
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)

    hop = oprod("d1†", "d2", s) + oprod("d2†", "d1", s)
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * (op("N1", s) * ε₁ + op("N2", s) * ε₂ - t * hop)
    W += onehot(d => 1, u => 2, r => 3) * oprod("d1†", "F2", s)     # site 1: V₁*c† channel
    W += onehot(d => 1, u => 2, r => 4) * oprod("d1", "F2", s)      # site 1: V₁c channel
    W += onehot(d => 1, u => 2, r => 5) * op("d2†", s)               # site 2: V₂*c† channel
    W += onehot(d => 1, u => 2, r => 6) * op("d2", s)                # site 2: V₂c channel

    W += onehot(d => 1, u => 3, r => 2) * op("N1", s) * U
    W += onehot(d => 1, u => 4, r => 2) * op("N2", s) * U

    # Row 2: identity pass-through
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)

    # Row 3: n₁A↑ pass-through
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)

    # Row 4: n₁A↓ new
    W += onehot(d => 4, u => 2, r => 2) * op("N1", s)

    # Row 5: n₂A↑ pass-through
    W += onehot(d => 5, u => 4, r => 2) * op("I", s)

    # Row 6: n₂A↓ new
    W += onehot(d => 6, u => 2, r => 2) * op("N2", s)

    # Rows 7-8: spin-flip from u=5 (d₁A↑† p₁p₂)
    W += onehot(d => 7, u => 5, r => 2) * op("d1", s)
    W += onehot(d => 8, u => 5, r => 2) * op("d1†", s)

    # Rows 9-10: pair-hopping from u=6 (d₁A↑ p₁p₂)
    W += onehot(d => 9, u => 6, r => 2) * op("d1†", s)
    W += onehot(d => 10, u => 6, r => 2) * op("d1", s)

    # Rows 11-12: spin-flip from u=7 (d₂A↑† p₂)
    W += onehot(d => 11, u => 7, r => 2) * oprod("F1", "d2", s)
    W += onehot(d => 12, u => 7, r => 2) * oprod("F1", "d2†", s)

    # Rows 13-14: pair-hopping from u=8 (d₂A↑ p₂)
    W += onehot(d => 13, u => 8, r => 2) * oprod("F1", "d2†", s)
    W += onehot(d => 14, u => 8, r => 2) * oprod("F1", "d2", s)

    return W
end


# Eq. 60: W^imp_{B↑} — second orbital, spin-up
# u: dim=14 (from A↓), d: dim=12, r: dim=7 (spin-up bath, V_jmk)
function W_cluster_imp_B_up(s::Index, d::Index, u::Index, r::Index,
    ε₁::Float64, ε₂::Float64, t::Float64, U′::Float64, J::Float64)

    # Row 1: H^IB_{B↑} + inter-orbital Coulomb
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)

    hop = oprod("d1†", "d2", s) + oprod("d2†", "d1", s)
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * (op("N1", s) * ε₁ + op("N2", s) * ε₂ - t * hop)
    W += onehot(d => 1, u => 2, r => 4) * oprod("d1†", "F2", s)     # site 1: V₁*c† channel
    W += onehot(d => 1, u => 2, r => 5) * oprod("d1", "F2", s)      # site 1: V₁c channel
    W += onehot(d => 1, u => 2, r => 6) * op("d2†", s)               # site 2: V₂*c† channel
    W += onehot(d => 1, u => 2, r => 7) * op("d2", s)                # site 2: V₂c channel

    # Inter-orbital Coulomb: (U'-J) same-spin, U' diff-spin
    W += onehot(d => 1, u => 3, r => 2) * op("N1", s) * (U′ - J)
    W += onehot(d => 1, u => 4, r => 2) * op("N1", s) * U′
    W += onehot(d => 1, u => 5, r => 2) * op("N2", s) * (U′ - J)
    W += onehot(d => 1, u => 6, r => 2) * op("N2", s) * U′

    # Rows 2-8: identity pass-through and new n, interleaved per site
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)       # n₁A↑ pass
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)       # n₁A↓ pass
    W += onehot(d => 5, u => 2, r => 2) * op("N1", s)      # n₁B↑ new
    W += onehot(d => 6, u => 5, r => 2) * op("I", s)       # n₂A↑ pass
    W += onehot(d => 7, u => 6, r => 2) * op("I", s)       # n₂A↓ pass
    W += onehot(d => 8, u => 2, r => 2) * op("N2", s)      # n₂B↑ new

    # Rows 9-12: SF/PH operators × d_{B↑} with Fermi string → bath index 3
    # Site 1: d=9 completed at B↓ with Jd₁†, d=10 with Jd₁
    W += onehot(d => 9, u => 7, r => 3) * oprod("d1", "F1F2", s)            # SF1: d₁B↑ F₁F₂
    W += onehot(d => 9, u => 10, r => 3) * (-1) * oprod("d1†", "F1F2", s)   # PH4: -d₁B↑† F₁F₂
    W += onehot(d => 10, u => 8, r => 3) * (-1) * oprod("d1", "F1F2", s)    # PH3: -d₁B↑ F₁F₂
    W += onehot(d => 10, u => 9, r => 3) * oprod("d1†", "F1F2", s)          # SF2: d₁B↑† F₁F₂

    # Site 2: d=11 completed at B↓ with Jp₁d₂†, d=12 with Jp₁d₂
    W += onehot(d => 11, u => 11, r => 3) * oprod("d2", "F2", s)            # SF1: d₂B↑ F₂
    W += onehot(d => 11, u => 14, r => 3) * (-1) * oprod("d2†", "F2", s)    # PH4: -d₂B↑† F₂
    W += onehot(d => 12, u => 12, r => 3) * (-1) * oprod("d2", "F2", s)     # PH3: -d₂B↑ F₂
    W += onehot(d => 12, u => 13, r => 3) * oprod("d2†", "F2", s)           # SF2: d₂B↑† F₂

    return W
end


# Eq. 61: W^imp_{B↓} — second orbital, spin-down (last tensor, row vector)
# u: dim=12 (from B↑), no d index, r: dim=6 (spin-down bath, V_jmk)
function W_cluster_imp_B_down(s::Index, u::Index, r::Index,
    ε₁::Float64, ε₂::Float64, t::Float64,
    U::Float64, U′::Float64, J::Float64)

    hop = oprod("d1†", "d2", s) + oprod("d2†", "d1", s)
    H_IB = op("N1", s) * ε₁ + op("N2", s) * ε₂ - t * hop

    W = onehot(u => 1, r => 2) * op("I", s)

    W += onehot(u => 2, r => 1) * op("I", s)
    W += onehot(u => 2, r => 2) * H_IB
    W += onehot(u => 2, r => 3) * oprod("d1†", "F2", s)     # site 1: V₁*c† channel
    W += onehot(u => 2, r => 4) * oprod("d1", "F2", s)      # site 1: V₁c channel
    W += onehot(u => 2, r => 5) * op("d2†", s)               # site 2: V₂*c† channel
    W += onehot(u => 2, r => 6) * op("d2", s)                # site 2: V₂c channel

    # Coulomb — matching Eq. 14 downstream order
    W += onehot(u => 3, r => 2) * op("N1", s) * U′           # n₁A↑ × U'n₁B↓
    W += onehot(u => 4, r => 2) * op("N1", s) * (U′ - J)     # n₁A↓ × (U'-J)n₁B↓
    W += onehot(u => 5, r => 2) * op("N1", s) * U             # n₁B↑ × Un₁B↓
    W += onehot(u => 6, r => 2) * op("N2", s) * U′           # n₂A↑ × U'n₂B↓
    W += onehot(u => 7, r => 2) * op("N2", s) * (U′ - J)     # n₂A↓ × (U'-J)n₂B↓
    W += onehot(u => 8, r => 2) * op("N2", s) * U             # n₂B↑ × Un₂B↓

    # SF/PH completion with J
    W += onehot(u => 9, r => 2) * op("d1†", s) * J
    W += onehot(u => 10, r => 2) * op("d1", s) * J
    W += onehot(u => 11, r => 2) * oprod("F1", "d2†", s) * J
    W += onehot(u => 12, r => 2) * oprod("F1", "d2", s) * J

    return W
end


# ─── Index construction for 2-orbital 2-site cluster AIM ───

function ftno_cluster_indices(N_bath::Int; conserve_qns=true)

    Lx = 4   # A↑, A↓, B↑, B↓
    Ly = N_bath + 1

    phys_idx = Matrix{Index}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)

    for x = 1:Lx
        # y=1: impurity site (2SiteFermion), y>1: bath site (Fermion)
        if x % 2 == 1 && conserve_qns
            imp_site = siteind("2SiteFermion"; conserve_qns=true, conserve_sz="Up")
            bath_sites = siteinds("Fermion", Ly - 1; conserve_qns=true, conserve_sz="Up")
        elseif x % 2 == 0 && conserve_qns
            imp_site = siteind("2SiteFermion"; conserve_qns=true, conserve_sz="Dn")
            bath_sites = siteinds("Fermion", Ly - 1; conserve_qns=true, conserve_sz="Dn")
        else
            imp_site = siteind("2SiteFermion"; conserve_qns=false)
            bath_sites = siteinds("Fermion", Ly - 1; conserve_qns=false)
        end

        imp_site = addtags(imp_site, "x=$(x),y=1")
        phys_idx[x, 1] = imp_site
        for y = 2:Ly
            phys_idx[x, y] = replacetags(bath_sites[y-1], "n=$(y-1)" => "x=$(x),y=$(y)")
        end

        # Arm auxiliary indices for V_jmk (FTN_DMFT.pdf Eq. 39-42)
        for y = 1:(Ly-1)
            if x % 2 == 1
                # Spin-up arm: dim=7 [εn, 𝟙, p, V₁c, V₁*c†, V₂c, V₂*c†]
                aux_y_idx[x, y] = Index([
                    QN(("Nf", 0, -1), ("Sz", 0)) => 3,
                    QN(("Nf", -1, -1), ("Sz", -1)) => 1,
                    QN(("Nf", +1, -1), ("Sz", +1)) => 1,
                    QN(("Nf", -1, -1), ("Sz", -1)) => 1,
                    QN(("Nf", +1, -1), ("Sz", +1)) => 1
                ], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
            else
                # Spin-down arm: dim=6 [εn, 𝟙, V₁c, V₁*c†, V₂c, V₂*c†]
                aux_y_idx[x, y] = Index([
                    QN(("Nf", 0, -1), ("Sz", 0)) => 2,
                    QN(("Nf", -1, -1), ("Sz", +1)) => 1,
                    QN(("Nf", +1, -1), ("Sz", -1)) => 1,
                    QN(("Nf", -1, -1), ("Sz", +1)) => 1,
                    QN(("Nf", +1, -1), ("Sz", -1)) => 1
                ], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
            end
        end
    end

    # Backbone auxiliary indices — QN blocks match entry order (not merged)
    # aux_x_idx[1]: A↑ → A↓, dim=8
    # 1-4: (0,0), 5: d₁†(-1,-1), 6: d₁(+1,+1), 7: d₂†(-1,-1), 8: d₂(+1,+1)
    aux_x_idx[1] = Index([
        QN(("Nf", 0, -1), ("Sz", 0)) => 4,
        QN(("Nf", -1, -1), ("Sz", -1)) => 1,
        QN(("Nf", 1, -1), ("Sz", 1)) => 1,
        QN(("Nf", -1, -1), ("Sz", -1)) => 1,
        QN(("Nf", 1, -1), ("Sz", 1)) => 1
    ], "FTNO,Backbone,x=(1-2),y=1"; dir=ITensors.Out)

    # aux_x_idx[2]: A↓ → B↑, dim=14
    # 1-6: (0,0)
    # 7: d₁↓(0,-2), 8: d₁↓†(-2,0), 9: d₁↓†(0,+2), 10: d₁↓(+2,0)
    # 11: F₁d₂↓(0,-2), 12: F₁d₂↓†(-2,0), 13: F₁d₂↓†(0,+2), 14: F₁d₂↓(+2,0)
    aux_x_idx[2] = Index([
        QN(("Nf", 0, -1), ("Sz", 0)) => 6,
        QN(("Nf", 0, -1), ("Sz", -2)) => 1,
        QN(("Nf", -2, -1), ("Sz", 0)) => 1,
        QN(("Nf", 0, -1), ("Sz", 2)) => 1,
        QN(("Nf", 2, -1), ("Sz", 0)) => 1,
        QN(("Nf", 0, -1), ("Sz", -2)) => 1,
        QN(("Nf", -2, -1), ("Sz", 0)) => 1,
        QN(("Nf", 0, -1), ("Sz", 2)) => 1,
        QN(("Nf", 2, -1), ("Sz", 0)) => 1
    ], "FTNO,Backbone,x=(2-3),y=1"; dir=ITensors.Out)

    # aux_x_idx[3]: B↑ → B↓, dim=12
    # 1-8: (0,0)
    # 9: (+1,-1), 10: (-1,+1), 11: (+1,-1), 12: (-1,+1)
    aux_x_idx[3] = Index([
        QN(("Nf", 0, -1), ("Sz", 0)) => 8,
        QN(("Nf", 1, -1), ("Sz", -1)) => 1,
        QN(("Nf", -1, -1), ("Sz", 1)) => 1,
        QN(("Nf", 1, -1), ("Sz", -1)) => 1,
        QN(("Nf", -1, -1), ("Sz", 1)) => 1
    ], "FTNO,Backbone,x=(3-4),y=1"; dir=ITensors.Out)

    if !conserve_qns
        for x = 1:Lx, y = 1:(Ly-1)
            aux_y_idx[x, y] = removeqns(aux_y_idx[x, y])
        end
        for x = 1:(Lx-1)
            aux_x_idx[x] = removeqns(aux_x_idx[x])
        end
    end

    return phys_idx, aux_x_idx, aux_y_idx
end


# ─── FTNO model construction ───

function ftno_cluster_aim_model(model_params::Dict{String,Any})

    N_bath = model_params["N_bath"]
    conserve_qns = model_params["conserve_qns"]
    U = model_params["U"]
    U′ = model_params["U′"]
    J = model_params["J"]
    t = model_params["t"]
    εₖ = model_params["εₖ"]    # 4 × (N_bath+2): [imp_site1, imp_site2, bath1, ..., bathN]
    Vₖ = model_params["Vₖ"]    # 4 × (N_bath+2) × 2: site-dependent V_jmk (ComplexF64)

    Lx = 4
    Ly = N_bath + 1

    Ws = Matrix{ITensor}(undef, Lx, Ly)
    phys_idx, aux_x_idx, aux_y_idx = ftno_cluster_indices(N_bath; conserve_qns=conserve_qns)

    # Bath tensors (phys y=2,...,Ly maps to εₖ/Vₖ column y+1)
    for x = 1:2:Lx
        for y = 2:Ly
            is_edge = (y == Ly)
            l = dag(aux_y_idx[x, y-1])
            r_up = is_edge ? nothing : aux_y_idx[x, y]
            r_dn = is_edge ? nothing : aux_y_idx[x+1, y]

            Ws[x, y] = W_cluster_bath_spin_up(
                phys_idx[x, y], l, r_up, εₖ[x, y+1],
                ComplexF64(Vₖ[x, y+1, 1]), ComplexF64(Vₖ[x, y+1, 2]); edge=is_edge)
            Ws[x+1, y] = W_cluster_bath_spin_down(
                phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), r_dn, εₖ[x+1, y+1],
                ComplexF64(Vₖ[x+1, y+1, 1]), ComplexF64(Vₖ[x+1, y+1, 2]); edge=is_edge)
        end
    end

    # Impurity tensors (y=1, ε₁=εₖ[x,1], ε₂=εₖ[x,2])
    # x=1: A↑
    Ws[1, 1] = W_cluster_imp_A_up(
        phys_idx[1, 1], aux_x_idx[1], aux_y_idx[1, 1],
        εₖ[1, 1], εₖ[1, 2], t)

    # x=2: A↓
    Ws[2, 1] = W_cluster_imp_A_down(
        phys_idx[2, 1], aux_x_idx[2], dag(aux_x_idx[1]), aux_y_idx[2, 1],
        εₖ[2, 1], εₖ[2, 2], t, U)

    # x=3: B↑
    Ws[3, 1] = W_cluster_imp_B_up(
        phys_idx[3, 1], aux_x_idx[3], dag(aux_x_idx[2]), aux_y_idx[3, 1],
        εₖ[3, 1], εₖ[3, 2], t, U′, J)

    # x=4: B↓
    Ws[4, 1] = W_cluster_imp_B_down(
        phys_idx[4, 1], dag(aux_x_idx[3]), aux_y_idx[4, 1],
        εₖ[4, 1], εₖ[4, 2], t, U, U′, J)

    return Ws, phys_idx, aux_x_idx, aux_y_idx
end


# ─── Initial state construction ───

function ftns_cluster_initial_state(phys_idx::AbstractMatrix{<:Index}, ρ::Float64;
    conserve_qns=true, imp_init::String="afm", bath_init::String="pair_symmetric")

    Lx = size(phys_idx, 1)
    Ly = size(phys_idx, 2)

    Ts = Matrix{ITensor}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)

    # Product state for each arm
    for x = 1:Lx
        # Impurity (y=1): 2SiteFermion states
        # Bath (y=2,...): Fermion states
        n_bath = Ly - 1
        n_occ_bath = clamp(round(Int, ρ * n_bath), 0, n_bath)

        bath_state = fill("Emp", n_bath)
        if bath_init == "pair_symmetric"
            # The cluster bath discretizer stores degenerate shell channels in
            # consecutive levels, so fill them in pairs to avoid seeding an
            # artificial site imbalance in the diagonal-bath benchmark.
            n_occ_pairs = fld(n_occ_bath, 2)
            for p in 1:n_occ_pairs
                i = 2p - 1
                bath_state[i] = "Occ"
                bath_state[i + 1] = "Occ"
            end
            if isodd(n_occ_bath)
                bath_state[2n_occ_pairs + 1] = "Occ"
            end
        elseif bath_init == "sequential"
            for i in 1:n_occ_bath
                bath_state[i] = "Occ"
            end
        else
            throw(ArgumentError("unknown bath_init=$bath_init"))
        end

        # Impurity initial state
        if imp_init == "afm"
            imp_occ = clamp(round(Int, ρ * 2), 0, 2)
            if imp_occ == 0
                imp_state = "Emp"
            elseif imp_occ == 1
                imp_state = (x % 2 == 1) ? "Occ1" : "Occ2"  # AF: ↑ on site1, ↓ on site2
            else
                imp_state = "Occ12"
            end
        else
            imp_state = imp_init
        end

        # Build arm as MPS for bath sites
        if n_bath > 0
            bath_mps = MPS(phys_idx[x, 2:Ly], bath_state)
            [replacetags!(linkinds, bath_mps, "Link,l=$(y)", "FTNS,Arm,x=$(x),y=($(y+1)-$(y+2))") for y in 1:n_bath]
            Ts[x, 2:Ly] .= bath_mps
            aux_y_idx[x, 2:end] .= linkinds(bath_mps)
        end

        # Impurity tensor with arm link
        imp_tensor = state(phys_idx[x, 1], imp_state)
        if n_bath > 0
            # Create arm link between impurity and first bath
            if conserve_qns
                arm_link = Index([QN("Nf", 0, -1) => 1],
                    "FTNS,Arm,x=$(x),y=(1-2)"; dir=ITensors.Out)
            else
                arm_link = Index(1, "FTNS,Arm,x=$(x),y=(1-2)")
            end
            aux_y_idx[x, 1] = arm_link
            imp_tensor = imp_tensor * onehot(arm_link => 1)
            Ts[x, 2] = Ts[x, 2] * onehot(dag(arm_link) => 1)
        end
        Ts[x, 1] = imp_tensor
    end

    # Backbone auxiliary indices (dim=1 for product state)
    for x = 1:(Lx-1)
        if conserve_qns
            aux_x_idx[x] = Index([QN("Nf", 0, -1) => 1],
                "FTNS,Backbone,x=($(x)-$(x+1)),y=1"; dir=ITensors.Out)
        else
            aux_x_idx[x] = Index(1, "FTNS,Backbone,x=($(x)-$(x+1)),y=1")
        end
    end

    # Attach backbone indices to impurity tensors
    for x = 1:Lx
        if x == 1
            Ts[x, 1] = Ts[x, 1] * onehot(aux_x_idx[x] => 1)
        elseif x == Lx
            Ts[x, 1] = Ts[x, 1] * onehot(dag(aux_x_idx[x-1]) => 1)
        else
            Ts[x, 1] = Ts[x, 1] * onehot(aux_x_idx[x] => 1, dag(aux_x_idx[x-1]) => 1)
        end
    end

    return Ts, aux_x_idx, aux_y_idx
end


end # module ClusterAndersonImpurityModel
