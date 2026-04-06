module AndersonImpurityModel
using ITensors, ITensorMPS, Random

export ftns_initial_state, ftno_aim_model


function _double_chain_counts_from_bath(εₖ::AbstractMatrix{<:Real})
    bath_energies = vec(εₖ[1, 2:end])
    N_occ = count(ε -> ε <= 0.0, bath_energies)
    return N_occ, length(bath_energies) - N_occ
end


function _double_chain_bath_sequence(N_occ::Int, N_emp::Int)
    seq = Symbol[]
    N_common = min(N_occ, N_emp)

    for _ in 1:N_common
        push!(seq, :occ)
        push!(seq, :emp)
    end

    if N_occ > N_common
        append!(seq, fill(:occ, N_occ - N_common))
    elseif N_emp > N_common
        append!(seq, fill(:emp, N_emp - N_common))
    end

    return seq
end


function _double_chain_site_kinds(N_bath::Int;
    N_occ::Union{Nothing,Int}=nothing, N_emp::Union{Nothing,Int}=nothing)
    (N_occ !== nothing && N_emp !== nothing) ||
        throw(ArgumentError("double chain requires N_occ and N_emp"))
    (0 <= N_occ <= N_bath && 0 <= N_emp <= N_bath && N_occ + N_emp == N_bath) ||
        throw(ArgumentError("double chain counts must satisfy N_occ + N_emp = N_bath; got N_occ=$N_occ, N_emp=$N_emp, N_bath=$N_bath"))

    kinds = Vector{Symbol}(undef, N_bath)
    if N_occ == N_emp
        for i in 1:N_bath
            kinds[i] = i < N_bath ? :dc_full_bulk : :dc_full_edge
        end
        return kinds
    end

    N_common = min(N_occ, N_emp)
    if N_common == 0
        for i in 1:N_bath
            kinds[i] = i < N_bath ? :dc_only_bulk : :dc_only_edge
        end
        return kinds
    end

    transition_idx = N_occ < N_emp ? (2 * N_common - 1) : (2 * N_common)

    for i in 1:N_bath
        if i < transition_idx
            kinds[i] = :dc_full_bulk
        elseif i == transition_idx
            kinds[i] = :dc_transition
        elseif i < N_bath
            kinds[i] = :dc_only_bulk
        else
            kinds[i] = :dc_only_edge
        end
    end

    return kinds
end


function _dc_counts(model_params::Dict{String,Any})
    if get(model_params, "Geometry", "star") != "double chain"
        return nothing, nothing
    end
    if haskey(model_params, "N_occ") && haskey(model_params, "N_emp")
        return Int(model_params["N_occ"]), Int(model_params["N_emp"])
    end

    if haskey(model_params, "εₖ")
        return _double_chain_counts_from_bath(model_params["εₖ"])
    end

    error("double chain requires N_occ/N_emp or εₖ in model_params")
end


function _dc_aux_index(x::Int, y::Int, spin::Symbol, dim_kind::Symbol)
    tag = "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"
    if spin == :up
        if dim_kind == :full
            return Index([QN(("Nf", 0, -1), ("Sz", 0)) => 3,
                QN(("Nf", -1, -1), ("Sz", -1)) => 1,
                QN(("Nf", +1, -1), ("Sz", +1)) => 1,
                QN(("Nf", -1, -1), ("Sz", -1)) => 1,
                QN(("Nf", +1, -1), ("Sz", +1)) => 1], tag; dir=ITensors.Out)
        end
        return Index([QN(("Nf", 0, -1), ("Sz", 0)) => 3,
            QN(("Nf", -1, -1), ("Sz", -1)) => 1,
            QN(("Nf", +1, -1), ("Sz", +1)) => 1], tag; dir=ITensors.Out)
    end

    if dim_kind == :full
        return Index([QN(("Nf", 0, -1), ("Sz", 0)) => 2,
            QN(("Nf", -1, -1), ("Sz", +1)) => 1,
            QN(("Nf", +1, -1), ("Sz", -1)) => 1,
            QN(("Nf", -1, -1), ("Sz", +1)) => 1,
            QN(("Nf", +1, -1), ("Sz", -1)) => 1], tag; dir=ITensors.Out)
    end
    return Index([QN(("Nf", 0, -1), ("Sz", 0)) => 2,
        QN(("Nf", -1, -1), ("Sz", +1)) => 1,
        QN(("Nf", +1, -1), ("Sz", -1)) => 1], tag; dir=ITensors.Out)
end

function ftns_initial_state(phys_idx::AbstractMatrix{<:Index}, ρ::Float64;
    conserve_qns=true, geometry::String="star",
    N_occ::Union{Nothing,Int}=nothing, N_emp::Union{Nothing,Int}=nothing)

    Lx = size(phys_idx, 1)
    Ly = size(phys_idx, 2)

    Ts = Matrix{ITensor}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)

    n_occ = if geometry == "double chain"
        N_occ !== nothing || throw(ArgumentError("double chain seed requires N_occ"))
        clamp(Int(N_occ) + 1, 0, Ly)
    else
        clamp(round(Int, ρ * Ly), 0, Ly)
    end
    state = if geometry == "double chain"
        double_chain_product_state(Ly, n_occ; N_occ=N_occ, N_emp=N_emp)
    else
        sequential_product_state(Ly, n_occ)
    end



    for x = 1:Lx
        product_state = MPS(phys_idx[x, :], state)
        [replacetags!(linkinds, product_state, "Link,l=$(y)", "FTNS,Arm,x=$(x),y=($(y)-$(y+1))") for y in 1:Ly]
        Ts[x, :] .= product_state
        aux_y_idx[x, :] .= linkinds(product_state)
    end


    for x = 1:(Lx-1)
        aux_x_idx[x] = Index([QN("Nf", 0, -1) => 1], "FTNS,Backbone,x=($(x)-$(x+1)),y=1"; dir=ITensors.Out)
        if !conserve_qns
            aux_x_idx[x] = removeqns(aux_x_idx[x])
        end
    end


    for x = 1:Lx
        if x == 1
            Ts[x, 1] = Ts[x, 1] * onehot(aux_x_idx[x] => 1)
        elseif x == Lx
            Ts[x, 1] = Ts[x, 1] * onehot(dag(aux_x_idx[x-1]) => 1)
        else
            Ts[x, 1] = Ts[x, 1] * onehot(aux_x_idx[x] => 1, dag(aux_x_idx[x-1]) => 1)
        end
    end

    return Ts, aux_x_idx, aux_y_idx, state

end


function ftno_aim_model(model_params::Dict{String,Any})

    Lx = 2 * model_params["N_orb"]
    Ly = model_params["N_bath"] + 1
    conserve_qns = model_params["conserve_qns"]
    Geometry = model_params["Geometry"]
    N_occ, N_emp = _dc_counts(model_params)

    Ws = Matrix{ITensor}(undef, Lx, Ly)

    phys_idx, aux_x_idx, aux_y_idx = ftno_indices(Lx, Ly, Geometry;
        conserve_qns=conserve_qns, N_occ=N_occ, N_emp=N_emp)

    ftno_Ws_bath!(Ws, phys_idx, aux_y_idx, model_params)
    ftno_Ws_impurity!(Ws, phys_idx, aux_x_idx, aux_y_idx, model_params)

    return Ws, phys_idx, aux_x_idx, aux_y_idx

end


function ftno_Ws_bath!(Ws::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_y_idx::Matrix{Index}, model_params::Dict{String,Any})

    Lx = 2 * model_params["N_orb"]
    Ly = model_params["N_bath"] + 1
    εₖ = model_params["εₖ"]
    Vₖ = model_params["Vₖ"]
    Geometry = model_params["Geometry"]
    N_occ, N_emp = _dc_counts(model_params)
    dc_kinds = Geometry == "double chain" ?
        _double_chain_site_kinds(model_params["N_bath"]; N_occ=N_occ, N_emp=N_emp) :
        Symbol[]

    for x = 1:2:Lx

        for y = 2:Ly
            if Geometry == "double chain"
                kind = dc_kinds[y - 1]
                has_right = kind ∉ (:dc_full_edge, :dc_only_edge)
                r_up = has_right ? aux_y_idx[x, y] : nothing
                r_dn = has_right ? aux_y_idx[x+1, y] : nothing
                Ws[x, y] = W_bath_spin_up(phys_idx[x, y], dag(aux_y_idx[x, y-1]), r_up,
                    εₖ[x, y], Vₖ[x, y], Geometry; kind=kind)
                Ws[x+1, y] = W_bath_spin_down(phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), r_dn,
                    εₖ[x+1, y], Vₖ[x+1, y], Geometry; kind=kind)
            elseif y < Ly
                Ws[x, y] = W_bath_spin_up(phys_idx[x, y], dag(aux_y_idx[x, y-1]), aux_y_idx[x, y], εₖ[x, y], Vₖ[x, y], Geometry; edge=false)
                Ws[x+1, y] = W_bath_spin_down(phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), aux_y_idx[x+1, y], εₖ[x+1, y], Vₖ[x+1, y], Geometry; edge=false)
            else
                Ws[x, y] = W_bath_spin_up(phys_idx[x, y], dag(aux_y_idx[x, y-1]), nothing, εₖ[x, y], Vₖ[x, y], Geometry; edge=true)
                Ws[x+1, y] = W_bath_spin_down(phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), nothing, εₖ[x+1, y], Vₖ[x+1, y], Geometry; edge=true)
            end
        end
    end

end


function ftno_Ws_impurity!(Ws::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index}, model_params::Dict{String,Any})

    Lx = 2 * model_params["N_orb"]
    εₖ = model_params["εₖ"]
    U = model_params["U"]
    U′ = model_params["U′"]
    J = model_params["J"]
    Geometry = model_params["Geometry"]

    for x = 1:2:Lx

        if x == 1
            Ws[x, 1] = W_imp1_spin_up(phys_idx[x, 1], aux_x_idx[x], aux_y_idx[x, 1], εₖ[x, 1], Geometry)
            Ws[x+1, 1] = W_imp1_spin_down(phys_idx[x+1, 1], aux_x_idx[x+1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U, Geometry)
        elseif x > 2 && x < Lx - 2
            Ws[x, 1] = W_imp2_spin_up(phys_idx[x, 1], aux_x_idx[x], dag(aux_x_idx[x-1]), aux_y_idx[x, 1], εₖ[x, 1], U′, J, Geometry)
            Ws[x+1, 1] = W_imp2_spin_down(phys_idx[x+1, 1], aux_x_idx[x+1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U, U′, J, Geometry)
        else
            Ws[x, 1] = W_imp3_spin_up(phys_idx[x, 1], aux_x_idx[x], dag(aux_x_idx[x-1]), aux_y_idx[x, 1], εₖ[x, 1], U′, J, Geometry)
            Ws[x+1, 1] = W_imp3_spin_down(phys_idx[x+1, 1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U, U′, J, Geometry)
        end

    end

end



function ftno_indices(Lx::Int, Ly::Int, Geometry::String; conserve_qns=true,
    N_occ::Union{Nothing,Int}=nothing,
    N_emp::Union{Nothing,Int}=nothing)

    phys_idx = Matrix{Index}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)
    dc_kinds = Geometry == "double chain" ?
        _double_chain_site_kinds(Ly - 1; N_occ=N_occ, N_emp=N_emp) :
        Symbol[]

    for x = 1:Lx

        if x % 2 == 1 && conserve_qns
            sites = siteinds("Fermion", Ly; conserve_qns=true, conserve_sz="Up")
        elseif x % 2 == 0 && conserve_qns
            sites = siteinds("Fermion", Ly; conserve_qns=true, conserve_sz="Dn")
        else
            sites = siteinds("Fermion", Ly; conserve_qns=false)
        end
        sites = [replacetags(sites[y], "n=$(y)" => "x=$(x),y=$(y)") for y in 1:Ly]
        phys_idx[x, :] .= sites

        for y = 1:(Ly-1)

            if Geometry == "star"

                if x % 2 == 1
                    aux_y_idx[x, y] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 3, QN(("Nf", -1, -1), ("Sz", -1)) => 1, QN(("Nf", +1, -1), ("Sz", +1)) => 1], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
                else
                    aux_y_idx[x, y] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 2, QN(("Nf", -1, -1), ("Sz", +1)) => 1, QN(("Nf", +1, -1), ("Sz", -1)) => 1], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
                end

            elseif Geometry == "double chain"

                dim_kind = if y == 1
                    :full
                elseif dc_kinds[y - 1] in (:dc_transition, :dc_only_bulk)
                    :only
                else
                    :full
                end
                spin = isodd(x) ? :up : :down
                aux_y_idx[x, y] = _dc_aux_index(x, y, spin, dim_kind)

            else
                error("AIM FTNO not implemented for this geometry.")
            end

        end
    end

    aux_x_idx[1] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 3, QN(("Nf", -1, -1), ("Sz", -1)) => 1, QN(("Nf", 1, -1), ("Sz", 1)) => 1], "FTNO,Backbone,x=(1-2),y=1"; dir=ITensors.Out)
    aux_x_idx[2] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 4, QN(("Nf", 0, -1), ("Sz", -2)) => 1, QN(("Nf", -2, -1), ("Sz", 0)) => 1, QN(("Nf", 0, -1), ("Sz", 2)) => 1, QN(("Nf", 2, -1), ("Sz", 0)) => 1], "FTNO,Backbone,x=(2-3),y=1"; dir=ITensors.Out)
    for x = 3:2:(Lx-3)
        aux_x_idx[x] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 5, QN(("Nf", 0, -1), ("Sz", -2)) => 1, QN(("Nf", -2, -1), ("Sz", 0)) => 1, QN(("Nf", 0, -1), ("Sz", 2)) => 1, QN(("Nf", 2, -1), ("Sz", 0)) => 1, QN(("Nf", -1, -1), ("Sz", -1)) => 1, QN(("Nf", 1, -1), ("Sz", 1)) => 1, QN(("Nf", 1, -1), ("Sz", -1)) => 1, QN(("Nf", -1, -1), ("Sz", 1)) => 2, QN(("Nf", 1, -1), ("Sz", -1)) => 1], "FTNO,Backbone,x=($(x)-$(x+1)),y=1"; dir=ITensors.Out)
        aux_x_idx[x+1] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 4, QN(("Nf", 0, -1), ("Sz", -2)) => 1, QN(("Nf", -2, -1), ("Sz", 0)) => 1, QN(("Nf", 0, -1), ("Sz", 2)) => 1, QN(("Nf", 2, -1), ("Sz", 0)) => 1], "FTNO,Backbone,x=($(x+1)-$(x+2)),y=1"; dir=ITensors.Out)
    end
    aux_x_idx[Lx-1] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 5, QN(("Nf", 1, -1), ("Sz", -1)) => 1, QN(("Nf", -1, -1), ("Sz", 1)) => 2, QN(("Nf", 1, -1), ("Sz", -1)) => 1], "FTNO,Backbone,x=($(Lx-1)-$(Lx)),y=1"; dir=ITensors.Out)


    if !conserve_qns
        for x = 1:Lx
            for y = 1:(Ly-1)
                aux_y_idx[x, y] = removeqns(aux_y_idx[x, y])
            end
        end

        for x = 1:(Lx-1)
            aux_x_idx[x] = removeqns(aux_x_idx[x])
        end

    end

    return phys_idx, aux_x_idx, aux_y_idx

end


function W_imp1_spin_up(s::Index, d::Index, r::Index, ε::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(d => 1, r => 1) * op("I", s)
        W += onehot(d => 1, r => 2) * op("N", s) * ε
        W += onehot(d => 1, r => 4) * op("c†", s)
        W += onehot(d => 1, r => 5) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(d => 1, r => 1) * op("I", s)
        W += onehot(d => 1, r => 2) * op("N", s) * ε
        W += onehot(d => 1, r => 4) * op("c†", s)
        W += onehot(d => 1, r => 5) * op("c", s)
        W += onehot(d => 1, r => 6) * op("c†", s)
        W += onehot(d => 1, r => 7) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    W += onehot(d => 2, r => 2) * op("I", s)
    W += onehot(d => 3, r => 2) * op("N", s)
    W += onehot(d => 4, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 5, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)

    return W

end


function W_imp1_spin_down(s::Index, d::Index, u::Index, r::Index, ε::Float64, U::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 4) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 4) * op("c", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 6) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    # Block 1
    W += onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * U
    W += onehot(d => 4, u => 2, r => 2) * op("N", s)

    # Block 2
    W += onehot(d => 5, u => 4, r => 2) * op("c", s)
    W += onehot(d => 6, u => 4, r => 2) * op("c†", s)

    # Block 3
    W += onehot(d => 7, u => 5, r => 2) * op("c†", s)
    W += onehot(d => 8, u => 5, r => 2) * op("c", s)

    return W

end


function W_imp2_spin_up(s::Index, d::Index, u::Index, r::Index, ε::Float64, U′::Float64, J::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c", s)
        W += onehot(d => 1, u => 2, r => 6) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 7) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    # Block 1
    W += onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)
    W += onehot(d => 6, u => 5, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * U′
    W += onehot(d => 5, u => 2, r => 2) * op("N", s)

    # Block 2
    W += onehot(d => 7, u => 6, r => 2) * op("I", s)
    W += onehot(d => 8, u => 7, r => 2) * op("I", s)
    W += onehot(d => 9, u => 8, r => 2) * op("I", s)

    # Block 3
    W += onehot(d => 10, u => 2, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 11, u => 2, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)

    # Block 4
    W += onehot(d => 12, u => 5, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 13, u => 6, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 14, u => 7, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 15, u => 8, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)

    return W

end


function W_imp2_spin_down(s::Index, d::Index, u::Index, r::Index, ε::Float64, U::Float64, U′::Float64, J::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 4) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 4) * op("c", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 6) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    # Block 1
    W += onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 3, u => 5, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * U′
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 5, r => 2) * op("N", s) * U
    W += onehot(d => 4, u => 2, r => 2) * op("N", s)

    # Block 2
    W += onehot(d => 1, u => 12, r => 2) * op("c†", s) * J
    W += onehot(d => 1, u => 13, r => 2) * op("c", s) * (-J)
    W += onehot(d => 1, u => 14, r => 2) * op("c", s) * J
    W += onehot(d => 1, u => 15, r => 2) * op("c†", s) * (-J)

    # Block 3
    W += onehot(d => 5, u => 6, r => 2) * op("I", s)
    W += onehot(d => 6, u => 7, r => 2) * op("I", s)
    W += onehot(d => 7, u => 8, r => 2) * op("I", s)
    W += onehot(d => 8, u => 9, r => 2) * op("I", s)

    # Block 4
    W += onehot(d => 5, u => 10, r => 2) * op("c", s)
    W += onehot(d => 6, u => 10, r => 2) * op("c†", s)

    # Block 5
    W += onehot(d => 7, u => 11, r => 2) * op("c†", s)
    W += onehot(d => 8, u => 11, r => 2) * op("c", s)

    return W

end


function W_imp3_spin_up(s::Index, d::Index, u::Index, r::Index, ε::Float64, U′::Float64, J::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(d => 1, u => 2, r => 1) * op("I", s)
        W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
        W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 5) * op("c", s)
        W += onehot(d => 1, u => 2, r => 6) * op("c†", s)
        W += onehot(d => 1, u => 2, r => 7) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    # Block 1
    W += onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * U′
    W += onehot(d => 5, u => 2, r => 2) * op("N", s)

    # Block 2
    W += onehot(d => 6, u => 5, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 7, u => 6, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 8, u => 7, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 9, u => 8, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)

    return W

end


function W_imp3_spin_down(s::Index, u::Index, r::Index, ε::Float64, U::Float64, U′::Float64, J::Float64, Geometry::String)

    # HB
    if Geometry == "star"

        W = onehot(u => 2, r => 1) * op("I", s)
        W += onehot(u => 2, r => 2) * op("N", s) * ε
        W += onehot(u => 2, r => 3) * op("c†", s)
        W += onehot(u => 2, r => 4) * op("c", s)

    elseif Geometry == "double chain"

        W = onehot(u => 2, r => 1) * op("I", s)
        W += onehot(u => 2, r => 2) * op("N", s) * ε
        W += onehot(u => 2, r => 3) * op("c†", s)
        W += onehot(u => 2, r => 4) * op("c", s)
        W += onehot(u => 2, r => 5) * op("c†", s)
        W += onehot(u => 2, r => 6) * op("c", s)

    else
        error("AIM FTNO not implemented for this geometry.")
    end

    # Block 1
    W += onehot(u => 1, r => 2) * op("I", s)
    W += onehot(u => 3, r => 2) * op("N", s) * U′
    W += onehot(u => 4, r => 2) * op("N", s) * (U′ - J)
    W += onehot(u => 5, r => 2) * op("N", s) * U

    # Block 2
    W += onehot(u => 6, r => 2) * op("c†", s) * J
    W += onehot(u => 7, r => 2) * op("c", s) * (-J)
    W += onehot(u => 8, r => 2) * op("c", s) * J
    W += onehot(u => 9, r => 2) * op("c†", s) * (-J)

    return W

end


function W_bath_spin_up(s::Index, l::Index, r::Union{Index,Nothing}, ε::Float64, V::Float64, Geometry::String;
    edge::Bool=false, kind::Symbol=:auto)
    kind = kind == :auto ? (Geometry == "star" ? (edge ? :star_edge : :star_bulk) : (edge ? :dc_full_edge : :dc_full_bulk)) : kind

    if kind == :star_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("F", s)
        W += onehot(l => 4) * op("c", s) * V
        W += onehot(l => 5) * op("c†", s) * V

    elseif kind == :star_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 5, r => 5) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 4, r => 2) * op("c", s) * V
        W += onehot(l => 5, r => 2) * op("c†", s) * V

    elseif kind == :dc_full_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("F", s)
        W += onehot(l => 6) * op("c", s) * V
        W += onehot(l => 7) * op("c†", s) * V

    elseif kind == :dc_full_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 6) * op("F", s)
        W += onehot(l => 5, r => 7) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 1, r => 4) * op("c†", s)
        W += onehot(l => 1, r => 5) * op("c", s)
        W += onehot(l => 6, r => 2) * op("c", s) * V
        W += onehot(l => 7, r => 2) * op("c†", s) * V

    elseif kind == :dc_transition

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 5, r => 5) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 6, r => 2) * op("c", s) * V
        W += onehot(l => 7, r => 2) * op("c†", s) * V

    elseif kind == :dc_only_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("F", s)
        W += onehot(l => 4) * op("c", s) * V
        W += onehot(l => 5) * op("c†", s) * V

    elseif kind == :dc_only_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 1, r => 4) * op("c†", s)
        W += onehot(l => 1, r => 5) * op("c", s)
        W += onehot(l => 4, r => 2) * op("c", s) * V
        W += onehot(l => 5, r => 2) * op("c†", s) * V

    else
        error("AIM FTNO bath tensor not implemented for kind=$kind")
    end

    return W
end


function W_bath_spin_down(s::Index, l::Index, r::Union{Index,Nothing}, ε::Float64, V::Float64, Geometry::String;
    edge::Bool=false, kind::Symbol=:auto)
    kind = kind == :auto ? (Geometry == "star" ? (edge ? :star_edge : :star_bulk) : (edge ? :dc_full_edge : :dc_full_bulk)) : kind

    if kind == :star_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("c", s) * V
        W += onehot(l => 4) * op("c†", s) * V

    elseif kind == :star_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 3, r => 2) * op("c", s) * V
        W += onehot(l => 4, r => 2) * op("c†", s) * V

    elseif kind == :dc_full_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 5) * op("c", s) * V
        W += onehot(l => 6) * op("c†", s) * V

    elseif kind == :dc_full_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 5) * op("F", s)
        W += onehot(l => 4, r => 6) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 1, r => 3) * op("c†", s)
        W += onehot(l => 1, r => 4) * op("c", s)
        W += onehot(l => 5, r => 2) * op("c", s) * V
        W += onehot(l => 6, r => 2) * op("c†", s) * V

    elseif kind == :dc_transition

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 5, r => 2) * op("c", s) * V
        W += onehot(l => 6, r => 2) * op("c†", s) * V

    elseif kind == :dc_only_edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("c", s) * V
        W += onehot(l => 4) * op("c†", s) * V

    elseif kind == :dc_only_bulk

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 1, r => 3) * op("c†", s)
        W += onehot(l => 1, r => 4) * op("c", s)
        W += onehot(l => 3, r => 2) * op("c", s) * V
        W += onehot(l => 4, r => 2) * op("c†", s) * V

    else
        error("AIM FTNO bath tensor not implemented for kind=$kind")
    end

    return W
end


function alternating_product_state(L, n)
    if 2n > L
        throw(ArgumentError("2n cannot be greater than L"))
    end

    # 벡터 초기화
    vector = Vector{String}(undef, L)

    # 첫 2n 개의 원소를 교차로 "Occ", "Emp"로 채움
    for i in 1:n
        vector[2i-1] = "Occ"
        vector[2i] = "Emp"
    end

    # 나머지 원소 처리
    remaining_occurrences = n
    start_index = 2n + 1

    # 남은 "Occ"를 먼저 채우고 나머지는 "Emp"
    for i in start_index:L
        if remaining_occurrences > 0
            vector[i] = "Occ"
            remaining_occurrences -= 1
        else
            vector[i] = "Emp"
        end
    end

    return vector
end


function sequential_product_state(L::Int, n::Int)
    0 <= n <= L || throw(ArgumentError("occupation count n must satisfy 0 <= n <= L"))
    state = fill("Emp", L)
    for i in 1:n
        state[i] = "Occ"
    end
    return state
end


function double_chain_product_state(L::Int, n::Int;
    N_occ::Union{Nothing,Int}=nothing, N_emp::Union{Nothing,Int}=nothing)
    0 <= n <= L || throw(ArgumentError("occupation count n must satisfy 0 <= n <= L"))

    state = fill("Emp", L)
    n == 0 && return state

    (N_occ !== nothing && N_emp !== nothing) ||
        throw(ArgumentError("double chain product state requires N_occ and N_emp"))
    seq = _double_chain_bath_sequence(N_occ, N_emp)
    length(seq) == L - 1 ||
        throw(ArgumentError("double chain sequence length $(length(seq)) does not match bath length $(L - 1)"))
    fill_order = Int[1]
    append!(fill_order, (findall(isequal(:occ), seq) .+ 1))
    append!(fill_order, (findall(isequal(:emp), seq) .+ 1))

    for idx in fill_order[1:n]
        state[idx] = "Occ"
    end

    return state
end


function random_product_state(L, n)
    arr = fill("Emp", L)
    indices = randperm(L)[1:n]
    for idx in indices
        arr[idx] = "Occ"
    end
    return arr
end


end # module AndersonImpurityModel
