module AndersonImpurityModel
using ITensors, ITensorMPS, Random

export ftns_initial_state, ftno_aim_model

function ftns_initial_state(phys_idx::Matrix{Index}, ρ::Float64; conserve_qns=true)

    Lx = size(phys_idx, 1)
    Ly = size(phys_idx, 2)

    Ts = Matrix{ITensor}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)

    # state = random_product_state(Ly, round(ρ * Ly))
    state = alternating_product_state(Ly, round(Int, ρ * Ly))
    # state = [i <= round(Int, ρ * Ly) ? "Occ" : "Emp" for i in 1:Ly]



    for x = 1:Lx
        product_state = MPS(phys_idx[x, :], state)
        [replacetags!(linkinds, product_state, "Link,l=$(y)", "FTNS,Arm,x=$(x),y=($(y)-$(y+1))") for y in 1:Ly]
        Ts[x, :] .= product_state
        aux_y_idx[x, :] .= linkinds(product_state)
    end


    for x = 1:(Lx-1)
        aux_x_idx[x] = Index([QN("Nf", 0, -1) => 1], "FTNS,Backbone,x=($(x)-$(x+1)),y=1"; dir=ITensors.Out)
        if !conserve_qns; aux_x_idx[x] = removeqns(aux_x_idx[x]); end
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

    Ws = Matrix{ITensor}(undef, Lx, Ly)

    phys_idx, aux_x_idx, aux_y_idx = ftno_indices(Lx, Ly; conserve_qns=conserve_qns)

    ftno_Ws_bath!(Ws, phys_idx, aux_y_idx, model_params)
    ftno_Ws_impurity!(Ws, phys_idx, aux_x_idx, aux_y_idx, model_params)

    return Ws, phys_idx, aux_x_idx, aux_y_idx

end


function ftno_Ws_bath!(Ws::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_y_idx::Matrix{Index}, model_params::Dict{String,Any})

    Lx = 2 * model_params["N_orb"]
    Ly = model_params["N_bath"] + 1
    εₖ = model_params["εₖ"]
    Vₖ = model_params["Vₖ"]

    for x = 1:2:Lx

        for y = 2:Ly
            if y < Ly
                Ws[x, y] = W_bath_spin_up(phys_idx[x, y], dag(aux_y_idx[x, y-1]), aux_y_idx[x, y], εₖ[x, y], Vₖ[x, y]; edge=false)
                Ws[x+1, y] = W_bath_spin_down(phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), aux_y_idx[x+1, y], εₖ[x+1, y], Vₖ[x+1, y]; edge=false)
            else
                Ws[x, y] = W_bath_spin_up(phys_idx[x, y], dag(aux_y_idx[x, y-1]), nothing, εₖ[x, y], Vₖ[x, y]; edge=true)
                Ws[x+1, y] = W_bath_spin_down(phys_idx[x+1, y], dag(aux_y_idx[x+1, y-1]), nothing, εₖ[x+1, y], Vₖ[x+1, y]; edge=true)
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

    for x = 1:2:Lx

        if x == 1
            Ws[x, 1] = W_imp1_spin_up(phys_idx[x, 1], aux_x_idx[x], aux_y_idx[x, 1], εₖ[x, 1])
            Ws[x+1, 1] = W_imp1_spin_down(phys_idx[x+1, 1], aux_x_idx[x+1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U)
        elseif x > 2 && x < Lx - 2
            Ws[x, 1] = W_imp2_spin_up(phys_idx[x, 1], aux_x_idx[x], dag(aux_x_idx[x-1]), aux_y_idx[x, 1], εₖ[x, 1], U′, J)
            Ws[x+1, 1] = W_imp2_spin_down(phys_idx[x+1, 1], aux_x_idx[x+1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U, U′, J)
        else
            Ws[x, 1] = W_imp3_spin_up(phys_idx[x, 1], aux_x_idx[x], dag(aux_x_idx[x-1]), aux_y_idx[x, 1], εₖ[x, 1], U′, J)
            Ws[x+1, 1] = W_imp3_spin_down(phys_idx[x+1, 1], dag(aux_x_idx[x]), aux_y_idx[x+1, 1], εₖ[x+1, 1], U, U′, J)
        end

    end

end



function ftno_indices(Lx::Int, Ly::Int; conserve_qns=true)

    phys_idx = Matrix{Index}(undef, Lx, Ly)
    aux_x_idx = Vector{Index}(undef, Lx - 1)
    aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)

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

            if x % 2 == 1
                aux_y_idx[x, y] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 3, QN(("Nf", -1, -1), ("Sz", -1)) => 1, QN(("Nf", +1, -1), ("Sz", +1)) => 1], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
            else
                aux_y_idx[x, y] = Index([QN(("Nf", 0, -1), ("Sz", 0)) => 2, QN(("Nf", -1, -1), ("Sz", +1)) => 1, QN(("Nf", +1, -1), ("Sz", -1)) => 1], "FTNO,Arm,x=$(x),y=($(y)-$(y+1))"; dir=ITensors.Out)
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
                aux_y_idx[x, y] = removeqns(aux_y_idx[x, y]); 
            end
        end

        for x=1:(Lx-1)
            aux_x_idx[x] = removeqns(aux_x_idx[x]); 
        end
    
    end

    return phys_idx, aux_x_idx, aux_y_idx

end


function W_imp1_spin_up(s::Index, d::Index, r::Index, ε::Float64)

    # HB
    W = onehot(d => 1, r => 1) * op("I", s)
    W += onehot(d => 1, r => 2) * op("N", s) * ε
    W += onehot(d => 1, r => 4) * op("c†", s)
    W += onehot(d => 1, r => 5) * op("c", s)

    W += onehot(d => 2, r => 2) * op("I", s)
    W += onehot(d => 3, r => 2) * op("N", s)
    W += onehot(d => 4, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 5, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)

    return W

end


function W_imp1_spin_down(s::Index, d::Index, u::Index, r::Index, ε::Float64, U::Float64)

    # Block 1
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * U
    W += onehot(d => 4, u => 2, r => 2) * op("N", s)

    # HB
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
    W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
    W += onehot(d => 1, u => 2, r => 4) * op("c", s)

    # Block 2
    W += onehot(d => 5, u => 4, r => 2) * op("c", s)
    W += onehot(d => 6, u => 4, r => 2) * op("c†", s)

    # Block 3
    W += onehot(d => 7, u => 5, r => 2) * op("c†", s)
    W += onehot(d => 8, u => 5, r => 2) * op("c", s)

    return W

end


function W_imp2_spin_up(s::Index, d::Index, u::Index, r::Index, ε::Float64, U′::Float64, J::Float64)

    # Block 1
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)
    W += onehot(d => 6, u => 5, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * U′
    W += onehot(d => 5, u => 2, r => 2) * op("N", s)

    # HB
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
    W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
    W += onehot(d => 1, u => 2, r => 5) * op("c", s)

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


function W_imp2_spin_down(s::Index, d::Index, u::Index, r::Index, ε::Float64, U::Float64, U′::Float64, J::Float64)

    # Block 1
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 3, u => 5, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * U′
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 5, r => 2) * op("N", s) * U
    W += onehot(d => 4, u => 2, r => 2) * op("N", s)

    # HB
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
    W += onehot(d => 1, u => 2, r => 3) * op("c†", s)
    W += onehot(d => 1, u => 2, r => 4) * op("c", s)

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


function W_imp3_spin_up(s::Index, d::Index, u::Index, r::Index, ε::Float64, U′::Float64, J::Float64)

    # Block 1
    W = onehot(d => 1, u => 1, r => 2) * op("I", s)
    W += onehot(d => 2, u => 2, r => 2) * op("I", s)
    W += onehot(d => 3, u => 3, r => 2) * op("I", s)
    W += onehot(d => 4, u => 4, r => 2) * op("I", s)

    W += onehot(d => 1, u => 3, r => 2) * op("N", s) * (U′ - J)
    W += onehot(d => 1, u => 4, r => 2) * op("N", s) * U′
    W += onehot(d => 5, u => 2, r => 2) * op("N", s)

    # HB
    W += onehot(d => 1, u => 2, r => 1) * op("I", s)
    W += onehot(d => 1, u => 2, r => 2) * op("N", s) * ε
    W += onehot(d => 1, u => 2, r => 4) * op("c†", s)
    W += onehot(d => 1, u => 2, r => 5) * op("c", s)

    # Block 2
    W += onehot(d => 6, u => 5, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 7, u => 6, r => 3) * swapprime(op("c", s)' * op("F", s), 2 => 1)
    W += onehot(d => 8, u => 7, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)
    W += onehot(d => 9, u => 8, r => 3) * swapprime(op("c†", s)' * op("F", s), 2 => 1)

    return W

end


function W_imp3_spin_down(s::Index, u::Index, r::Index, ε::Float64, U::Float64, U′::Float64, J::Float64)

    # Block 1
    W = onehot(u => 1, r => 2) * op("I", s)

    # HB
    W += onehot(u => 2, r => 1) * op("I", s)
    W += onehot(u => 2, r => 2) * op("N", s) * ε
    W += onehot(u => 2, r => 3) * op("c†", s)
    W += onehot(u => 2, r => 4) * op("c", s)

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


function W_bath_spin_up(s::Index, l::Index, r::Union{Index,Nothing}, ε::Float64, V::Float64; edge::Bool=false)

    if edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("F", s)
        W += onehot(l => 4) * op("c", s) * V
        W += onehot(l => 5) * op("c†", s) * V

    else

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 5, r => 5) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 4, r => 2) * op("c", s) * V
        W += onehot(l => 5, r => 2) * op("c†", s) * V

    end

    return W
end


function W_bath_spin_down(s::Index, l::Index, r::Union{Index,Nothing}, ε::Float64, V::Float64; edge::Bool=false)

    if edge

        W = onehot(l => 1) * op("N", s) * ε
        W += onehot(l => 2) * op("I", s)
        W += onehot(l => 3) * op("c", s) * V
        W += onehot(l => 4) * op("c†", s) * V

    else

        W = onehot(l => 1, r => 1) * op("I", s)
        W += onehot(l => 2, r => 2) * op("I", s)
        W += onehot(l => 3, r => 3) * op("F", s)
        W += onehot(l => 4, r => 4) * op("F", s)
        W += onehot(l => 1, r => 2) * op("N", s) * ε
        W += onehot(l => 3, r => 2) * op("c", s) * V
        W += onehot(l => 4, r => 2) * op("c†", s) * V

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


function random_product_state(L, n)
    arr = fill("Emp", L)
    indices = randperm(L)[1:n]
    for idx in indices
        arr[idx] = "Occ"
    end
    return arr
end


end # module AndersonImpurityModel