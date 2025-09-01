"""
    Struct: ForkTensorNetworkState
    Representing the Fork Tensor Network State
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
mutable struct ForkTensorNetworkState

    Lx::Int                                                    # Number of Sites in the x-direction
    Ly::Int                                                    # Number of Sites in the y-direction
    χˣ::Union{Integer,Nothing}                                # Maximum bond dimension along the x-direction
    χʸ::Union{Integer,Nothing}                                # Maximum bond dimension along the y-direction
    canonical_center::Union{Tuple{Integer,Integer},Nothing}    # Canonical center
    network_matrix::Matrix{Integer}                            # Network Matrix
    coordinates::Matrix{Integer}                               # Coordinates of the sites
    sites::Matrix{Integer}                                     # Site indices
    phys_idx::Matrix{Index}                                    # Physical Index
    aux_x_idx::Vector{Index}                                   # Auxiliary Index in the x-direction
    aux_y_idx::Matrix{Index}                                   # Auxiliary Index in the y-direction
    Ts::Matrix{ITensor}                                        # List of Tensors


    """
        ForkTensorNetworkState(Lx::Int, Ly::Int, phys_idx::Vector{Index}, χˣ::Int, χʸ::Int)
        Constructor for the ForkTensorNetworkState
    """
    function ForkTensorNetworkState(Lx::T, Ly::T, phys_idx::Vector{Index{T}}; χˣ=nothing, χʸ=nothing) where {T<:Integer}
        ψ = new(
            Lx,                                         # Number of Sites in the x-direction
            Ly,                                         # Number of Sites in the y-direction
            χˣ,                                         # Maximum bond dimension along the x-direction
            χʸ,                                         # Maximum bond dimension along the y-direction
            nothing,                                    # Canonical center
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),   # Network Matrix 
            Matrix{Integer}(undef, Lx * Ly, 2),         # Coordinates of the sites 
            Matrix{Integer}(undef, Lx, Ly),             # Site indices 
            Matrix{Index}(undef, Lx, Ly),               # Physical Index 
            Vector{Index}(undef, Lx - 1),               # Auxiliary Index in the x-direction 
            Matrix{Index}(undef, Lx, Ly - 1),           # Auxiliary Index in the y-direction 
            Matrix{ITensor}(undef, Lx, Ly)              # List of Tensors 
        )
        create_fork_graph_matrix!(ψ)
        initialize_indices!(ψ, phys_idx)
        initialize_tensors_random!(ψ)
        return ψ
    end


    """
    ForkTensorNetworkState(Lx::Int, Ly::Int, phys_idx::Matrix{Index}, χˣ::Int, χʸ::Int)
    Constructor for the ForkTensorNetworkState
    """
    function ForkTensorNetworkState(Lx::T, Ly::T, phys_idx::Matrix{Index{T}}; χˣ=nothing, χʸ=nothing) where {T<:Integer}
        ψ = new(
            Lx,                                         # Number of Sites in the x-direction
            Ly,                                         # Number of Sites in the y-direction
            χˣ,                                         # Maximum bond dimension along the x-direction
            χʸ,                                         # Maximum bond dimension along the y-direction
            nothing,                                    # Canonical center
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),   # Network Matrix
            Matrix{Integer}(undef, Lx * Ly, 2),         # Coordinates of the sites
            Matrix{Integer}(undef, Lx, Ly),             # Site indices
            phys_idx,                                   # Physical Index
            Vector{Index}(undef, Lx - 1),               # Auxiliary Index in the x-direction
            Matrix{Index}(undef, Lx, Ly - 1),           # Auxiliary Index in the y-direction
            Matrix{ITensor}(undef, Lx, Ly)              # List of Tensors
        )
        create_fork_graph_matrix!(ψ)
        initialize_indices!(ψ)
        initialize_tensors_random!(ψ)
        return ψ
    end


    """
    ForkTensorNetworkState(s::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index}, χˣ::Int, χʸ::Int)
    Constructor for the ForkTensorNetworkState
    """
    function ForkTensorNetworkState(Ts::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index}; χˣ=nothing, χʸ=nothing)
        Lx = size(Ts, 1)
        Ly = size(Ts, 2)
        ψ = new(
            Lx,                                           # Number of Sites in the x-direction 
            Ly,                                           # Number of Sites in the y-direction 
            χˣ,                                           # Maximum bond dimension along the x-direction 
            χʸ,                                           # Maximum bond dimension along the y-direction 
            nothing,                                      # Canonical center 
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),     # Network Matrix  
            Matrix{Integer}(undef, Lx * Ly, 2),           # Coordinates of the sites  
            Matrix{Integer}(undef, Lx, Ly),               # Site indices  
            phys_idx,                                     # Physical Index  
            aux_x_idx,                                    # Auxiliary Index in the x-direction  
            aux_y_idx,                                    # Auxiliary Index in the y-direction  
            Ts                                            # List of Tensors  
        )
        create_fork_graph_matrix!(ψ)
        return ψ
    end

end # struct ForkTensorNetworkState


"""
    initialize_tensors_random!(ψ::ForkTensorNetworkState)
    Initializes the tensors with random elements
"""
function initialize_tensors_random!(ψ::ForkTensorNetworkState)

    for x = 1:ψ.Lx
        for y = 1:ψ.Ly

            if y == ψ.Ly
                ψ.Ts[x, y] = random_itensor(ψ.aux_y_idx[x, y-1], ψ.phys_idx[x, y])
            elseif y == 1 && x == 1
                ψ.Ts[x, y] = random_itensor(ψ.aux_y_idx[x, y], ψ.aux_x_idx[x], ψ.phys_idx[x, y])
            elseif y == 1 && x == ψ.Lx
                ψ.Ts[x, y] = random_itensor(ψ.aux_y_idx[x, y], ψ.aux_x_idx[x-1], ψ.phys_idx[x, y])
            elseif y == 1 && x > 1 && x < ψ.Lx
                ψ.Ts[x, y] = random_itensor(ψ.aux_y_idx[x, y], ψ.aux_x_idx[x-1], ψ.aux_x_idx[x], ψ.phys_idx[x, y])
            else
                ψ.Ts[x, y] = random_itensor(ψ.aux_y_idx[x, y-1], ψ.aux_y_idx[x, y], ψ.phys_idx[x, y])
            end

        end
    end

end


"""
    initialize_indices!(ψ::ForkTensorNetworkState)
    Initializes the indices of the tensors
"""
function initialize_indices!(ψ::ForkTensorNetworkState, phys_idx::Union{Nothing,Vector{Index{T}}}=nothing) where {T<:Integer}

    if phys_idx !== nothing
        for x = 1:ψ.Lx
            for y = 1:ψ.Ly
                ψ.phys_idx[x, y] = phys_idx[(x-1)*ψ.Ly+y]
            end
        end
    end

    for x = 1:ψ.Lx
        for y = 1:(ψ.Ly-1)
            ψ.aux_y_idx[x, y] = Index(ψ.χʸ; tags="Arm,x=$(x),y=($(y)-$(y+1))")
        end
    end

    for x = 1:(ψ.Lx-1)
        ψ.aux_x_idx[x] = Index(ψ.χˣ; tags="Backbone,x=($(x)-$(x+1)),y=1")
    end

end


"""
    plot_network(ψ::ForkTensorNetworkState)
    Plots the network
"""
function plot_network(ψ::ForkTensorNetworkState)

    g = ψ.network_matrix
    xs = ψ.coordinates[:, 2]         # to transpose the position
    ys = -ψ.coordinates[:, 1]        # to transpose the position
    color = [1 for i in 1:length(xs)]

    if ψ.canonical_center !== nothing
        color[ψ.sites[ψ.canonical_center[1], ψ.canonical_center[2]]] = 2
    end

    name = ["($(ψ.coordinates[i,1]), $(ψ.coordinates[i,2]))" for i in 1:length(xs)]
    # name = ["$(ψ.sites[i])" for i in 1:(ψ.Lx*ψ.Ly)]
    graphplot(g, x=xs, y=ys, markersize=0.5, markercolor=color, names=name)

end


"""
    create_fork_graph_matrix!(ψ::ForkTensorNetworkState)
    Creates the graph matrix for the Fork Tensor Network
"""
function create_fork_graph_matrix!(ψ::ForkTensorNetworkState)

    Lx = ψ.Lx
    Ly = ψ.Ly

    ψ.network_matrix = zeros(Int, Lx * Ly, Lx * Ly)

    for x in 1:Lx
        for y in 1:Ly

            site = (x - 1) * Ly + y
            if y < Ly
                ψ.network_matrix[site, site+1] = 1
            end
            ψ.coordinates[site, :] = [x, y]
            ψ.sites[x, y] = site
        end
    end

    for x in 1:(Lx-1)
        ψ.network_matrix[Ly*(x-1)+1, Ly*x+1] = 1
    end

    ψ.network_matrix = ψ.network_matrix + transpose(ψ.network_matrix)

end


"""
    canonize_arm!(ψ::ForkTensorNetworkState, x::Int, yi::Int, yf::Int, dir::String="left")
    Make Canonical Form of Arm-x from yi to yf
"""
function canonize_arm!(ψ::ForkTensorNetworkState, x::Integer, yi::Integer, yf::Integer, direction::String="left")

    if direction == "right"
        yf == ψ.Ly && throw(ArgumentError("yf must be smaller than ψ.Ly when direction is right"))
        (a, dy) = (0, 1)
    elseif direction == "left"
        yf == 1 && throw(ArgumentError("yf must be larger than 1 when direction is left"))
        (a, dy) = (-1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    for y = yi:dy:yf
        if ψ.χʸ !== nothing
            U, S, V = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-10, maxdim=ψ.χʸ, righttags=tags(ψ.aux_y_idx[x, y+a]))
        else
            U, S, V = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-10, righttags=tags(ψ.aux_y_idx[x, y+a]))
        end
        ψ.Ts[x, y] = V
        ψ.Ts[x, y+dy] = (S * U) * ψ.Ts[x, y+dy]
        ψ.aux_y_idx[x, y+a] = commonind(S, V)

        if direction == "right"
            ψ.aux_y_idx[x, y+a] = dag(ψ.aux_y_idx[x, y+a])
        end # to make the direction of bond consistent

        ψ.network_matrix[ψ.sites[x, y], ψ.sites[x, y+dy]] = 1
        ψ.network_matrix[ψ.sites[x, y+dy], ψ.sites[x, y]] = 0

    end

end

"""
    canonize_backbone!(ψ::ForkTensorNetworkState, xi::Int, xf::Int, dir::String="down")
    Make Canonical Form of Backbone from xi to xf
"""
function canonize_backbone!(ψ::ForkTensorNetworkState, xi::Integer, xf::Integer, direction::String="down")

    if direction == "down"
        xf == ψ.Lx && throw(ArgumentError("xf must be smaller than ψ.Lx when direction is down"))
        (a, dx) = (0, 1)
    elseif direction == "up"
        xf == 1 && throw(ArgumentError("xf must be larger than 1 when direction is up"))
        (a, dx) = (-1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    for x = xi:dx:xf
        if ψ.χˣ !== nothing
            U, S, V = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-10, maxdim=ψ.χˣ, righttags=tags(ψ.aux_x_idx[x+a]))
        else
            U, S, V = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-10, righttags=tags(ψ.aux_x_idx[x+a]))
        end
        ψ.Ts[x, 1] = V
        ψ.Ts[x+dx, 1] = (S * U) * ψ.Ts[x+dx, 1]
        ψ.aux_x_idx[x+a] = commonind(S, V)

        if direction == "down"
            ψ.aux_x_idx[x+a] = dag(ψ.aux_x_idx[x+a])
        end # to make the direction of bond consistent

        ψ.network_matrix[ψ.sites[x, 1], ψ.sites[x+dx, 1]] = 1
        ψ.network_matrix[ψ.sites[x+dx, 1], ψ.sites[x, 1]] = 0

    end

end


"""
    canonical_form!(ψ::ForkTensorNetworkState, xc::Int, yc::Int)
    Make Canonical Form of the Fork Tensor Network State with the center at (xc, yc)
"""
function canonical_form!(ψ::ForkTensorNetworkState, xc::Integer, yc::Integer)

    for x = 1:ψ.Lx
        if x != xc
            canonize_arm!(ψ, x, ψ.Ly, 2, "left")
        end
    end

    canonize_backbone!(ψ, 1, xc - 1, "down")
    canonize_backbone!(ψ, ψ.Lx, xc + 1, "up")

    if yc > 1
        canonize_arm!(ψ, xc, 1, yc - 1, "right")
    end

    canonize_arm!(ψ, xc, ψ.Ly, yc + 1, "left")
    ψ.canonical_center = (xc, yc)

end


"""
    canonical_center_move!(ψ::ForkTensorNetworkState, dir::String)
    Move the canonical center in the Fork Tensor Network State
"""
function canonical_center_move!(ψ::ForkTensorNetworkState, dir::String)

    xc = ψ.canonical_center[1]
    yc = ψ.canonical_center[2]

    if dir == "right"
        yc == ψ.Ly && throw(ArgumentError("Canonization cannot move right when yc is at the rightmost site"))
        canonize_arm!(ψ, xc, yc, yc, "right")
        ψ.canonical_center = (xc, yc + 1)
    elseif dir == "left"
        yc == 1 && throw(ArgumentError("Canonization cannot move left when yc is at the leftmost site"))
        canonize_arm!(ψ, xc, yc, yc, "left")
        ψ.canonical_center = (xc, yc - 1)
    elseif dir == "up"
        xc == 1 && throw(ArgumentError("Canonization cannot move up when xc is at the topmost site"))
        yc != 1 && throw(ArgumentError("yc should be at 1 when canonizing backbone"))
        canonize_backbone!(ψ, xc, xc, "up")
        ψ.canonical_center = (xc - 1, yc)
    elseif dir == "down"
        xc == ψ.Lx && throw(ArgumentError("Canonization cannot move down when xc is at the bottommost site"))
        yc != 1 && throw(ArgumentError("yc should be at 1 when canonizing backbone"))
        canonize_backbone!(ψ, xc, xc, "down")
        ψ.canonical_center = (xc + 1, yc)
    else
        throw(ArgumentError("Invalid direction"))
    end

end


"""
    network_update!(ψ::ForkTensorNetworkState, direction::String)
    Update the network matrix of the Fork Tensor Network State
"""
function network_update!(ψ::ForkTensorNetworkState, direction::String)

    if direction == "right"
        (dx, dy) = (0, 1)
    elseif direction == "left"
        (dx, dy) = (0, -1)
    elseif direction == "up"
        (dx, dy) = (-1, 0)
    elseif direction == "down"
        (dx, dy) = (1, 0)
    else
        throw(ArgumentError("Invalid direction"))
    end

    (x, y) = ψ.canonical_center
    ψ.network_matrix[ψ.sites[x, y], ψ.sites[x+dx, y+dy]] = 1
    ψ.network_matrix[ψ.sites[x+dx, y+dy], ψ.sites[x, y]] = 0
    ψ.canonical_center = (x + dx, y + dy)

end


"""
    normalize_ftn!(ψ::ForkTensorNetworkState)
    Normalize the Fork Tensor Network State
"""
function normalize_ftn!(ψ::ForkTensorNetworkState)

    if ψ.canonical_center !== nothing
        xc = ψ.canonical_center[1]
        yc = ψ.canonical_center[2]
        ψ.Ts[xc, yc] *= 1 / norm(ψ.Ts[xc, yc])
    else
        canonical_form!(ψ, 1, 1)
        ψ.Ts[1, 1] *= 1 / norm(ψ.Ts[1, 1])
    end

end


"""
    norm_ftn(ψ::ForkTensorNetworkState)
    Returns the norm of the Fork Tensor Network State
"""
function norm_ftn(ψ::ForkTensorNetworkState)

    if ψ.canonical_center === nothing
        Tx = 1
        for x = 1:ψ.Lx
            Ty = 1
            for y = ψ.Ly:-1:1
                Ty = (Ty * noprime(prime(ψ.Ts[x, y]); tags="Site")) * dag(ψ.Ts[x, y])
            end
            Tx *= Ty
        end
        C = scalar(Tx)

    else
        xc = ψ.canonical_center[1]
        yc = ψ.canonical_center[2]
        C = norm(ψ.Ts[xc, yc])
    end

    return C

end


"""
    overlap_ftn(ψ1::ForkTensorNetworkState, ψ2::ForkTensorNetworkState)
    Returns the overlap between two Fork Tensor Network States
"""
function overlap_ftn(ψ1::ForkTensorNetworkState, ψ2::ForkTensorNetworkState)

    Tx = 1
    for x = 1:ψ1.Lx
        Ty = 1
        for y = ψ1.Ly:-1:1
            Ty = (Ty * dag(replaceind(ψ1.Ts[x, y], ψ1.phys_idx[x, y], ψ2.phys_idx[x, y]))) * noprime(prime(ψ2.Ts[x, y]); tags="Site")
        end
        Tx *= Ty
    end

    return scalar(Tx)

end


"""
    expectation_value_ftn(ψ::ForkTensorNetworkState, Ĥ::ForkTensorNetworkOperator)
    Returns the expectation value of the Fork Tensor Network Operator
"""
function expectation_value_ftn(ψ::ForkTensorNetworkState, Ĥ::ForkTensorNetworkOperator)

    Tx = 1
    for x = 1:ψ.Lx
        Ty = 1
        for y = ψ.Ly:-1:1
            Ty = ((Ty * ψ.Ts[x, y]) * Ĥ.Ws[x, y]) * prime(dag(ψ.Ts[x, y]))
        end
        Tx *= Ty
    end

    return scalar(Tx) / norm_ftn(ψ)

end



function expectation_value_ftn(ψ::ForkTensorNetworkState, ops::Vector{Tuple{T,T,String}}) where {T<:Integer}

    normalize_ftn!(ψ)

    if length(ops) == 1

        x, y, op_name = ops[1]
        if (x, y) != ψ.canonical_center
            canonical_form!(ψ, x, y)
        end

        return scalar(ψ.Ts[x, y] * op(op_name, ψ.phys_idx[x, y]) * dag(prime(ψ.Ts[x, y]; tags="Site")))

    else

        ψ′ = deepcopy(ψ)
        applying_local_operators!(ψ′, ops)

        return overlap_ftn(ψ, ψ′)

    end

end



function applying_local_operators!(ψ::ForkTensorNetworkState, ops::Vector{Tuple{T,T,String}}) where {T<:Integer}

    for i = 1:length(ops)

        x, y, op_name = ops[i]
        ψ.Ts[x, y] .= noprime(op(ψ.phys_idx[x, y], op_name) * ψ.Ts[x, y])

    end

end


