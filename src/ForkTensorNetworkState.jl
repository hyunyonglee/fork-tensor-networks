"""
    ForkTensorNetworkState

Mutable struct representing a Fork Tensor Network (FTN) state.
The network has a backbone along the x-direction and arms branching in the y-direction.

# Fields
- `Lx::Int`: Number of sites in the x-direction (backbone length).
- `Ly::Int`: Number of sites in the y-direction (arm length).
- `χˣ`: Maximum bond dimension along the backbone (or `nothing` for unconstrained).
- `χʸ`: Maximum bond dimension along the arms (or `nothing` for unconstrained).
- `canonical_center`: Current canonical center as `(x, y)` tuple, or `nothing`.
- `network_matrix`: Adjacency matrix encoding the directed canonical structure.
- `coordinates`: Mapping from linear site index to `(x, y)` coordinates.
- `sites`: Mapping from `(x, y)` to linear site index.
- `phys_idx`: Physical indices for each site.
- `aux_x_idx`: Auxiliary (bond) indices along the backbone.
- `aux_y_idx`: Auxiliary (bond) indices along the arms.
- `Ts`: Tensor storage for each site.
"""
mutable struct ForkTensorNetworkState

    Lx::Int
    Ly::Int
    χˣ::Union{Integer,Nothing}
    χʸ::Union{Integer,Nothing}
    canonical_center::Union{Tuple{Integer,Integer},Nothing}
    network_matrix::Matrix{Integer}
    coordinates::Matrix{Integer}
    sites::Matrix{Integer}
    phys_idx::Matrix{Index}
    aux_x_idx::Vector{Index}
    aux_y_idx::Matrix{Index}
    Ts::Matrix{ITensor}


    """
        ForkTensorNetworkState(Lx, Ly, phys_idx::Vector{Index}; χˣ=nothing, χʸ=nothing)

    Construct an FTN state from a flat vector of physical indices (mapped row-major to the 2D grid).
    Auxiliary indices and random tensors are initialized automatically.
    """
    function ForkTensorNetworkState(Lx::T, Ly::T, phys_idx::Vector{Index{T}}; χˣ=nothing, χʸ=nothing) where {T<:Integer}
        ψ = new(
            Lx, Ly, χˣ, χʸ, nothing,
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),
            Matrix{Integer}(undef, Lx * Ly, 2),
            Matrix{Integer}(undef, Lx, Ly),
            Matrix{Index}(undef, Lx, Ly),
            Vector{Index}(undef, Lx - 1),
            Matrix{Index}(undef, Lx, Ly - 1),
            Matrix{ITensor}(undef, Lx, Ly)
        )
        create_fork_graph_matrix!(ψ)
        initialize_indices!(ψ, phys_idx)
        initialize_tensors_random!(ψ)
        return ψ
    end


    """
        ForkTensorNetworkState(Lx, Ly, phys_idx::Matrix{Index}; χˣ=nothing, χʸ=nothing)

    Construct an FTN state from a 2D matrix of physical indices.
    Auxiliary indices and random tensors are initialized automatically.
    """
    function ForkTensorNetworkState(Lx::T, Ly::T, phys_idx::Matrix{Index{T}}; χˣ=nothing, χʸ=nothing) where {T<:Integer}
        ψ = new(
            Lx, Ly, χˣ, χʸ, nothing,
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),
            Matrix{Integer}(undef, Lx * Ly, 2),
            Matrix{Integer}(undef, Lx, Ly),
            phys_idx,
            Vector{Index}(undef, Lx - 1),
            Matrix{Index}(undef, Lx, Ly - 1),
            Matrix{ITensor}(undef, Lx, Ly)
        )
        create_fork_graph_matrix!(ψ)
        initialize_indices!(ψ)
        initialize_tensors_random!(ψ)
        return ψ
    end


    """
        ForkTensorNetworkState(Ts, phys_idx, aux_x_idx, aux_y_idx; χˣ=nothing, χʸ=nothing)

    Construct an FTN state from pre-existing tensors and indices.
    Only the graph matrix is initialized; tensors and indices are used as provided.
    """
    function ForkTensorNetworkState(Ts::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index}; χˣ=nothing, χʸ=nothing)
        Lx = size(Ts, 1)
        Ly = size(Ts, 2)
        ψ = new(
            Lx, Ly, χˣ, χʸ, nothing,
            Matrix{Integer}(undef, Lx * Ly, Lx * Ly),
            Matrix{Integer}(undef, Lx * Ly, 2),
            Matrix{Integer}(undef, Lx, Ly),
            phys_idx, aux_x_idx, aux_y_idx, Ts
        )
        create_fork_graph_matrix!(ψ)
        return ψ
    end

end


"""
    initialize_tensors_random!(ψ::ForkTensorNetworkState)

Fill all tensors in `ψ` with random elements, using the appropriate index structure
for each site (backbone junctions, arm endpoints, interior arm sites).
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
    initialize_indices!(ψ::ForkTensorNetworkState, phys_idx=nothing)

Initialize auxiliary bond indices for the FTN state. If `phys_idx` is a `Vector{Index}`,
the physical indices are also populated from the flat vector (row-major order).
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

Visualize the FTN graph using `GraphRecipes`. The canonical center is highlighted.
"""
function plot_network(ψ::ForkTensorNetworkState)
    g = ψ.network_matrix
    xs = ψ.coordinates[:, 2]
    ys = -ψ.coordinates[:, 1]
    color = [1 for i in 1:length(xs)]

    if ψ.canonical_center !== nothing
        color[ψ.sites[ψ.canonical_center[1], ψ.canonical_center[2]]] = 2
    end

    name = ["($(ψ.coordinates[i,1]), $(ψ.coordinates[i,2]))" for i in 1:length(xs)]
    graphplot(g, x=xs, y=ys, markersize=0.5, markercolor=color, names=name)
end


"""
    create_fork_graph_matrix!(ψ::ForkTensorNetworkState)

Build the adjacency matrix for the fork tensor network graph.
Arms are connected along y, and backbone sites are connected along x at y=1.
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
    canonize_arm!(ψ, x, yi, yf, direction=:left)

Bring arm `x` into canonical form by performing SVD sweeps from site `yi` to `yf`.
Direction `:left` sweeps toward the backbone; `:right` sweeps away from it.
"""
function canonize_arm!(ψ::ForkTensorNetworkState, x::Integer, yi::Integer, yf::Integer, direction::Symbol=:left)
    if direction == :right
        yf == ψ.Ly && throw(ArgumentError("yf must be smaller than ψ.Ly when direction is right"))
        (a, dy) = (0, 1)
    elseif direction == :left
        yf == 1 && throw(ArgumentError("yf must be larger than 1 when direction is left"))
        (a, dy) = (-1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    for y = yi:dy:yf
        if ψ.χʸ !== nothing
            U, S, V = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-12, maxdim=ψ.χʸ, righttags=tags(ψ.aux_y_idx[x, y+a]))
        else
            U, S, V = svd(ψ.Ts[x, y], ψ.aux_y_idx[x, y+a]; cutoff=1e-12, righttags=tags(ψ.aux_y_idx[x, y+a]))
        end
        ψ.Ts[x, y] = V
        ψ.Ts[x, y+dy] = (S * U) * ψ.Ts[x, y+dy]
        ψ.aux_y_idx[x, y+a] = commonind(S, V)

        if direction == :right
            ψ.aux_y_idx[x, y+a] = dag(ψ.aux_y_idx[x, y+a])
        end

        ψ.network_matrix[ψ.sites[x, y], ψ.sites[x, y+dy]] = 1
        ψ.network_matrix[ψ.sites[x, y+dy], ψ.sites[x, y]] = 0
    end
end


"""
    canonize_backbone!(ψ, xi, xf, direction=:down)

Bring the backbone into canonical form by performing SVD sweeps from site `xi` to `xf`.
Direction `:down` sweeps toward increasing x; `:up` sweeps toward decreasing x.
"""
function canonize_backbone!(ψ::ForkTensorNetworkState, xi::Integer, xf::Integer, direction::Symbol=:down)
    if direction == :down
        xf == ψ.Lx && throw(ArgumentError("xf must be smaller than ψ.Lx when direction is down"))
        (a, dx) = (0, 1)
    elseif direction == :up
        xf == 1 && throw(ArgumentError("xf must be larger than 1 when direction is up"))
        (a, dx) = (-1, -1)
    else
        throw(ArgumentError("Invalid direction"))
    end

    for x = xi:dx:xf
        if ψ.χˣ !== nothing
            U, S, V = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-12, maxdim=ψ.χˣ, righttags=tags(ψ.aux_x_idx[x+a]))
        else
            U, S, V = svd(ψ.Ts[x, 1], ψ.aux_x_idx[x+a]; cutoff=1e-12, righttags=tags(ψ.aux_x_idx[x+a]))
        end
        ψ.Ts[x, 1] = V
        ψ.Ts[x+dx, 1] = (S * U) * ψ.Ts[x+dx, 1]
        ψ.aux_x_idx[x+a] = commonind(S, V)

        if direction == :down
            ψ.aux_x_idx[x+a] = dag(ψ.aux_x_idx[x+a])
        end

        ψ.network_matrix[ψ.sites[x, 1], ψ.sites[x+dx, 1]] = 1
        ψ.network_matrix[ψ.sites[x+dx, 1], ψ.sites[x, 1]] = 0
    end
end


"""
    canonical_form!(ψ, xc, yc)

Bring the full FTN state into canonical form with the canonical center at site `(xc, yc)`.
All arms and backbone segments are canonized to point toward the center.
"""
function canonical_form!(ψ::ForkTensorNetworkState, xc::Integer, yc::Integer)
    for x = 1:ψ.Lx
        if x != xc
            canonize_arm!(ψ, x, ψ.Ly, 2, :left)
        end
    end

    canonize_backbone!(ψ, 1, xc - 1, :down)
    canonize_backbone!(ψ, ψ.Lx, xc + 1, :up)

    if yc > 1
        canonize_arm!(ψ, xc, 1, yc - 1, :right)
    end

    canonize_arm!(ψ, xc, ψ.Ly, yc + 1, :left)
    ψ.canonical_center = (xc, yc)
end


"""
    canonical_center_move!(ψ, dir)

Move the canonical center one step in direction `dir` (`:right`, `:left`, `:up`, `:down`)
by performing a single SVD and absorbing the bond matrix into the neighbor.
"""
function canonical_center_move!(ψ::ForkTensorNetworkState, dir::Symbol)
    xc = ψ.canonical_center[1]
    yc = ψ.canonical_center[2]

    if dir == :right
        yc == ψ.Ly && throw(ArgumentError("Canonization cannot move right when yc is at the rightmost site"))
        canonize_arm!(ψ, xc, yc, yc, :right)
        ψ.canonical_center = (xc, yc + 1)
    elseif dir == :left
        yc == 1 && throw(ArgumentError("Canonization cannot move left when yc is at the leftmost site"))
        canonize_arm!(ψ, xc, yc, yc, :left)
        ψ.canonical_center = (xc, yc - 1)
    elseif dir == :up
        xc == 1 && throw(ArgumentError("Canonization cannot move up when xc is at the topmost site"))
        yc != 1 && throw(ArgumentError("yc should be at 1 when canonizing backbone"))
        canonize_backbone!(ψ, xc, xc, :up)
        ψ.canonical_center = (xc - 1, yc)
    elseif dir == :down
        xc == ψ.Lx && throw(ArgumentError("Canonization cannot move down when xc is at the bottommost site"))
        yc != 1 && throw(ArgumentError("yc should be at 1 when canonizing backbone"))
        canonize_backbone!(ψ, xc, xc, :down)
        ψ.canonical_center = (xc + 1, yc)
    else
        throw(ArgumentError("Invalid direction"))
    end
end


"""
    network_update!(ψ, direction)

Update the adjacency matrix to reflect a canonical center move in `direction`
(`:right`, `:left`, `:up`, `:down`) without performing any SVD.
"""
function network_update!(ψ::ForkTensorNetworkState, direction::Symbol)
    if direction == :right
        (dx, dy) = (0, 1)
    elseif direction == :left
        (dx, dy) = (0, -1)
    elseif direction == :up
        (dx, dy) = (-1, 0)
    elseif direction == :down
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

Normalize the FTN state in-place. If a canonical center exists, normalizes the center tensor;
otherwise, first brings the state into canonical form at `(1, 1)`.
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
    norm_ftn(ψ::ForkTensorNetworkState) -> Number

Compute the norm of the FTN state. If the state is in canonical form, returns the norm
of the center tensor; otherwise, contracts the full network.
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
        return scalar(Tx)
    else
        xc = ψ.canonical_center[1]
        yc = ψ.canonical_center[2]
        return norm(ψ.Ts[xc, yc])
    end
end


"""
    overlap_ftn(ψ1, ψ2) -> Number

Compute the overlap ⟨ψ1|ψ2⟩ by contracting the full fork tensor networks.
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
    expectation_value_ftn(ψ, H::ForkTensorNetworkOperator) -> Number

Compute ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ by contracting the state with the MPO-like FTN operator.
"""
function expectation_value_ftn(ψ::ForkTensorNetworkState, H::ForkTensorNetworkOperator)
    Tx = 1
    for x = 1:ψ.Lx
        Ty = 1
        for y = ψ.Ly:-1:1
            Ty = ((Ty * ψ.Ts[x, y]) * H.Ws[x, y]) * prime(dag(ψ.Ts[x, y]))
        end
        Tx *= Ty
    end
    return scalar(Tx) / norm_ftn(ψ)
end


"""
    expectation_value_ftn(ψ, ops::Vector{Tuple{Int,Int,String}}) -> Number

Compute the expectation value of a product of local operators.
Each entry `(x, y, op_name)` specifies a site and an operator name (ITensors `op`).
For a single operator, the canonical center is moved to the site for efficiency.
For multiple operators, applies them to a copy and computes the overlap.
"""
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


"""
    applying_local_operators!(ψ, ops)

Apply a sequence of local operators to the FTN state in-place.
Each entry `(x, y, op_name)` in `ops` applies `op(op_name)` at site `(x, y)`.
"""
function applying_local_operators!(ψ::ForkTensorNetworkState, ops::Vector{Tuple{T,T,String}}) where {T<:Integer}
    for i = 1:length(ops)
        x, y, op_name = ops[i]
        ψ.Ts[x, y] .= noprime(op(ψ.phys_idx[x, y], op_name) * ψ.Ts[x, y])
    end
end