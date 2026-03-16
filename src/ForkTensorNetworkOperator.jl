"""
    ForkTensorNetworkOperator

Immutable struct representing a Fork Tensor Network operator (e.g., Hamiltonian in MPO form).
The operator shares the same fork geometry as `ForkTensorNetworkState`.

# Fields
- `Lx::Int`: Number of sites in the x-direction (backbone length).
- `Ly::Int`: Number of sites in the y-direction (arm length).
- `phys_idx`: Physical indices for each site.
- `aux_x_idx`: Auxiliary (bond) indices along the backbone.
- `aux_y_idx`: Auxiliary (bond) indices along the arms.
- `Ws`: MPO tensor storage for each site.
"""
struct ForkTensorNetworkOperator

    Lx::Int
    Ly::Int
    phys_idx::Matrix{Index}
    aux_x_idx::Vector{Index}
    aux_y_idx::Matrix{Index}
    Ws::Matrix{ITensor}


    """
        ForkTensorNetworkOperator(Ws, phys_idx, aux_x_idx, aux_y_idx)

    Construct an FTN operator from pre-built MPO tensors and index arrays.
    """
    function ForkTensorNetworkOperator(Ws::AbstractMatrix{ITensor}, phys_idx::AbstractMatrix{<:Index}, aux_x_idx::AbstractVector{<:Index}, aux_y_idx::AbstractMatrix{<:Index})
        Lx = size(Ws, 1)
        Ly = size(Ws, 2)
        return new(Lx, Ly, phys_idx, aux_x_idx, aux_y_idx, Ws)
    end

end


"""
    flux_check(H::ForkTensorNetworkOperator)

Verify quantum number flux conservation for every MPO tensor in `H`.
Prints the flux of each tensor. Throws an error if QNs are not present.
"""
function flux_check(H::ForkTensorNetworkOperator)
    hasqns(H.phys_idx[1, 1]) != true && throw(ArgumentError("The QN is not conserved"))

    println("Checking Flux...")
    println("Lx = $(H.Lx), Ly = $(H.Ly)")

    for x = 1:H.Lx
        for y = 1:H.Ly
            if checkflux(H.Ws[x, y]) === nothing
                println("x = $(x), y = $(y): ", flux(H.Ws[x, y]))
            end
        end
    end
end