#!/home/heungsikim//.local/bin/julia
using LinearAlgebra
using Strided
using ITensors
using JLD2
using QuadGK
include("../models/AndersonImpurityModel.jl")
using .AndersonImpurityModel
include("../src/ForkTensorNetworks.jl")
using .ForkTensorNetworks
include("Functions/evol_time.jl")
include("Functions/get_ground.jl")
BLAS.set_num_threads(2)
Strided.set_num_threads(2)
println("all packages are loaded")
flush(stdout)

let
    N_bath = 4
    N_orb = 2
    D = 1

    Dmrg_params = Dict(
        "method" => "two-site",       # option: "single-site", "two-site"
        "χˣ" => 50,                    # Maximum bond dimension for the backbone
        "χʸ" => 400,                    # Maximum bond dimension for the arm
        "max_iter" => 30,             # Maximum number of DMRG sweeps
        "svd_tol" => 1e-8,            # Threshold for truncating singular values
        "convergence_tol" => 1e-10,    # Threshold for energy convergence
        "verbose" => true,             # Print information during DMRG sweeps
        "subspace_expansion" => true,  # Use subspace expansion or not (Highly recommended for the single-site DMRG)
        "α" => 0.6,                    # subspace expansion parameter
        "α_decay" => 0.1               # Must be less than or equal to 1.0
    )


    @time get_state(D,N_bath,N_orb,Dmrg_params)


    tdvp_params = Dict(
        "method" => "single-site",   # option: "single-site", "two-site"
        "χˣ" => 50,                  # Maximum bond dimension for the backbone
        "χʸ" => 300,                  # Maximum bond dimension for the arm
        "verb_level" => 0,           # Print information during TDVP sweeps (0: no print, 1: print params, 2: print details)
        "δt" => 0.1,                # Time step
        "Ncut" => 30,                  # Maximum number of Kyrlov vectors used in the Kylov exponentiation
        "svd_tol" => 1E-8,
        "t_max" => 20.0                
    )

    orbital_idx = [1 1; #Cdag idx1
                    1 1]#Cdag idx2

    @time get_TDVP(tdvp_params,N_bath,N_orb,orbital_idx)
    flush(stdout)
end

