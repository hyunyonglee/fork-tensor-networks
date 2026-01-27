#!/home/heungsikim//.local/bin/julia
# using LinearAlgebra
# using Strided
# using ITensors
# using JLD2
# using QuadGK
# println("Started")
# include("../../models/AndersonImpurityModel.jl")
# using .AndersonImpurityModel
# include("../../src/ForkTensorNetworks.jl")
# using .ForkTensorNetworks
include("../Functions/Half_Elliptical_Bath_Params.jl")
# include("../Functions/save_dmrg.jl")
# BLAS.set_num_threads(2)
# Strided.set_num_threads(2)
# println("all packages are loaded")
flush(stdout)
using Serialization
import HDF5

function get_state(D,N_bath,N_orb,dmrg_params)
    # N_orb = N_orbital               # Number of orbitals
    # N_bath = N_bath              # Number of bath sites

    V_arr,omega_arr = get_Half_elliptical_Bath_Fit(D,N_bath)


    ρ = 0.5                 # Filling of the Impurity model (NOT impurity site)
    U = 2.0                 # Coulomb repulsion
    J = 0.3                 # Spin-flip, pair-hopping term
    U′  = U - 2*J           # Coulomb repulsion for different orbital
    εₖ = zeros( 2 * N_orb, N_bath + 1 )
    Vₖ = zeros(Complex, 2 * N_orb, N_bath + 1 )

    for l=1:(2*N_orb)
        for k=1:N_bath
            Vₖ[l,2:end] = V_arr
            εₖ[l,2:end] = omega_arr
        end
    end

    εₖ[1:end,1] .= - U/2 #Half filling
    

    model_params = Dict(
        "model" => "AndersonImpurity",
        "N_orb" => N_orb,
        "N_bath" => N_bath,
        "εₖ" => εₖ,
        "Vₖ" => Vₖ,
        "U" => U,
        "U′" => U′,
        "J" => J,
        "ρ" => ρ,
        "conserve_qns" => true # false: No symmetry, true: U(1)xU(1) symmetry
    )


    


    Ws, phys_idx, ftno_aux_x_idx, ftno_aux_y_idx = ftno_aim_model(model_params);
    Ts, ftns_aux_x_idx, ftns_aux_y_idx = ftns_initial_state(phys_idx, model_params["ρ"]; conserve_qns=model_params["conserve_qns"]);
   
    Ĥ = ForkTensorNetworkOperator(Ws, phys_idx, ftno_aux_x_idx, ftno_aux_y_idx);
    ψ = ForkTensorNetworkState(Ts, phys_idx, ftns_aux_x_idx, ftns_aux_y_idx);
    
    println("FTNO & FTNS are prepared with \n * D => $(D)\n * N_bath => $(N_bath) \n * N_orb => $(N_orb)")
    flush(stdout)
    


    dmrg = DMRG(dmrg_params);
    @time E, ψ = run_dmrg!(dmrg, Ĥ, ψ);

    if Int("N_bath$(N_bath)_N_orb$(N_orb)" in readdir()) == 0
        mkdir("N_bath$(N_bath)_N_orb$(N_orb)")
    end
        

    DMRG_Result = Dict("Ground energy" => E, "Hamiltonian" => Ĥ, "Ground state" => ψ)


    open("N_bath$(N_bath)_N_orb$(N_orb)/DMRG_RESULT.jls", "w") do f
        serialize(f, DMRG_Result)
    end
    println("Energy:", E)
    println("Save done")
    flush(stdout)
end
