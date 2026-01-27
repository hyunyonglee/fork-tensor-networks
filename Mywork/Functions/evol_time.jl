#!/home/heungsikim//.local/bin/julia
# using LinearAlgebra
# using Strided
# using ITensors
# using JLD2
# using QuadGK
# println("Started")
# include("../models/AndersonImpurityModel.jl")
# using .AndersonImpurityModel

# include("../src/ForkTensorNetworks.jl")
# using .ForkTensorNetworks
# println("all packages are loaded")
# flush(stdout)


function get_TDVP(tdvp_params,N_bath,N_orb,orbital_idx)


    DMRG_results = open("N_bath$(N_bath)_N_orb$(N_orb)/DMRG_RESULT.jls", "r") do f
        deserialize(f)
    end

    
    Ground_state = DMRG_results["Ground state"]
    Ground_energy = DMRG_results["Ground energy"]
    H = DMRG_results["Hamiltonian"]

    phi_t = deepcopy(Ground_state)
    phi_0 = deepcopy(Ground_state)

    applying_local_operators!(phi_0, [(orbital_idx[1,1], orbital_idx[1,2], "cdag")])
    applying_local_operators!(phi_t, [(orbital_idx[2,1], orbital_idx[2,2], "cdag")])
    

    tdvp = TDVP(tdvp_params);

    GlstFTN = []
    GlstFTN_raw = []
    auto_corr = overlap_ftn(phi_0, phi_t)
    push!(GlstFTN, auto_corr)
    push!(GlstFTN_raw, auto_corr)
    Ts = [0.0+ 0.0im]
    xmaxbond = []
    ymaxbond = []

    tdvp_time = 0
    overlap_time = 0
    for i = 1:Int(tdvp_params["t_max"]/tdvp_params["δt"])
        
        
        tdvp_start_time = time()

        println("Time evolution step ",i*tdvp_params["δt"])
        flush(stdout)

        @time run_tdvp!(tdvp, H, phi_t, 1)

        println("Done, time = ", time() - tdvp_start_time)

        tdvp_time += time() - tdvp_start_time
        flush(stdout)




        overlap_start_time = time()

        println("Overlap calculation")
        flush(stdout)

        @time auto_corr = overlap_ftn(phi_0, phi_t)

        println("Done, time = ", time() - overlap_start_time)
        flush(stdout)

        overlap_time += time() - overlap_start_time

        append!(GlstFTN, auto_corr*exp(1im * Ground_energy *i*tdvp_params["δt"]))

        append!(GlstFTN_raw,auto_corr)

        println(GlstFTN)
        flush(stdout)

        append!(Ts, tdvp.time)

    end

    save_object("TDVP_RESULT_$(orbital_idx[1,1])_$(orbital_idx[2,1]).jld2",Dict("GlstFTN" => GlstFTN, "GlstFTN_raw" => GlstFTN_raw, "t_step" => tdvp_params["δt"], "t_max" => tdvp_params["t_max"], "Ts" => Ts))
    open("N_bath$(N_bath)_N_orb$(N_orb)/FINAL_STATE_$(orbital_idx[1,1])_$(orbital_idx[2,1]).jls", "w") do f
        serialize(f, phi_t)
    end
    println("Total TDVP time = ", tdvp_time + overlap_time)
    println("Total TDVP time = ", tdvp_time)
    println("Total Overlap time = ", overlap_time)
end

