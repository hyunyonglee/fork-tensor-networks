"""
    Module ForkTensorNetworks
    Reference: PHYSICAL REVIEW X 7, 031013 (2017)
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
module ForkTensorNetworks

using ITensors, ITensorMPS, GraphRecipes, Plots, Printf
using LinearAlgebra: diagm, SymTridiagonal

include("Functions.jl")
include("ForkTensorNetworkOperator.jl")
include("ForkTensorNetworkState.jl")
include("DMRG.jl")
include("TDVP.jl")

# Export structs
export ForkTensorNetworkOperator, ForkTensorNetworkState, DMRG, TDVP

# Export methods related to ForkTensorNetworkOperator
export flux_check

# Export methods related to ForkTensorNetworkState
export initialize_tensors_random!, canonical_form!, canonical_center_move!, plot_network
export overlap_ftn, expectation_value_ftn, applying_local_operators!
export norm_ftn, normalize_ftn!

# Export methods related to DMRG
export run_dmrg!

# Export methods related to TDVP
export run_tdvp!

end # module ForkTensorNetworks