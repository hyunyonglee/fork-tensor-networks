"""
    Module ForkTensorNetworks

Fork Tensor Network algorithms for quantum many-body systems, including
DMRG (ground state search) and TDVP (time evolution) on fork-shaped tensor networks.

Reference: Phys. Rev. X 7, 031013 (2017)

Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
module ForkTensorNetworks

using ITensors, ITensorMPS, GraphRecipes, Plots, Printf
using LinearAlgebra: SymTridiagonal

include("Functions.jl")
include("ForkTensorNetworkOperator.jl")
include("ForkTensorNetworkState.jl")
include("FTNEnvironments.jl")
include("SubspaceExpansion.jl")
include("DMRG.jl")
include("TDVP.jl")

# Structs & Parameters
export ForkTensorNetworkOperator, ForkTensorNetworkState
export DMRGParams, DMRG
export TDVPParams, TDVP

# DMRG & TDVP
export run_dmrg!, run_tdvp!

# ForkTensorNetworkState methods
export overlap_ftn, expectation_value_ftn, applying_local_operators!
export norm_ftn, normalize_ftn!
export plot_network

end # module ForkTensorNetworks
