using HDF5

export save_ftns, load_ftns, save_ftno, load_ftno

"""
    save_ftns(filename, ψ; group="ftns")

Save ForkTensorNetworkState to HDF5 file.
"""
function save_ftns(filename::AbstractString, ψ::ForkTensorNetworkState; group::String="ftns")
    h5open(filename, "cw") do f
        g = create_group(f, group)
        attributes(g)["type"] = "ForkTensorNetworkState"
        attributes(g)["Lx"] = ψ.Lx
        attributes(g)["Ly"] = ψ.Ly
        attributes(g)["chi_x"] = ψ.χˣ === nothing ? -1 : Int(ψ.χˣ)
        attributes(g)["chi_y"] = ψ.χʸ === nothing ? -1 : Int(ψ.χʸ)
        attributes(g)["cc_x"] = ψ.canonical_center === nothing ? -1 : Int(ψ.canonical_center[1])
        attributes(g)["cc_y"] = ψ.canonical_center === nothing ? -1 : Int(ψ.canonical_center[2])

        # Indices
        g_phys = create_group(g, "phys_idx")
        for x in 1:ψ.Lx, y in 1:ψ.Ly
            write(g_phys, "$(x)_$(y)", ψ.phys_idx[x, y])
        end

        g_aux_x = create_group(g, "aux_x_idx")
        for x in eachindex(ψ.aux_x_idx)
            write(g_aux_x, "$x", ψ.aux_x_idx[x])
        end

        g_aux_y = create_group(g, "aux_y_idx")
        for x in 1:ψ.Lx, y in 1:(ψ.Ly - 1)
            write(g_aux_y, "$(x)_$(y)", ψ.aux_y_idx[x, y])
        end

        # Tensors
        g_ts = create_group(g, "tensors")
        for x in 1:ψ.Lx, y in 1:ψ.Ly
            write(g_ts, "$(x)_$(y)", ψ.Ts[x, y])
        end
    end
end

"""
    load_ftns(filename; group="ftns") -> ForkTensorNetworkState

Load ForkTensorNetworkState from HDF5 file.
"""
function load_ftns(filename::AbstractString; group::String="ftns")
    h5open(filename, "r") do f
        g = f[group]
        Lx = read(attributes(g), "Lx")
        Ly = read(attributes(g), "Ly")
        chi_x = read(attributes(g), "chi_x")
        chi_y = read(attributes(g), "chi_y")
        cc_x = read(attributes(g), "cc_x")
        cc_y = read(attributes(g), "cc_y")

        χˣ = chi_x < 0 ? nothing : chi_x
        χʸ = chi_y < 0 ? nothing : chi_y
        canonical_center = cc_x < 0 ? nothing : (cc_x, cc_y)

        # Indices
        phys_idx = Matrix{Index}(undef, Lx, Ly)
        g_phys = g["phys_idx"]
        for x in 1:Lx, y in 1:Ly
            phys_idx[x, y] = read(g_phys, "$(x)_$(y)", Index)
        end

        aux_x_idx = Vector{Index}(undef, Lx - 1)
        g_aux_x = g["aux_x_idx"]
        for x in 1:(Lx - 1)
            aux_x_idx[x] = read(g_aux_x, "$x", Index)
        end

        aux_y_idx = Matrix{Index}(undef, Lx, Ly - 1)
        g_aux_y = g["aux_y_idx"]
        for x in 1:Lx, y in 1:(Ly - 1)
            aux_y_idx[x, y] = read(g_aux_y, "$(x)_$(y)", Index)
        end

        # Tensors
        Ts = Matrix{ITensor}(undef, Lx, Ly)
        g_ts = g["tensors"]
        for x in 1:Lx, y in 1:Ly
            Ts[x, y] = read(g_ts, "$(x)_$(y)", ITensor)
        end

        # Use existing constructor (auto-generates network_matrix, coordinates, sites)
        ψ = ForkTensorNetworkState(Ts, phys_idx, aux_x_idx, aux_y_idx; χˣ=χˣ, χʸ=χʸ)
        ψ.canonical_center = canonical_center
        return ψ
    end
end

"""
    save_ftno(filename, H; group="ftno")

Save ForkTensorNetworkOperator to HDF5 file.
"""
function save_ftno(filename::AbstractString, H::ForkTensorNetworkOperator; group::String="ftno")
    h5open(filename, "cw") do f
        g = create_group(f, group)
        attributes(g)["type"] = "ForkTensorNetworkOperator"
        attributes(g)["Lx"] = H.Lx
        attributes(g)["Ly"] = H.Ly

        g_phys = create_group(g, "phys_idx")
        for x in 1:H.Lx, y in 1:H.Ly
            if isassigned(H.phys_idx, x, y)
                write(g_phys, "$(x)_$(y)", H.phys_idx[x, y])
            end
        end

        g_aux_x = create_group(g, "aux_x_idx")
        for x in eachindex(H.aux_x_idx)
            write(g_aux_x, "$x", H.aux_x_idx[x])
        end

        g_aux_y = create_group(g, "aux_y_idx")
        for x in 1:H.Lx, y in 1:H.Ly
            if isassigned(H.aux_y_idx, x, y)
                write(g_aux_y, "$(x)_$(y)", H.aux_y_idx[x, y])
            end
        end

        g_ws = create_group(g, "tensors")
        for x in 1:H.Lx, y in 1:H.Ly
            if isassigned(H.Ws, x, y)
                write(g_ws, "$(x)_$(y)", H.Ws[x, y])
            end
        end
    end
end

"""
    load_ftno(filename; group="ftno") -> ForkTensorNetworkOperator

Load ForkTensorNetworkOperator from HDF5 file.
"""
function load_ftno(filename::AbstractString; group::String="ftno")
    h5open(filename, "r") do f
        g = f[group]
        Lx = read(attributes(g), "Lx")
        Ly = read(attributes(g), "Ly")

        phys_idx = Matrix{Index}(undef, Lx, Ly)
        g_phys = g["phys_idx"]
        for x in 1:Lx, y in 1:Ly
            key = "$(x)_$(y)"
            if haskey(g_phys, key)
                phys_idx[x, y] = read(g_phys, key, Index)
            end
        end

        aux_x_idx = Vector{Index}(undef, Lx - 1)
        g_aux_x = g["aux_x_idx"]
        for x in 1:(Lx - 1)
            aux_x_idx[x] = read(g_aux_x, "$x", Index)
        end

        aux_y_idx = Matrix{Index}(undef, Lx, Ly)
        g_aux_y = g["aux_y_idx"]
        for x in 1:Lx, y in 1:Ly
            key = "$(x)_$(y)"
            if haskey(g_aux_y, key)
                aux_y_idx[x, y] = read(g_aux_y, key, Index)
            end
        end

        Ws = Matrix{ITensor}(undef, Lx, Ly)
        g_ws = g["tensors"]
        for x in 1:Lx, y in 1:Ly
            key = "$(x)_$(y)"
            if haskey(g_ws, key)
                Ws[x, y] = read(g_ws, key, ITensor)
            end
        end

        return ForkTensorNetworkOperator(Ws, phys_idx, aux_x_idx, aux_y_idx)
    end
end
