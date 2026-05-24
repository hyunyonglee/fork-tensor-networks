using ITensors

# ─── SiteType definition ───

function ITensors.space(::SiteType"2SiteFermion";
    conserve_qns=true,
    conserve_sz="")

    if conserve_qns && conserve_sz == "Up"
        return [
            QN(("Nf", 0, -1), ("Sz", 0)) => 1,     # |00⟩
            QN(("Nf", 1, -1), ("Sz", 1)) => 2,     # |10⟩, |01⟩
            QN(("Nf", 2, -1), ("Sz", 2)) => 1,     # |11⟩
        ]
    elseif conserve_qns && conserve_sz == "Dn"
        return [
            QN(("Nf", 0, -1), ("Sz", 0)) => 1,
            QN(("Nf", 1, -1), ("Sz", -1)) => 2,
            QN(("Nf", 2, -1), ("Sz", -2)) => 1,
        ]
    else
        return 4
    end
end


# ─── States ───

ITensors.state(::StateName"Emp", ::SiteType"2SiteFermion") = [1.0, 0, 0, 0]
ITensors.state(::StateName"Occ1", ::SiteType"2SiteFermion") = [0, 1.0, 0, 0]
ITensors.state(::StateName"Occ2", ::SiteType"2SiteFermion") = [0, 0, 1.0, 0]
ITensors.state(::StateName"Occ12", ::SiteType"2SiteFermion") = [0, 0, 0, 1.0]
ITensors.state(::StateName"Bond", ::SiteType"2SiteFermion") = [0, inv(sqrt(2.0)), inv(sqrt(2.0)), 0]


# ─── Operators ───

# Identity
function ITensors.op(::OpName"I", ::SiteType"2SiteFermion")
    return [1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 1]
end

# Number operators
function ITensors.op(::OpName"N1", ::SiteType"2SiteFermion")
    return [0 0 0 0
            0 1 0 0
            0 0 0 0
            0 0 0 1]
end

function ITensors.op(::OpName"N2", ::SiteType"2SiteFermion")
    return [0 0 0 0
            0 0 0 0
            0 0 1 0
            0 0 0 1]
end

function ITensors.op(::OpName"Ntot", ::SiteType"2SiteFermion")
    return [0 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 2]
end

# Creation operators
# d1†: bare (no JW string needed)
function ITensors.op(::OpName"d1†", ::SiteType"2SiteFermion")
    return [0 0 0 0
            1 0 0 0
            0 0 0 0
            0 0 1 0]
end

# d1 (annihilation)
function ITensors.op(::OpName"d1", ::SiteType"2SiteFermion")
    return [0 1 0 0
            0 0 0 0
            0 0 0 1
            0 0 0 0]
end

# d2†: bare (no JW string)
function ITensors.op(::OpName"d2†", ::SiteType"2SiteFermion")
    return [ 0 0 0 0
             0 0 0 0
             1 0 0 0
             0 1 0 0]
end

# d2: bare (no JW string)
function ITensors.op(::OpName"d2", ::SiteType"2SiteFermion")
    return [0 0 1 0
            0 0 0 1
            0 0 0 0
            0 0 0 0]
end

# Physical d2† = p1 * d2†_bare (for Green's function)
function ITensors.op(::OpName"D2†", ::SiteType"2SiteFermion")
    return [ 0 0 0 0
             0 0 0 0
             1 0 0 0
             0 -1 0 0]
end

# Physical d2 = p1 * d2_bare (for Green's function)
function ITensors.op(::OpName"D2", ::SiteType"2SiteFermion")
    return [0 0 1 0
            0 0 0 -1
            0 0 0 0
            0 0 0 0]
end

# Fermi sign operators
# F1 = (-1)^{n1}
function ITensors.op(::OpName"F1", ::SiteType"2SiteFermion")
    return [1  0 0  0
            0 -1 0  0
            0  0 1  0
            0  0 0 -1]
end

# F2 = (-1)^{n2}
function ITensors.op(::OpName"F2", ::SiteType"2SiteFermion")
    return [1 0  0  0
            0 1  0  0
            0 0 -1  0
            0 0  0 -1]
end

# F1F2 = (-1)^{n1+n2}
function ITensors.op(::OpName"F1F2", ::SiteType"2SiteFermion")
    return [1  0  0 0
            0 -1  0 0
            0  0 -1 0
            0  0  0 1]
end
