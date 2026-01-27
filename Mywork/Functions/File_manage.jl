import HDF5


function save_dmrg(Path,DMRG_Result)
    f = HDF5.h5open("Path","w")
    write(f,"FTNS",DMRG_Result["Ground state"])
    write(f,"FTNO",DMRG_Result["Hamiltonian"])
    write(f,"Ground energy",DMRG_Result["Ground energy"])
    close(f)
end

function load_dmrg()
    f = HDF5.h5open("Path","r")
    FTNS = read(f,"FTNS")
    FTNO = read(f,"FTNO")
    Ground_energy = read(f,"Ground energy")
    close(f)
    return FTNS, FTNO
end