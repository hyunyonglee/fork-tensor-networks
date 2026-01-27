#!/home/heungsikim//.local/bin/julia
#!/home/heungsikim//.local/bin/julia

# using QuadGK

function get_J_square(D,_ome)
    return (D/(2*pi)) .* sqrt.(1 .- ((_ome)/D) .^(2))

end


function get_Re_hyb(ω, D) #get real part of hybridization function by using kronig-kramers relation
    ϵ  = 1e-10 
    integranda(ωp) = -get_J_square(D, ωp) / (ωp - ω)
    left_part, _ = quadgk(integranda, -D , - ϵ + ω )
    right_part, _ = quadgk(integranda, ω+ ϵ, D )
    return (left_part + right_part) / π
    # end
end


function get_Half_elliptical_Bath_Fit(D,N_bath)

    # small positive number for the pole

    omega_arr = Vector(range(-D,D,N_bath+1))

    integrand(ωp) =  get_J_square(D,ωp)  #Integration for the discritization of V
    integrandx(ωp) =  get_J_square(D,ωp)*ωp #Integration for the discritization of Bath Energy

    abs_V_arr = []
    e_bath_arr = []

    for I = 1:length(omega_arr)-1

        abs_V_square, _ = quadgk(integrand, omega_arr[I], omega_arr[I+1])
        push!(abs_V_arr,sqrt(abs_V_square))

        e_bath_comp, _ = quadgk(integrandx, omega_arr[I], omega_arr[I+1])
        push!(e_bath_arr,e_bath_comp/abs_V_square)
    end

    Im_hyb_arr = get_J_square.(D,e_bath_arr) .*(-pi)
    Re_hyb_arr =[]

    for e_bath = e_bath_arr
        Re_hyb = get_Re_hyb(e_bath, D)
        push!(Re_hyb_arr,Re_hyb)
    end

    phase_arr = exp.(1im .*atan.(Im_hyb_arr , Re_hyb_arr))

    return abs_V_arr .* phase_arr, e_bath_arr
end
