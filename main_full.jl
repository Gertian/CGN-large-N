using QuadGK
using Optim
using Roots
using LsqFit
using LaTeXStrings
using LinearAlgebra
using FileIO

######################
#Note!:
#       x,gx is coupling of sigma
#       y,gy is coupling of pi
#       m,mo is mass (perturbation)
#Note Note:
#   Daan defines x as g^2 <- independent of N
######################

function sc_ek(s, p, x, y, m, k)
    return sqrt(y^2*p^2+(x*s-m)^2*cos(k/2)^2 + 4*sin(k/2)^2 )
end
function sc_energy(s, p, x, y, m)
    a =  (x*s^2+y*p^2)/2 - quadgk( k -> 1/(2*pi)*sc_ek(s, p, x, y, m, k)   , -pi, pi, rtol=1e-12)[1]
    return a
end
function sc_dsigma(s, p, x, y, m)
    return x*s - (x*s-m)*x*quadgk(k -> (cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, m, k)), -pi, pi, rtol=1e-12)[1]
end
function sc_dpi(s, p, x, y, m)
    return y*p - quadgk(k -> (y^2*p)/(2*pi*sc_ek(s, p, x, y, m, k)), -pi, pi, rtol=1e-12)[1]
end
function sc_ss(s, p, x, y, m)
    out = 0.
    out += x
    out += quadgk(k -> (x^2*(x*s-m)^2*cos(k/2)^4)/(2*pi*sc_ek(s, p, x, y, m, k)^3), -pi, pi, rtol=1e-12)[1]
    out -= quadgk(k -> (x^2    *cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, m, k)  ), -pi, pi, rtol=1e-12)[1]
    return out
end
function sc_pp(s, p, x, y, m)
    out = 0.
    out += y
    out += quadgk(k -> (y^4*p^2)/(2*pi*sc_ek(s, p, x, y, m, k)^3), -pi, pi, rtol=1e-12)[1]
    out -= quadgk(k -> (y^2    )/(2*pi*sc_ek(s, p, x, y, m, k)  ), -pi, pi, rtol=1e-12)[1]
    return out
end
function sc_ps(s, p, x, y, m)
    out = 0.
    out += quadgk(k -> (x*(x*s-m)*y^2*p*cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, m, k)^3), -pi, pi, rtol=1e-12)[1]
end
function calc_order_parameters(x, y, m = 0)
    fun(order) = sc_energy(order[1], order[2], x, y, m)
    function der!(G, order)
        G[1] = sc_dsigma(order[1], order[2], x, y, m)
        G[2] =    sc_dpi(order[1], order[2], x, y, m)
    end
    function   h!(H, order)
        H[1,1] = sc_ss(order[1], order[2], x, y, m)
        H[1,2] = sc_ps(order[1], order[2], x, y, m)
        H[2,1] = H[1,2]
        H[2,2] = sc_pp(order[1], order[2], x, y, m)
    end
    if x == 0 && y == 0
        if m == 0
            return (0., 0., 0., 0., 0.)
        else
            return -m * quadgk(k -> cos(k/2)^2 / (2*pi*sc_ek(s,p,0,0,m,k)), -pi, pi, rtol=1e-12)[1]
        end
    else
        if m > 0
            lower = [0.]
            upper = [Inf]
            initial = [-1.,1.]
        else
            lower = [-Inf]
            upper = [0.]
            initial = [1.,1.]
        end
        a,b = optimize(fun,der!,h!,initial).minimizer
        if m == 0
            a = abs(a)
            b = abs(b)
            if x == 0
                a = 0
            elseif y == 0
                b = 0
            end
            return (a,b, atan(b/a),0.)
        else
            false_a = optimize(fun,der!,h!,lower,upper,-1 .*initial).minimizer[1]
        end
        return (a,b, atan(b/a),false_a)
    end
end
function calc_energy(x, y, m)
    sigma,pi,_,sigma_false = calc_order_parameters(x, y, m)
    return sc_energy(sigma, pi, x, y, m),sc_energy(sigma_false, pi, x, y, m)
end
function calc_lamhes(x ,y)
    sigma, pi, _ , _= calc_order_parameters(x,y)
    ss = sc_ss(sigma, pi, x, y, m)
    pp = sc_pp(sigma, pi, x, y, m)
    ps = sc_ps(sigma, pi, x, y, m)
    return (pp + ss)/2 - sqrt(ps^2 + (pp - ss)^2/4), (pp + ss)/2 + sqrt(ps^2 + (pp - ss)^2/4)
end
function plot_cut(x, ys, ms)
    if length(ys) == 1
        orders   = map(mi -> calc_order_parameters(x, ys, mi), ms )
        energies = map(mi -> calc_energy(x, ys, mi), ms)
        energies_true = map(Ei -> Ei[1], energies)
        energies_false = map(Ei -> Ei[2], energies)
        sigmas = map(xi -> xi[1], orders)
        sigmas_false = map(xi -> xi[4], orders)

        p1 = scatter(ms, sigmas, title = "at gx = $(x)", xlabel = L"ma", label = L"<\sigma>")
        scatter!(p1, ms, sigmas_false                                 , label = L"<\sigma_f>")

        p2 = scatter(ms, energies_true, title = "at gx = $(x)", xlabel = L"ma", label = L"<H>")

        p3 = scatter(sigmas, energies_true   , title = "at gx = $(x)", xlabel = L"<\sigma>", label = L"<H>")
        scatter!(p3, sigmas, energies_false                                               , label = L"<H_f>")

        E0s = map(sigma -> sc_energy(sigma,0., x, 0., 0.), sigmas)
        test_sigmas = range(-0.3, stop = 0.3, length = 100)
        chosen_m = 0.005
        #scatter!(p3, test_sigmas, map(test_sigma -> sc_energy(test_sigma,0.,x,0.,m), label = L"Theory")
        p4 = scatter(sigmas,E0s .+ chosen_m .* sigmas, title = "at g = $(1.3), m = $(chosen_m)", xlabel = L"<\sigma>", label = L"E_{MF,t}")
        scatter!(p4, test_sigmas,map(test_sigma -> sc_energy(test_sigma, 0, x, 0, chosen_m), test_sigmas), label = L"E_{calc}")
        scatter!(p4, sigmas_false,E0s .+ chosen_m .* sigmas_false, label = L"E_{MF,f}")
        return p1,p2,p3,p4

    elseif length(ms) == 1
        orders   = map(yi -> calc_order_parameters(x, yi, m)    , ys )
        energies = map(yi ->           calc_energy(x, yi, m)[1]    , ys )
        hess     = map(yi ->           calc_lamhes(x, yi, m)[1], ys )
        sigmas = map(xi -> xi[1], orders)
        pis    = map(xi -> xi[2], orders)

        p1 = scatter(gys, sigmas, title = "at gx = $(x)", xlabel = L"gy", label = L"<\sigma>")
        scatter!(p1, gys, pis                                            , label = L"<\pi>")

        p2 = scatter(gys, energies, title = "at gx = $(x)", xlabel = L"gy", label = L"<H>")

        p3 = scatter(gys, log.(hess)    , title = "at gx = $(x)", xlabel = L"gy", label = L"min(\lambda_{H})")

        return p1, p2, p3

    else
        println("Not yet implemented")
        # need 3D plot
    end
end
####################################
####################################
####################################
mutable struct state
    length::Int

    sigma_op
       pi_op

    sigma::Array{Float64}
       pii::Array{Float64}
       mis::Array{Float64}
end
function sigma(state::state, i,j,k)
    return state.sigma[mod1(i, state.length)]
end
function sigma!(state::state, i,j,k, new)
    state.sigma[mod1(i, state.length)] = new
end
function sigma_op(state::state, i,j,k)
    return state.sigma_op[mod1(i, state.length)]
end
function sigma(state::state, i,j,k,l)
    return 0.5*(sigma(state, i,j,k)-sigma(state, j,k,l))
end
function pii(state::state, i,j)
    return state.pii[mod1(i, state.length)]
end
function pii!(state::state, i,j, new)
    state.pii[mod1(i, state.length)] = new
end
function pii_op(state::state, i,j)
    return state.pi_op[mod1(i, state.length)]
end
function pii(state::state, i,j,k)
    return 0.5*(pii(state, i,j)-pii(state, j,k))
end
function calc_energy(state::state, vt,  gx, gy)
    indmax = Int(state.length / 2)
    Ht = make_H(state, gx, gy)
    norms = gx/4*norm(state.sigma)^2 + gy/4*norm(state.pii)^2
    energy = real(sum(diag(vt'*Ht*vt)[1:indmax]))+norms
    kinetic = real(sum(diag(vt'*make_K(state.length)*vt)[1:indmax]))
    #return energy, gx/4*norm(state.sigma)^2, gy/4*norm(state.pii)^2
    return kinetic, norms, energy - kinetic - norms
end
function calc_orderp_fluc(state::state, vt)
    sigmas_sq, pis_sq = [], []
    indmax = Int(state.length / 2)
    for i in 1:L
        tmp = real(sum(diag(vt'*sigma_op(state, i,i+1, i+2)*sigma_op(state, i,i+1, i+2)*vt)[1:indmax]))
        push!(sigmas_sq, tmp)
        tmp = real(sum(diag(vt'*pii_op(state, i,i+1, i+2)*pii_op(state, i,i+1, i+2)*vt)[1:indmax]))
        push!(pis_sq, tmp)
    end
    sigma_fluc = state.sigmas .^ 2 - sigmas_sq
    pi_fluc = state.pii .^ 2 - pis_sq
    return sigma_fluc,pi_fluc
end
function init_state(length, sigma_0, pi_0; BC = "triv", mis = Float64[])
    sis = Float64[]
    pis = Float64[]
    if BC == "triv"
        for loc in 1:length
            push!(sis, sigma_0*(-1)^loc)
            push!(pis,    pi_0*(-1)^loc)
        end
    elseif BC == "pi_kink"
        for loc in 1:length
            push!(sis, sigma_0*(-1)^loc)
        end
        for loc in 1:length/2-1
            push!(pis, pi_0*(-1)^loc)
        end
        push!(pis, 0.)
        for loc in length/2+1:length-1
            push!(pis, pi_0*(-1)^(loc+1))
        end
        push!(pis, 0.)
    elseif BC == "sigma_kink"
        for loc in 1:length/2-2
            push!(sis, sigma_0*(-1)^loc)
        end
        push!(sis, 0.)
        for loc in length/2:length-2
            push!(sis, sigma_0*(-1)^(loc+1))
        end
        push!(sis, 0.)
        push!(sis, sigma_0)
        for loc in 1:length
            push!(pis,    pi_0*(-1)^loc)
        end
    else
        throw("invalid BC errors")
    end

    si_ops = []
    for loc in 1:length
        tmp = ComplexF64.(zeros(length, length))
        tmp[mod1(loc  , length),mod1(loc+1, length)] = -0.5*im
        tmp[mod1(loc+1, length),mod1(loc+2, length)] = +0.5*im
        tmp += tmp'
        push!(si_ops, tmp)
    end

    pi_ops = []
    for loc in 1:length
        tmp = ComplexF64.(zeros(length, length))
        tmp[mod1(loc  , length),mod1(loc  , length)] =  1.
        tmp[mod1(loc+1, length),mod1(loc+1, length)] = -1.
        push!(pi_ops, tmp)
    end

    return state(length, si_ops, pi_ops, sis, pis, mis)
end
function create_ops(length)
    si_ops = []
    for loc in 1:length
        tmp = ComplexF64.(zeros(length, length))
        tmp[mod1(loc  , length),mod1(loc+1, length)] = -0.5*im
        tmp[mod1(loc+1, length),mod1(loc+2, length)] = +0.5*im
        tmp += tmp'
        push!(si_ops, tmp)
    end

    pi_ops = []
    for loc in 1:length
        tmp = ComplexF64.(zeros(length, length))
        tmp[mod1(loc  , length),mod1(loc  , length)] =  1.
        tmp[mod1(loc+1, length),mod1(loc+1, length)] = -1.
        push!(pi_ops, tmp)
    end
    return si_ops,pi_ops
end
function saveMF(filename, state, v, energy, t)
    tosave = Dict(   "sigmas"    => state.sigma
                    ,"pis"       => state.pii
                    ,"mis"       => state.mis
                    ,"v"         => v
                    ,"energy"    => energy
                    ,"length"     => state.length
                    )
    save(filename,tosave)
    println("State at t = $(t) has been saved")
end
function loadMF(filename)
    toload = load(filename)
    L = toload["length"]
    si_ops, pi_ops = create_ops(L)
    loaded_state = state(L, si_ops, pi_ops, toload["sigmas"], toload["pis"], toload["mis"]);
    return loaded_state, toload["v"], toload["energy"]
end
function eval_fluctuations(state, gx, gy)
    L = state.length

    H = make_H(state, gx, gy)
    eigs = eigen(H)
    eigvals = eigs.values
    v       = eigs.vectors
    indmax = findlast(xi -> xi<=0., eigvals)
    v = v'
    #at this point we have that eigvecs'*diagm(eigvals)*eigvecs = H
    fluc_sigma = Float64[]
    fluc_pi    = Float64[]
    for i in 1:L
        tmp = real(sum(diag(v*sigma_op(state, i,i+1, i+2)^2*v')[1:indmax]))
        push!(fluc_sigma, tmp)

        tmp = real(sum(diag(v*pii_op(state, i,i+1)^2*v')[1:indmax]))
        push!(fluc_pi, tmp)
    end
    return fluc_sigma, fluc_pi
end
function make_H(state::state, gx, gy)
    L=state.length
    out = ComplexF64.(zeros(L,L))
    for i in 1:L
        out[mod1(i,L)  ,mod1(i,L)]   +=      gy*pii(state, i-1,i,i+1)
        out[mod1(i+1,L),mod1(i  ,L)] +=      im+0.5*im*(gx*sigma(state, i-1,i,i+1,i+2)+state.mis[i]*(-1)^i)
        out[mod1(i  ,L),mod1(i+1,L)] += conj(im+0.5*im*(gx*sigma(state, i-1,i,i+1,i+2) +state.mis[i]*(-1)^i) )
    end
    return out
end
function make_K(L)
    out = ComplexF64.(zeros(L,L))
    for i in 1:L
        out[mod1(i+1,L),mod1(i  ,L)] +=      im
        out[mod1(i  ,L),mod1(i+1,L)] +=     -im
    end
    return out
end
function update_state(state::state, gx, gy)
    L = state.length

    H = make_H(state, gx, gy)
    eigs = eigen(H)
    eigvals = eigs.values
    v       = eigs.vectors
    indmax = findlast(xi -> xi<=0., eigvals)
    #@show eigvals[1:3]
    #v = v'
    #at this point we have that eigvecs'*diagm(eigvals)*eigvecs = H
    for i in 1:L
        tmp = real(sum(diag(v'*sigma_op(state, i,i+1, i+2)*v)[1:indmax]))
        sigma!(state, i, i+1, i+2, tmp)

        tmp = real(sum(diag(v'*pii_op(state, i,i+1)*v)[1:indmax]))
        pii!(state, i, i+1, tmp)
    end
    energy = sum(eigvals[1:indmax]) + gx/4*norm(state.sigma)^2 + gy/4*norm(state.pii)^2

    return state, energy
end
function evolve_state(dt, state::state, v, gx, gy)
    L = state.length
    indmax = Int(L/2)
    H = make_H(state, gx, gy)
    evolved_v = exp(-im*dt*H)*v
    for i in 1:L
        tmp = real(sum(diag(evolved_v'*sigma_op(state, i,i+1, i+2)*evolved_v)[1:indmax]))
        sigma!(state, i, i+1, i+2, tmp)

        tmp = real(sum(diag(evolved_v'*pii_op(state, i,i+1)*evolved_v)[1:indmax]))
        pii!(state, i, i+1, tmp)
    end
    energy = sum(calc_energy(state, evolved_v, gx, gy))
    return state, energy, evolved_v
end
function plot_state(gx, gy, m, state; BC = "triv")
    s0 , p0, _ , s0_false= calc_order_parameters(gx, gy, m)
    s0_negm , p0_negm, _ , s0_false_negm= calc_order_parameters(gx, gy, -m)
    iis = Int.([range(0, stop=state.length, length=state.length+1)...])
    if BC == "triv"
        a = plot(title = "order parameters")
        hline!([s0], label = "vaccum expectance sigma", color = "blue", linestyle = :dot)
        hline!([s0_negm], label = "s0_negm", color = "red", linestyle = :dot)
        hline!([s0_false],label = "s0_false", color = "green", linestyle = :dot)
        if gy != 0
            hline!([p0], label = "vaccum expectance  pi", color = "orange", linestyle = :dot)
        end
        scatter!(iis.+0.5, map(ii-> (-1)^ii*pii(state, ii, ii+1),iis), color = "orange", label = L"pi_{i, i+1}" )
        scatter!(iis    , map(ii-> (-1)^(ii-1)*sigma(state, ii-1, ii, ii+1),iis), color = "blue"  , label = L"\sigma_{i-1, i, i+1}")


        return plot(a, size = (2000, 1000))
    elseif BC == "fluc_triv"
        a = plot(title = "fluctuations of the order parameters")
        hline!([s0], label = "vaccum expectance simga", color = "blue", linestyle = :dot)
        hline!([p0], label = "vaccum expectance  pi", color = "orange", linestyle = :dot)

        fs, fp = eval_fluctuations(state, gx, gy)

        scatter!(iis.+0.5, fp, color = "orange", label = L"pi^2_{i, i+1}" )
        scatter!(iis     , fs, color = "blue"  , label = L"\sigma^2_{i-1, i, i+1}")


        return plot(a, size = (2000, 1000))
    elseif BC == "pi_kink"
        a = plot(title = "order parameters")
        hline!([s0], label = "vaccum expectance simga", color = "blue", linestyle = :dot)
        hline!([p0, -p0], label = "vaccum expectance  pi", color = "orange", linestyle = :dot)
        vline!([0.5, state.length/2+0.5, state.length+0.5], color = "red", label = "kink locations")

        plot!(iis.+0.5, map(ii-> (-1)^ii*pii(state, ii, ii+1),iis), color = "orange", label = L"pi_{i, i+1}", marker = :dot )
        plot!(iis    , map(ii-> (-1)^(ii-1)*sigma(state, ii-1, ii, ii+1),iis), color = "blue"  , label = L"\sigma_{i-1, i, i+1}", marker = :dot)

        return plot(a, size = (2000, 1000))
    elseif BC == "sigma_kink"
        a = plot(title = "order parameters")
        hline!([s0, -s0], label = "vaccum expectance simga", color = "blue", linestyle = :dot)
        hline!([p0], label = "vaccum expectance  pi", color = "orange", linestyle = :dot)
        vline!([0., state.length/2, state.length], color = "red", label = "kink locations")

        plot!(iis.+0.5, map(ii-> (-1)^ii*pii(state, ii, ii+1),iis), color = "orange", label = L"pi_{i, i+1}", marker = :dot )
        plot!(iis    , map(ii-> (-1)^(ii-1)*sigma(state, ii-1, ii, ii+1),iis), color = "blue"  , label = L"\sigma_{i-1, i, i+1}", marker = :dot)

        return plot(a, size = (2000, 1000))
    else
        throw("BC invalid")
    end
end
function make_datapoint(gx, gy, L; tol = 1e-2)
    @assert iseven(L)

    s0, p0, _ = calc_order_parameters(gx, gy)
    e0 = L/2*calc_energy(gx, gy)

    triv = init_state(L, s0, p0; BC = "triv")
    terr = Inf
    tene = Inf
    while terr > tol
        triv, tene_n = update_state(triv, gx, gy)
        terr = abs(tene-tene_n)
        tene = tene_n
        @show tene, terr
    end
    plot_triv = plot_state(gx, gy, triv; BC = "triv")
    plot_fluc = plot_state(gx, gy, triv; BC = "fluc_triv")
    plot!(plot_triv, title = "E_{ref} = $(tene) vs $(e0)")
    display(plot_triv)

    piki = init_state(L, s0, p0; BC = "pi_kink")
    perr = Inf
    pene = Inf
    while perr > tol
        piki, pene_n = update_state(piki, gx, gy)
        perr = abs(pene-pene_n)
        pene = pene_n
        @show pene, perr
    end
    plot_piki = plot_state(gx, gy, piki; BC = "pi_kink")
    plot!(plot_piki, title = "E = $(pene-tene)")
    display(plot_piki)

    siki = init_state(L, s0, p0; BC = "sigma_kink")
    serr = Inf
    sene = Inf
    while serr > tol
        siki, sene_n = update_state(siki, gx, gy)
        serr = abs(sene-sene_n)
        sene = sene_n
        @show sene, serr
    end
    plot_siki = plot_state(gx, gy, siki; BC = "sigma_kink")
    plot!(plot_siki, title = "E_ref = $(sene-tene)")
    display(plot_siki)
    @show sene

    return plot_triv, plot_piki, plot_siki, plot_fluc, tene, pene, sene
end
function potential_trapping(gx, L, mo, trial_mo; tol = 1e-5)

    function make_mis(L,mo)
        ms = []
        for i in 1:Int(L/2-20)
            push!(ms,-mo)
        end
        for i in Int(L/2-19):Int(L/2+19)
            push!(ms,mo)
        end
        for i in Int(L/2+20):L
            push!(ms,-mo)
        end
        return ms
    end

    @assert iseven(L)
    gy = 0.
    s0, p0, _ = calc_order_parameters(gx, gy, trial_mo)
    e0 = L/2*calc_energy(gx, gy, trial_mo)[1]
    #potential trapping for initial state
    triv = init_state(L, s0, p0; BC = "triv", mis = make_mis(L,trial_mo))

    terr = Inf
    tene = Inf
    while terr > tol
        triv, tene_n = update_state(triv, gx, gy)
        terr = abs(tene-tene_n)
        tene = tene_n
        @show tene, terr
        #plot_triv = plot_state(gx, gy, trial_mo, triv; BC = "triv")
        #display(plot_triv)
    end
    s0, p0, _ = calc_order_parameters(gx, gy, mo)
    triv.mis = make_mis(L,mo)
    plot_triv = 0

    terr = Inf
    tene = Inf
    while terr > tol
        triv, tene_n = update_state(triv, gx, gy)
        terr = abs(tene-tene_n)
        tene = tene_n
        @show tene, terr
        #plot_triv = plot_state(gx, gy, mo, triv; BC = "triv")
        #display(plot_triv)
    end
    triv.mis = [range(mo;stop=mo,length=L)...]
    return triv

    terr = Inf
    tene = Inf
    while terr > tol
        triv, tene_n = update_state(triv, gx, gy)
        terr = abs(tene-tene_n)
        tene = tene_n
        @show tene, terr
        #plot_triv = plot_state(gx, gy, mo, triv; BC = "triv")
        #display(plot_triv)
    end
    plot_triv = plot_state(gx, gy, mo, triv; BC = "triv")
    #plot(plot_triv, title = "m = $(mo), t = 0")
    return triv
end
function initial_perturbation(PerturbationRange,gx, L, mo; tol=1e-14)
    @assert iseven(L)
    gy = 0.
    s0, p0, _ ,s0_false = calc_order_parameters(gx, gy, mo)
    triv = init_state(L, s0_false, p0; BC = "triv", mis = [range(mo;stop=mo,length=L)...])

    terr = Inf
    tene = Inf
    while terr > tol
        triv, tene_n = update_state(triv, gx, gy)
        terr = abs(tene-tene_n)
        tene = tene_n
        @show tene, terr
        #plot_triv = plot_state(gx, gy, mo, triv; BC = "triv")
        #display(plot_triv)
    end
    for site in PerturbationRange
        sigma!(triv,site,site+1,site+2,-sigma(triv,site,site+1,site+2))
        #sigma!(triv,site,site+1,site+2,(-1)^(site)*s0)
    end
    #plot_triv = plot_state(gx, gy, mo, triv; BC = "triv")
    #display(plot_triv)
    #savefig(plot(plot_triv),"test.pdf")
    return triv
end
function time_evolution(state, v, dt, N, gx, gy, mo, topfolder; starttime = 0.)
    t = starttime
    savestep = Int(round(0.1/dt))
    for step in 1:N
        state, energy, v = evolve_state(dt, state, v, gx, gy)
        t = round(step*dt+starttime; digits=floor(Int, -log(dt)))
        if step % savestep == 0.
            name = topfolder*"MF__gx_$(gx)__gy_$(gy)__mo_$(mo)__t_$(t)__dt_$(dt).jld2"
            saveMF(name, state, v, energy, t)
        end
        #state_plot = plot_state(gx, gy, mo, state; BC = "triv")
        #display(state_plot)
        @show (t,energy)
    end
    #state_plot = plot_state(gx, gy, mo, state; BC = "triv")
    #plot(state_plot, title = "t=$(N*dt), dt=$(dt)")
    return state
end
