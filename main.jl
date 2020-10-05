using Plots
pyplot()
using QuadGK
using Optim
using Roots
using LsqFit
using LaTeXStrings
using LinearAlgebra

function sc_ek(s, p, x, y, k)
    return sqrt(y^2*p^2 + x^2*s^2*cos(k/2)^2 + 4*sin(k/2)^2 )
end
function sc_energy(s, p, x, y)
    a =  (x*s^2+y*p^2)/2 - quadgk( k -> 1/(2*pi)*sc_ek(s,p,x,y,k)   , -pi, pi, rtol=1e-12)[1]
    return a
end

function sc_dsigma(s, p, x, y)
    return x*s - quadgk(k -> (x^2*s*cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, k)), -pi, pi, rtol=1e-12)[1]
end
function sc_dpi(s, p, x, y)
    return y*p - quadgk(k -> (y^2*p)/(2*pi*sc_ek(s, p, x, y, k)), -pi, pi, rtol=1e-12)[1]
end
function sc_ps(s, p, x, y)
    out = 0.
    out += x
    out += quadgk(k -> (x^4*s^2*cos(k/2)^4)/(2*pi*sc_ek(s, p, x, y, k)^3), -pi, pi, rtol=1e-12)[1]
    out -= quadgk(k -> (x^2    *cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, k)  ), -pi, pi, rtol=1e-12)[1]
    return out
end
function sc_pp(s, p, x, y)
    out = 0.
    out += y
    out += quadgk(k -> (y^4*p^2)/(2*pi*sc_ek(s, p, x, y, k)^3), -pi, pi, rtol=1e-12)[1]
    out -= quadgk(k -> (y^2    )/(2*pi*sc_ek(s, p, x, y, k)  ), -pi, pi, rtol=1e-12)[1]
    return out
end
function sc_pm(s, p, x, y)
    out = 0.
    out += quadgk(k -> (x^2*y^2*s*p*cos(k/2)^2)/(2*pi*sc_ek(s, p, x, y, k)^3), -pi, pi, rtol=1e-12)[1]
end
function calc_order_parameters(x, y)
    fun(order) = sc_energy(order[1], order[2], x, y)
    function der!(G, order)
        G[1] = sc_dsigma(order[1], order[2], x, y)
        G[2] =    sc_dpi(order[1], order[2], x, y)
    end
    function   h!(H, order)
        H[1,1] = sc_ps(order[1], order[2], x, y)
        H[1,2] = sc_pm(order[1], order[2], x, y)
        H[2,1] = H[1,2]
        H[2,2] = sc_pp(order[1], order[2], x, y)
    end
    if x == 0 && y == 0
        return (0., 0., 0.)
    else
        (a,b) = abs.(optimize(fun, der!, h!,  [1., 1.]).minimizer)
        if x == 0
            a = 0
        elseif y == 0
            b = 0
        end
        return (a,b, atan(b/a))
    end
end
function calc_energy(x, y)
    sigma, pi, _ = calc_order_parameters(x,y)
    return sc_energy(sigma, pi, x, y)
end
function calc_lamhes(x ,y)
    sigma, pi, _ = calc_order_parameters(x,y)
    ps = sc_ps(sigma, pi, x, y)
    pp = sc_pp(sigma, pi, x, y)
    pm = sc_pm(sigma, pi, x, y)
    return (pp + ps)/2 - sqrt(pm^2 + (pp - ps)^2/4), (pp + ps)/2 + sqrt(pm^2 + (pp - ps)^2/4)
end

##########3
#make some plots specific for my CGN project
##########

function plot_cut(gx, gys)
    orders   = map(gyi -> calc_order_parameters(gx, gyi)    , gys )
    energies = map(gyi ->           calc_energy(gx, gyi)    , gys )
    hess     = map(gyi ->           calc_lamhes(gx, gyi)[1], gys )
    sigmas = map(xi -> xi[1], orders)
    pis    = map(xi -> xi[2], orders)

    p1 = scatter(gys, sigmas, title = "at gx = $(gx)", xlabel = L"gy", label = L"<\sigma>")
    scatter!(p1, gys, pis                                            , label = L"<\pi>")

    p2 = scatter(gys, energies, title = "at gx = $(gx)", xlabel = L"gy", label = L"<H>")

    p3 = scatter(gys, log.(hess)    , title = "at gx = $(gx)", xlabel = L"gy", label = L"min(\lambda_{H})")

    return p1, p2, p3
end

p1, p2, p3 = plot_cut(2., range(1.2, stop = 1.3, length = 100))
out = plot(p1,p2,p3, size = (2000, 1000))
savefig(out, "cut_MF")

##################################################################
##################################################################
##################################################################

mutable struct state
    length::Int

    sigma_op
       pi_op

    sigma::Array{Float64}
       pii::Array{Float64}
end
function sigma(state, i,j,k)
    return state.sigma[mod1(i, state.length)]
end
function sigma!(state, i,j,k, new)
    state.sigma[mod1(i, state.length)] = new
end
function sigma_op(state, i,j,k)
    return state.sigma_op[mod1(i, state.length)]
end
function sigma(state, i,j,k,l)
    return 0.5*(sigma(state, i,j,k)-sigma(state, j,k,l))
end

function pii(state, i,j)
    return state.pii[mod1(i, state.length)]
end
function pii!(state, i,j, new)
    state.pii[mod1(i, state.length)] = new
end
function pii_op(state, i,j)
    return state.pi_op[mod1(i, state.length)]
end
function pii(state, i,j,k)
    return 0.5*(pii(state, i,j)-pii(state, j,k))
end

function init_state(length, sigma_0, pi_0; BC = "triv")
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

    return state(length, si_ops, pi_ops, sis, pis)
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
function make_H(state, gx, gy)
    L=state.length
    out = ComplexF64.(zeros(L,L))
    for i in 1:L
        out[mod1(i,L)  ,mod1(i,L)]   +=      gy*pii(state, i-1,i,i+1)
        out[mod1(i+1,L),mod1(i  ,L)] +=      im+0.5*im*gx*sigma(state, i-1,i,i+1,i+2)
        out[mod1(i  ,L),mod1(i+1,L)] += conj(im+0.5*im*gx*sigma(state, i-1,i,i+1,i+2)  )
    end
    return out
end
function update_state(state, gx, gy)
    L = state.length

    H = make_H(state, gx, gy)
    eigs = eigen(H)
    eigvals = eigs.values
    v       = eigs.vectors
    indmax = findlast(xi -> xi<=0., eigvals)
    v = v'
    #at this point we have that eigvecs'*diagm(eigvals)*eigvecs = H
    for i in 1:L
        tmp = real(sum(diag(v*sigma_op(state, i,i+1, i+2)*v')[1:indmax]))
        sigma!(state, i, i+1, i+2, tmp)

        tmp = real(sum(diag(v*pii_op(state, i,i+1)*v')[1:indmax]))
        pii!(state, i, i+1, tmp)
    end
    energy = sum(eigvals[1:indmax]) + gx/4*norm(state.sigma)^2 + gy/4*norm(state.pii)^2

    return state, energy
end
function plot_state(gx, gy, state; BC = "triv")
    s0 , p0, _ = calc_order_parameters(gx, gy)
    iis = Int.([range(0, stop=state.length, length=state.length+1)...])
    if BC == "triv"
        a = plot(title = "order parameters")
        hline!([s0], label = "vaccum expetance simga", color = "blue", linestyle = :dot)
        hline!([p0], label = "vaccum expetance  pi", color = "orange", linestyle = :dot)

        scatter!(iis.+0.5, map(ii-> (-1)^ii*pii(state, ii, ii+1),iis), color = "orange", label = L"pi_{i, i+1}" )
        scatter!(iis    , map(ii-> (-1)^(ii-1)*sigma(state, ii-1, ii, ii+1),iis), color = "blue"  , label = L"\sigma_{i-1, i, i+1}")


        return plot(a, size = (2000, 1000))
    elseif BC == "fluc_triv"
        a = plot(title = "fluctuations of the order parameters")
        hline!([s0], label = "vaccum expetance simga", color = "blue", linestyle = :dot)
        hline!([p0], label = "vaccum expetance  pi", color = "orange", linestyle = :dot)

        fs, fp = eval_fluctuations(state, gx, gy)

        scatter!(iis.+0.5, fp, color = "orange", label = L"pi^2_{i, i+1}" )
        scatter!(iis     , fs, color = "blue"  , label = L"\sigma^2_{i-1, i, i+1}")


        return plot(a, size = (2000, 1000))
    elseif BC == "pi_kink"
        a = plot(title = "order parameters")
        hline!([s0], label = "vaccum expetance simga", color = "blue", linestyle = :dot)
        hline!([p0, -p0], label = "vaccum expetance  pi", color = "orange", linestyle = :dot)
        vline!([0.5, state.length/2+0.5, state.length+0.5], color = "red", label = "kink locations")

        plot!(iis.+0.5, map(ii-> (-1)^ii*pii(state, ii, ii+1),iis), color = "orange", label = L"pi_{i, i+1}", marker = :dot )
        plot!(iis    , map(ii-> (-1)^(ii-1)*sigma(state, ii-1, ii, ii+1),iis), color = "blue"  , label = L"\sigma_{i-1, i, i+1}", marker = :dot)

        return plot(a, size = (2000, 1000))
    elseif BC == "sigma_kink"
        a = plot(title = "order parameters")
        hline!([s0, -s0], label = "vaccum expetance simga", color = "blue", linestyle = :dot)
        hline!([p0], label = "vaccum expetance  pi", color = "orange", linestyle = :dot)
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




ref,piking,sigmakink,flucplot,_=make_datapoint(4.5, 1.5, 100; tol = 1e-12)
savefig(plot(ref, piking, sigmakink, flucplot), "MF_kink")


