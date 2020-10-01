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

####################################
####################################
####################################




mutable struct state
    length::Int

    sigma_op_i
       pi_op_i

    sigma_i::Array{Float64}
       pi_i::Array{Float64}
end
function init_state(length, sigma_0, pi_0; BC = "triv")
    @assert length > 2
    @assert iseven(length)
    sigmas = [range(sigma_0, sigma_0, length=length)...]
    sigma_ops = []
    for i in 1:Int(floor(length/2))
        sigmas[i] = (sigmas[i]*(-1)^i)
    end
    if BC == "sigma_kink"
        for i in Int(floor(length/2))+1:length
            sigmas[i] = sigmas[i]*(-1)^(i+1)
        end
    else
        for i in Int(floor(length/2))+1:length
            sigmas[i] = sigmas[i]*(-1)^(i)
        end
    end

    for i in 1:length-2
        tmp = zeros(length, length) + im*zeros(length, length)
        tmp[i,   i+1] = -0.5*im
        tmp[i+1, i  ] =  0.5*im
        tmp[i+1, i+2] =  0.5*im
        tmp[i+2, i+1] = -0.5*im
        push!(sigma_ops, tmp)
    end
    tmp = zeros(length, length) + im*zeros(length, length)
    tmp[length-1, length] = -0.5*im
    tmp[length, length-1] =  0.5*im
    tmp[length, 1] =  0.5*im
    tmp[1, length] = -0.5*im
    push!(sigma_ops, tmp)
    tmp = zeros(length, length) + im*zeros(length, length)
    tmp[length, 1] = -0.5*im
    tmp[1, length] =  0.5*im
    tmp[1, 2] =  0.5*im
    tmp[2, 1] = -0.5*im
    push!(sigma_ops, tmp)

    pis    = [range(pi_0, pi_0, length=length)...]
    pi_ops = []
    for i in 1:Int(floor(length/2))
        pis[i] = pis[i]*(-1)^i
    end
    if BC == "pi_kink"
        for i in Int(floor(length/2))+1:length
            pis[i] = pis[i]*(-1)^(i+1)
        end
    else
        for i in Int(floor(length/2))+1:length
            pis[i] = pis[i]*(-1)^(i)
        end
    end

    for i in 1:length-1
        tmp = zeros(length, length) + im*zeros(length, length)
        tmp[i  , i  ] =  1
        tmp[i+1, i+1] = -1
        push!(pi_ops, tmp)
    end
    tmp = zeros(length, length) + im*zeros(length, length)
    tmp[length  , length  ] =  1
    tmp[1, 1] = -1
    push!(pi_ops, tmp)
    return state(length, sigma_ops, pi_ops, sigmas, pis)
end
function get_sigma(state::state, i::Int64, j::Int64, k::Int64)
    @assert i+1 == j
    @assert j+1 == k
    return state.sigma_i[mod1(i, state.length)]
end
function get_pi(state::state, i::Int64, j::Int64)
    return state.pi_i[mod1(i, state.length)]
end
function construct_H(state, gx, gy)
    L = state.length
    #first we construct the hamiltonian and diagonalize it
    H = zeros(L, L) + im*zeros(L, L)
    for i in 1:L
        H[i,i] += gy*0.5*(get_pi(state, i-1,i)-get_pi(state, i, i+1)) + gx*get_sigma(state, i-1,i,i+1)^2/2 + gy*get_pi(state, i, i+1)^2/2
    end
    for i in 1:L-1
        H[i, i+1] = -im*(1+0.5*gx*0.5*(get_sigma(state,i-1,i,i+1)-get_sigma(state, i,i+1,i+2)))
        H[i+1, i] = H[i, i+1]'
    end
    H[1, L] = im*(1+0.5*gx*0.5*(get_sigma(state,L-1,L,L+1)-get_sigma(state, L,L+1,L+2)))
    H[L, 1] = H[1, L]'
    return H
end
function construct_H_blocks(state, gx, gy)
    L = state.length
    out = []
    for loc in 1:L-1
        tmp = zeros(L, L) + im*zeros(L, L)
        tmp[loc,loc]     += 0.5*(gy*0.5*(get_pi(state, loc-1,loc)-get_pi(state, loc  , loc+1)) + gx*get_sigma(state, loc-1,loc  ,loc+1)^2/2 + gy*get_pi(state, loc  , loc+1)^2/2)
        tmp[loc+1,loc+1] += 0.5*(gy*0.5*(get_pi(state, loc,loc+1)-get_pi(state, loc+1, loc+2)) + gx*get_sigma(state, loc  ,loc+1,loc+2)^2/2 + gy*get_pi(state, loc+1, loc+2)^2/2)

        tmp[loc, loc+1] = -im*(1+0.5*gx*0.5*(get_sigma(state,loc-1,loc,loc+1)-get_sigma(state, loc,loc+1,loc+2)))
        tmp[loc+1, loc] = tmp[loc, loc+1]'
        push!(out, tmp)
    end
    tmp = zeros(L, L) + im*zeros(L, L)
    tmp[1,1] += 0.5*(gy*0.5*(get_pi(state, 0,1)-get_pi(state, 1, 2)) + gx*get_sigma(state, 0,1  ,2)^2/2 + gy*get_pi(state, 1  , 2)^2/2)
    tmp[L,L] += 0.5*(gy*0.5*(get_pi(state, L-1,L)-get_pi(state, L, L+1)) + gx*get_sigma(state, L-1  ,L,L+1)^2/2 + gy*get_pi(state, L, L+1)^2/2)
    tmp[1, L] = im*(1+0.5*gx*0.5*(get_sigma(state,L-1,L,L+1)-get_sigma(state, L,L+1,L+2)))
    tmp[L, 1] = tmp[1, L]'
    push!(out, tmp)

    return out
end
function update_state(state, gx, gy)
    L = state.length
    H = construct_H(state, gx, gy)
    decomp = eigen(H)
    eigvals = decomp.values
    v       = decomp.vectors

    energy = sum(eigvals[1:Int(floor(L/2))])
    v = v'
    #at this point we have that eigvecs'*diagm(eigvals)*eigvecs = H
    error = 0.
    for i in 1:L
        tmp = real(sum(diag(v*state.sigma_op_i[i]*v')[1:Int(floor(L/2))]))
        error += abs(state.sigma_i[i] - tmp)
        state.sigma_i[i] = tmp

        tmp = real(sum(diag(v*state.pi_op_i[i]*v')[1:Int(floor(L/2))]))
        error += abs(state.pi_i[i] - tmp)
        state.pi_i[i] = tmp
    end
    return state, energy, error
end
function energy_state(state, gx, gy)
    L = state.length
    H  = construct_H(state, gx, gy)
    Hi = construct_H_blocks(state, gx, gy)
    decomp = eigen(H)
    eigvals = decomp.values
    v       = decomp.vectors
    v = v'

    energy = sum(eigvals[1:Int(floor(L/2))])
    energies = Float64[]
    for i in 1:L-1
        push!(energies, 0.5*real(sum(diag(v*Hi[i]*v')[1:Int(floor(L/2))]))+0.5*real(sum(diag(v*Hi[i+1]*v')[1:Int(floor(L/2))])))
    end
        push!(energies, 0.5*real(sum(diag(v*Hi[L]*v')[1:Int(floor(L/2))]))+0.5*real(sum(diag(v*Hi[1]*v')[1:Int(floor(L/2))])))
    return energy, energies
end
function plot_state(state::state, gx, gy; BC = "trivial")
    is = Int[range(1, state.length, length=state.length)...]
    eref = 0.5*calc_energy(gx, gy)
    _, energies = energy_state(state, gx, gy)
    s0, p0 = calc_order_parameters(gx, gy)

    @show eref
    @show energies

    a = plot(xlabel = "i")
    hline!([s0], color = "orange", label = "", linestyle = :dot)
    hline!([p0], color = "blue"  , label = "", linestyle = :dot)
    if BC == "sigma_kink"
        hline!([-s0], color = "orange", label = "", linestyle = :dot)
    elseif BC == "pi_kink"
        hline!([-p0], color = "blue"  , label = "", linestyle = :dot)
    end
    plot!(is, map(ii -> (-1)^ii*get_sigma(state,ii, ii+1, ii+2 ), is) , color = "orange", label = L"<\sigma_{i,i+1,i+2}>")
    plot!(is, map(ii -> abs(0.5*(get_pi(state,ii-1, ii)-get_pi(state,ii,ii+1)  )), is) , color = "blue"  , label = L"<\pi_{i,i+1}>")

    de = abs(minimum(energies)-maximum(energies))+0.1
    b = plot(xlabel = "", ylims = (eref-de, eref+de))
    hline!([eref], color = "red", linestyle = :dot, label = "")
    plot!(b, is, energies, color = "red", label = "")

    return a, b
end
function find_state(L, gx, gy,tol ;BC="trivial")
    s0, p0 = calc_order_parameters(gx, gy)
    init = init_state(L, s0, p0; BC = BC);
    error = Inf
    energy = Inf
    iter = 0
    while error > tol
        init, energy, error = update_state(init, gx, gy)
        iter += 1
        @show iter, error
    end
    return init, energy
end

gs, energy = find_state(100, 1., 0., 0.001; BC = "trivial")
order_plot, energy_plot = plot_state(gs, 1., 0.; BC = "trivial")
display(plot(order_plot, energy_plot))
