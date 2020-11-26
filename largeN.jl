include("main_full.jl")

function filestructure(dts,Ls)
    #initialize the subfolder structure
    existing = []
    for L in Ls
        try
            mkdir(DATAFOLDER*"/L=$(L)")
        catch
        end
        for dt in dts
            try
                mkdir(DATAFOLDER*"/L=$(L)"*"/dt=$(dt)")
            catch
            end
        end
    end
end
########################
# Physical parameters
########################
gx = 1.69
gy = 0.
mo = 0.005
########################

########################
# Simulation parameters
########################
dts = [1e-03]
Ls = [200]
########################
TODO = Dict(   (150,1e-3) => (10.0,10.)
              ,(150,5e-4) => (0.0,1.)
                )
DATAFOLDER = "DataPert"
filestructure(dts,Ls);

for L in Ls
    halfL = Int(L/2)
    for dt in dts
        topfolder = DATAFOLDER*"/L=$(L)/dt=$(dt)/75 site/"
        starttime, total_time = TODO[(L,dt)]
        N = Int(round(total_time/dt))
        if starttime == 0
            initial_state = initial_perturbation(Int.([range(round(Int,L/4), stop= round(Int,3*L/4), step= 1)...]),gx, L, mo);
            #initial_state = potential_trapping(gx, L, mo, 1.; tol = 1e-6);
            H    = make_H(initial_state, gx, gy);
            v    = eigen(H).vectors;
            energy = calc_energy(initial_state, gx, gy, mo)
            filename = topfolder*"MF__gx_$(gx)__gy_$(gy)__mo_$(mo)__t_0.0__dt_$(dt).jld2"
            saveMF(filename, initial_state, v, energy, starttime)
        end
        loadfile = topfolder*"MF__gx_$(gx)__gy_$(gy)__mo_$(mo)__t_$(starttime)__dt_$(dt).jld2"
        initial_state, v, energy = loadMF(loadfile)
        final_state = time_evolution(deepcopy(initial_state), v, dt, N, gx, gy, mo, topfolder; starttime=starttime);
    end
end
