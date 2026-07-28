# File: /Users/chase/QDT/characterization_control.jl
# Purpose: End-to-end characterization + control optimization loop for a single qudit.
# Notes:
# - This file runs an iterative loop: generate/optimize controls, run physical experiments (simulated),
#   infer parameter posteriors, update priors, and repeat until infidelity tolerance is met.
# - The script is written as a script (top-level); consider refactoring into functions for testability
#   and reusability (see improvement notes below).

include("src/QDT_src.jl")
Random.seed!(50)
# Random_seed_list = [10,20,30,40,50,60,70,80,90,100,110,120,130140,150,160,170,180,190,200]

# flags for saving and loading data, plotting, and for running the full characterization loop


#=
# -------------------------
# Basic model and timescale
# -------------------------
ω1 = 4.5
ω2 = 4.6
ωr = (ω1 + ω2)/2
ξ1   = 0.0
ξ2    = 0.0
ξ12   = 0.1 # Artificially large to allow fast coupling. Actual value: 1e-6 
J12    = 0.0 # 2*pi * 2.3E-3
Ne = 2
Ng = 0
N_tot = Ne + Ng
M_spam_order = 1E-4

n_iters_opt = 100
dt_opt = 0.2
ω1_min = ω1 - 0.2
ω1_max = ω1 + 0.2
ω2_min = ω2 - 0.2
ω2_max = ω2 + 0.2
ξmin = ξ12 - 0.1
ξmax = ξ12 + 0.1

# Control parametrization: B-splines
degree = 2
n_splines = 10
T = 50
nsteps = 100
dt = T/nsteps
Δ1 = ω1 - ωr
Δ2 = ω2 - ωr
carrier_freqs_1 = [Δ1, Δ1 - ξ12]
carrier_freqs_2 = [Δ2, Δ2 - ξ12]
carrier_freqs = [carrier_freqs_1, carrier_freqs_2]
n_freq_1 = length(carrier_freqs_1)
n_freq_2 = length(carrier_freqs_2)

#################################################################
# SETUP
#################################################################

# Create two qudits
q1 = DigitalQudit(Ne, Ng)
q2 = DigitalQudit(Ne, Ng)
# Set their parameter vlaues
add_param_samples(q1, [ω1], [ξ1])
add_param_samples(q2, [ω2], [ξ2])
# Set them to a shared rotating frame
# omega_rot = 0.5*(q1.omega_rot+q2.omega_rot)
q1.omega_rot = ωr
q2.omega_rot = ωr

# Put the qubits into a pair 
pair = DigitalQuditPair(q1, q2)
# Set their coupling values 
add_param_samples(pair, [ξ12], [J12]) 

# Create the controls for the CNOT gate
base_control = FortranBSplineControl(degree, n_splines, T)
q1_control = CarrierControl(
                base_control, 
                carrier_freqs_1
             )
q2_control = CarrierControl(
                base_control, 
                carrier_freqs_2
             )

q1_control_QDT = QuditControl(q1_control)
q2_control_QDT = QuditControl(q2_control)
# q1_control_QDT = QuditControl(base_control)
# q2_control_QDT = QuditControl(base_control)
push!(q1_control_QDT.coeffs, 0, 0.1*rand(2*n_splines*n_freq_1))
push!(q2_control_QDT.coeffs, 0, 0.1*rand(2*n_splines*n_freq_2))


add_control(pair, CNOT, q1_control_QDT, q2_control_QDT)

# Create a PhysicalQudit instance used for simulating real device outcomes
# Random.seed!(60)
phys_q_1 = PhysicalQudit(
                    Ne, Ng, 
                    ω1, ωr, ξ1,
                    M_spam_order=M_spam_order
                )

phys_q_2 = PhysicalQudit(
                    Ne, Ng, 
                    ω2, ωr, ξ2,
                    M_spam_order = M_spam_order
)

physical_pair = PhysicalQuditPair(phys_q_1, phys_q_2, ξ12, J12, M_spam_order = M_spam_order)

# Optimize the CNOT gate
# optimize_control(pair, CNOT, dt=dt_opt, 
#                 options=["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250])

# Testing the control
# dt = 0.25
# Psi= run_control(pair, 
#                   pair.controls[CNOT][1], 
#                   pair.controls[CNOT][2], 
#                   dt=dt)

# Psi = Psi[1,:,:]
Psi_physical, history_physical = run_control_physical(physical_pair, 
                  pair.controls[CNOT][1], 
                  pair.controls[CNOT][2], 
                  dt=dt)


# p_inf, s_inf, hist = measure_infidelity(physical_pair, 
#                 CNOT, pair.controls[CNOT][1], 
#                 pair.controls[CNOT][2], 100000, 
#                 add_SPAM = true, dt = dt)
 
 
# pcof_optimal_1 = get(pair.controls[CNOT][1].coeffs)
# pcof_optimal_2 = get(pair.controls[CNOT][2].coeffs)

# create maximum amount of data for using characterization
max_data = 1000
max_characterizations = 10
data_count = 1

nsteps_forward = Int(T/dt)


xi_perturbation = 0.0
omega_1_perturbation = 0.01
omega_2_perturbation = 0.0

# change it so initial controls are quadratic splines with carrier frequencies
# after obtaining data then use optimal control to opimize for data points 
max_control_parameter = 0.1
pcof_optimal_1 = (0.5 .- rand(q1_control.N_coeff)) .* max_control_parameter
pcof_optimal_2 = (0.5 .- rand(q2_control.N_coeff)) .* max_control_parameter

pcof_optimal_1 = rand() * max_control_parameter .* ones(q1_control.N_coeff) 
pcof_optimal_2 = rand() * max_control_parameter .* ones(q2_control.N_coeff)


degree = 2
n_splines = 10 
T = 50.0
pcof_optimal_1 = 0.1*[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
pcof_optimal_2 = 0.1*[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
Random.seed!(42)
pcof_optimal_1 = 0.1*rand(2*length(carrier_freqs_1)*n_splines)
pcof_optimal_2 = 0.1*rand(2*length(carrier_freqs_2)*n_splines)

event_obs, _, event_state = forward_event_quantum_multi([Ne, Ne], [Ng, Ng], [ω1, ω2], [ωr, ωr], [ξ1, ξ2], J12, ξ12, degree, n_splines, Matrix(1.0*I, 4, 4), T, 200, [pcof_optimal_1, pcof_optimal_2], carrier_freqs)
event_obs_2, _, event_state_2 = forward_event_quantum_multi([Ne, Ne], [Ng, Ng], [ω1 + omega_1_perturbation, ω2 + omega_2_perturbation], [ωr, ωr], [ξ1, ξ2], J12, ξ12 + xi_perturbation, degree, n_splines, Matrix(1.0*I, 4, 4), T, nsteps, [pcof_optimal_1, pcof_optimal_2], carrier_freqs)
Random.seed!(60)
M_spam = physical_pair.M_spam
M_spam = [0.999838 2.6558e-5 2.6558e-5 7.05444e-10;
 8.11831e-5 0.999892 2.15641e-9 2.65595e-5;
 8.11831e-5 2.15641e-9 0.999892 2.65595e-5;
 6.59177e-9 8.11875e-5 8.11875e-5 0.999947]
event_obs_SPAM = sample_quantum_state_history(100000, M_spam, event_obs)
event_obs_SPAM_2 = sample_quantum_state_history(100000, M_spam, event_obs_2)

println("Difference in last time step of event_obs_SPAM: ", norm(event_obs[:,end,:] - event_obs_2[:,end,:]))
println("Difference in last time step of event_state: ", norm(event_state[:,end,:] - event_state_2[:,end,:]))

U_target = zeros(4,4)
U_target[1,1] = 1.0
U_target[2,2] = 1.0
U_target[3,4] = 1.0
U_target[4,3] = 1.0

# measured infidelity 
p_inf = infidelity_population(event_obs_SPAM[:,end,:], abs2.(U_target))
println("Population Infidelity: ", p_inf)



experiment_name = "results/characterization_2qubits"
if save_data 
    folder, tag = make_unique_folder(experiment_name)
    pcof_dir = joinpath(folder, "pcof_optimal")
    mkpath(pcof_dir)
    @save joinpath(pcof_dir, "initial_pcof.jld2") pcof_optimal_1 pcof_optimal_2
    data_folder = joinpath(folder, "data")
    mkpath(data_folder)
    chain_data = joinpath(data_folder, "chain_data")
    mkpath(chain_data)
    # event_obs_folder = joinpath(folder, "event_obs")
    # mkpath(event_obs_folder)
    folder_histogram = joinpath(folder, "histogram")
    mkpath(folder_histogram)
    folder_wasserstein = joinpath(folder, "wasserstein")
    mkpath(folder_wasserstein)
end

iterations = 10000
burnin = 5000
thin = 10

run_characterization = true
calculate_landscape = false

if calculate_landscape 
    ω1_range = LinRange(ω1_min, ω1_max, 101)
    ω2_range = LinRange(ω2_min, ω2_max, 101)
    ξ12_range = LinRange(ξmin, ξmax, 101)
    U0 = Matrix(1.0*I, 4, 4)
    ω1_ω2_loss = zeros(length(ω1_range), length(ω2_range))
    ω1_ξ12_loss = zeros(length(ω1_range), length(ξ12_range))
    ω2_ξ12_loss = zeros(length(ω2_range), length(ξ12_range))
    for i in eachindex(ω1_range)
        for j in eachindex(ω2_range)
            if ((i-1)*length(ω1_range) + j) % 1000 == 0
                println("Evaluated for $((i-1)*length(ω1_range) + j) iterations")
            end
            θ = [ω1_range[i], ω2_range[j], ξ12]
            _, _, Φ, _ = true_posterior_multi([event_obs_SPAM], 
                                [
                                    Uniform(ω1_min, ω1_max),
                                    Uniform(ω2_min, ω2_max),
                                    Uniform(ξmin, ξmax)
                                ],
                                θ, 
                                [ωr, ωr], 
                                [2], [10], 
                                carrier_freqs, 
                                U0, [T], [nsteps], 
                                [[pcof_optimal_1, pcof_optimal_2]], 1, 10.0)
            ω1_ω2_loss[i,j] = mean(Φ)
        end
    end

    for i in eachindex(ω1_range)
        for j in eachindex(ξ12_range)
            
            if ((i-1)*length(ω1_range) + j) % 1000 == 0
                println("Evaluated for $((i-1)*length(ω1_range) + j) iterations")
            end
            θ = [ω1_range[i], ω2, ξ12_range[j]]
            _, _, Φ, _ = true_posterior_multi([event_obs_SPAM], 
                                [
                                    Uniform(ω1_min, ω1_max),
                                    Uniform(ω2_min, ω2_max),
                                    Uniform(ξmin, ξmax)
                                ],
                                θ, 
                                [ωr, ωr], 
                                [2], [10], 
                                carrier_freqs, 
                                U0, [T], [nsteps], 
                                [[pcof_optimal_1, pcof_optimal_2]], 1, 10.0)
            ω1_ξ12_loss[i,j] = mean(Φ)
        end
    end

    for i in eachindex(ω2_range)
        for j in eachindex(ξ12_range)
            if ((i-1)*length(ω2_range) + j) % 1000 == 0
                println("Evaluated for $((i-1)*length(ω1_range) + j) iterations")
            end
            θ = [ω1, ω2_range[i], ξ12_range[j]]
            _, _, Φ, _ = true_posterior_multi([event_obs_SPAM], 
                                [
                                    Uniform(ω1_min, ω1_max),
                                    Uniform(ω2_min, ω2_max),
                                    Uniform(ξmin, ξmax)
                                ],
                                θ, 
                                [ωr, ωr], 
                                [0], [1], 
                                carrier_freqs, 
                                U0, [T], [nsteps], 
                                [[pcof_optimal_1, pcof_optimal_2]], 1, 10.0)
            ω2_ξ12_loss[i,j] = mean(Φ)
        end
    end
    ω1_ω2_loss_heatmap = heatmap(ω1_range, ω2_range, log10.(ω1_ω2_loss), xlabel = "ω1", ylabel = "ω2")

    ω1_ξ12_loss_heatmap = heatmap(ω1_range, ξ12_range, log10.(ω1_ξ12_loss), xlabel = "ω1", ylabel = "ξ12")

    ω2_ξ12_loss_heatmap = heatmap(ω2_range, ξ12_range, log10.(ω2_ξ12_loss), xlabel = "ω2", ylabel = "ξ12")
end


function characterization_2qubits(n_rounds)
    # Replace the repeated blocks with a configurable iterative loop
    iterations = 5000
    burnin = 4000
    thin = 20

    run_characterization = true
    calculate_landscape = false

    # number of characterization rounds to run (was duplicated twice in original selection)

    if run_characterization
        # initial settings (used for round 1)
        ω_init_default = [4.4 4.7 0.2]
        ωr_vec = [ωr, ωr]
        degree_cfg = [2]
        n_splines_cfg = [10]
        U0 = Matrix(1.0*I, 4, 4)
        T_cfg = [50]
        nsteps_cfg = [200]
        # Random.seed!(42)
        # pcof_optimal_1 = 0.1*rand(length(carrier_freqs_1)*n_splines_cfg[1]*2)
        # pcof_optimal_2 = 0.1*rand(length(carrier_freqs_2)*n_splines_cfg[1]*2)
        pcof_optimal_total = [[pcof_optimal_1, pcof_optimal_2]]

        carrier_freqs = [carrier_freqs_1, carrier_freqs_2]
        total_data_count = 1
        ωmin = [4.3, 4.4, 0.0]
        ωmax = [4.7, 4.8, 0.3]
        event_obs_total = Vector{Array{Float64}}(undef, 1)
        event_obs_total[1] = event_obs_SPAM

        # downweight factor used when forming the downgraded covariance/prior
        downweight_power = 0.9

        # holder for last chain
        w2_chain_multi = nothing

        for round in 1:n_rounds
            println("Characterization round $round / $n_rounds")

            # choose initial guess / prior depending on round
            if round == 1
                θ_init = ω_init_default
                # run without providing an explicit prior
                w2_chain_multi = run_w2_chain_quantum_multi_adaptive(
                    event_obs_total;
                    θ = θ_init,
                    ωr = ωr_vec,
                    degree = degree_cfg,
                    n_splines = n_splines_cfg,
                    U0 = U0,
                    T = T_cfg,
                    λ_log0 = log(10.0),
                    nsteps = nsteps_cfg,
                    pcof_optimal_total = pcof_optimal_total,
                    carrier_freqs = carrier_freqs,
                    total_data_count = total_data_count,
                    ωmin = ωmin,
                    ωmax = ωmax,
                    iterations = iterations,
                    scale_factor = nsteps_cfg[1] + 1,
                    burnin = burnin,
                    thin = thin,
                    rng = Random.seed!(rand(1:100000))
                )
            else
                # form prior from previous diagnostic chain (downgraded MVN)
                covariance_mat = cov(w2_chain_multi.diagnostic_chain[:,1,:])
                covariance_downgrade = (1 / downweight_power) * covariance_mat
                mean_data = mean.(eachcol(w2_chain_multi.diagnostic_chain[:,1,:]))
                downgrade_prior = MvNormal(mean_data, covariance_downgrade)

                # use mean_data as starting theta
                θ_init = reshape(mean_data, 1, length(mean_data))

                # run with prior
                w2_chain_multi = run_w2_chain_quantum_multi_adaptive(
                    event_obs_total;
                    λ_log0 = log(10.0),
                    prior = downgrade_prior,
                    θ = θ_init,
                    ωr = ωr_vec,
                    degree = degree_cfg,
                    n_splines = n_splines_cfg,
                    U0 = U0,
                    T = T_cfg,
                    nsteps = nsteps_cfg,
                    pcof_optimal_total = pcof_optimal_total,
                    carrier_freqs = carrier_freqs,
                    total_data_count = total_data_count,
                    ωmin = ωmin,
                    ωmax = ωmax,
                    iterations = iterations,
                    scale_factor = nsteps_cfg[1] + 1,
                    burnin = burnin,
                    thin = thin, 
                    rng = Random.seed!(rand(1:100000))
                )
            end
            # save per-round data if requested
            

            # form downgraded MVN prior from diagnostic chain for next round (or final use)
            covariance_mat = cov(w2_chain_multi.diagnostic_chain[:,1,:])
            covariance_downgrade = (1 / downweight_power) * covariance_mat
            mean_data = mean.(eachcol(w2_chain_multi.diagnostic_chain[:,1,:]))
            downgrade_prior = MvNormal(mean_data, covariance_downgrade)

            # add samples to qudit objects and re-optimize controls (risk-neutral) using the inferred params
            add_param_samples(q1, w2_chain_multi.diagnostic_chain[:,1,1], zeros(length(w2_chain_multi.diagnostic_chain[:,1,1])), iter = round - 1)
            add_param_samples(q2, w2_chain_multi.diagnostic_chain[:,1,2], zeros(length(w2_chain_multi.diagnostic_chain[:,1,2])), iter = round - 1)
            add_param_samples(pair, w2_chain_multi.diagnostic_chain[:,1,3], zeros(length(w2_chain_multi.diagnostic_chain[:,1,3])), iter = round - 1)
            # add_param_samples(q1, [4.5], zeros(1), iter = round - 1)
            # add_param_samples(q2, [4.6], zeros(1), iter = round - 1)
            # add_param_samples(pair, [0.1], zeros(1), iter = round - 1)
            q1.omega_rot = ωr 
            q2.omega_rot = ωr
            # println("ωr q1: ", q1.omega_rot)
            # println("ωr q2: ", q2.omega_rot)
            if save_data
                @save joinpath(data_folder, "w2_chain_round$(round).jld2") w2_chain_multi mean_data covariance_mat
            end
            # optional expensive landscape calculation (unchanged)
            if calculate_landscape
                ω1_vec = LinRange(4.0, 5.0, 51)
                ω2_vec = LinRange(4.0, 5.0, 51)
                ξ12_vec = LinRange(0.0, 0.2, 51)
                loss = Array{Float64}(undef, length(ω1_vec), length(ω2_vec), length(ξ12_vec))
                global count = 0
                for (i, ω1) in enumerate(ω1_vec)
                    for (j, ω2) in enumerate(ω2_vec)
                        for (k, ξ12_val) in enumerate(ξ12_vec)
                            ω_pt = [ω1, ω2, ξ12_val]
                            loss[i, j, k] = true_posterior_multi(
                                event_obs_total,
                                [
                                    Uniform(ωmin[1], ωmax[1]),
                                    Uniform(ωmin[2], ωmax[2]),
                                    Uniform(ωmin[3], ωmax[3])
                                ],
                                ω_pt,
                                ωr_vec,
                                degree_cfg,
                                n_splines_cfg,
                                U0,
                                T_cfg,
                                nsteps_cfg,
                                pcof_optimal_total,
                                total_data_count
                            )[3]
                            global count += 1
                            if count % 1000 == 0
                                println("Calculated loss for $count points")
                            end
                        end
                    end
                end
                if save_data
                    @save joinpath(data_folder, "loss_data_round$(round).jld2") loss ω1_vec ω2_vec ξ12_vec
                end
            end

            # println("control 1 before")
            # display(last(pair.controls[CNOT][1].coeffs))
            if n_rounds > 1 
                optimize_control(pair, CNOT, dt = dt_opt,
                    options = ["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250],
                    iter = round - 1)
            end

            # get measured infidelity 
            # println("control 1 after")
            # display(last(pair.controls[CNOT][1].coeffs))
            inf_s, inf_p, pop_history, psi_final = measure_infidelity(physical_pair, CNOT, pair.controls[CNOT][1], pair.controls[CNOT][2], 100000, dt = dt)
            println("measured infidelity: ", inf_p)
            println("Last population: ")
            display(abs2.(psi_final))
            # println("last population 2: ")
            # _, state_hist = run_control_physical(physical_pair, pair.controls[CNOT][1], pair.controls[CNOT][2], dt = dt)
            # display(abs2.(state_hist[:,end,:]))
            # update pcof_optimal for possible reuse in later inference rounds
            pcof_optimal_1 = get(pair.controls[CNOT][1].coeffs)
            pcof_optimal_2 = get(pair.controls[CNOT][2].coeffs)

            if save_data 
                @save joinpath(pcof_dir, "pcof_optimal_$round.jld2") degree_cfg n_splines_cfg T_cfg pcof_optimal_1 pcof_optimal_2
            end

        end # for round

        # expose final chain and params
        param_chain = w2_chain_multi.diagnostic_chain
    else
        println("Skipping characterization (run_characterization = false).")
    end
    return w2_chain_multi
end

w2_chain_multi = characterization_2qubits(2)
=#





function characterization_control_2qubits(max_characterizations; initial_guess::Union{Nothing, AbstractVecOrMat} = nothing)

        # -------------------------
    # Basic model and timescale
    # -------------------------
    ω1 = 4.5 * 2pi
    ω2 = 4.6 * 2pi
    # ωr = (ω1 + ω2)/2
    ωr = 4.52 * 2pi
    ωr_vec = [ωr, ωr]

    ξ1  = 0.0
    ξ2  = 0.0
    ξ12 = 0.01 * 2pi # Artificially large to allow fast coupling. Actual value: 1e-6 
    J12 = 0.00 * 2pi # 2*pi * 2.3E-3
    true_params = [ω1, ω2, ωr, ξ12, J12]
    Ne = 2
    Ne_list = [2,2]
    Ng = 0
    Ng_list = [0,0]
    N_tot = Ne + Ng
    M_spam_order = 1E-4

    n_iters_opt = 100
    dt_opt = 0.2
    ω1_min = ω1 - ω1/20
    ω1_max = ω1 + ω1/20
    ω2_min = ω2 - ω2/20
    ω2_max = ω2 + ω2/20
    ξmin = ξ12 - ξ12/20
    ξmax = ξ12 + ξ12/20
    Jmin = J12 - J12/20
    Jmax = J12 + J12/20

    # Control parametrization: B-splines
    degree = 2
    degree_init = 2
    n_splines = 35
    n_splines_init = 35
    T = 200.0
    T_init = T/10
    nsteps = 300
    nsteps_init = Int(nsteps/10)
    dt = T/nsteps
    Δ1 = ω1 - ωr
    Δ2 = ω2 - ωr


    #################################################################
    # SETUP
    #################################################################

    # Create two qudits
    q1 = DigitalQudit(Ne, Ng)
    q2 = DigitalQudit(Ne, Ng)
    # Set their parameter vlaues
    add_param_samples(q1, [ω1], [ξ1])
    add_param_samples(q2, [ω2], [ξ2])
    # Set them to a shared rotating frame
    # omega_rot = 0.5*(q1.omega_rot+q2.omega_rot)
    q1.omega_rot = ωr
    q2.omega_rot = ωr

    # Put the qubits into a pair 
    pair = DigitalQuditPair(q1, q2)
    # Set their coupling values 
    add_param_samples(pair, [ξ12], [J12]) 

    SchroProb = get_schrodinger_problems(pair, T, dt)[1]
    H_drift = SchroProb.system_sym
    H_sym = SchroProb.sym_operators
    H_asym = SchroProb.asym_operators


    # carrier_freqs = [carrier_freqs_1, carrier_freqs_2]
    # om, _ = get_resonances(Ne = Ne_list, Ng = Ng_list, Hsys = H_drift, Hc_re = H_sym, Hc_im = H_asym, rotfreq = [ωr, ωr])
    # carrier_freqs_1 = om[1] .* 2pi
    # carrier_freqs_2 = om[2] .* 2pi
    carrier_freqs_1 = [Δ1, Δ1 - ξ12]
    carrier_freqs_2 = [Δ2, Δ2 - ξ12]
    carrier_freqs = [carrier_freqs_1, carrier_freqs_2]
    carrier_freqs = nothing
    n_freq_1 = length(carrier_freqs_1)
    n_freq_2 = length(carrier_freqs_2)

    # Create the controls for the CNOT gate
    control_q1 = FortranBSplineControl(degree, n_splines_init, T)
    control_q2 = FortranBSplineControl(degree, n_splines_init, T)

    if !isnothing(carrier_freqs)

        control_q1 = CarrierControl(
                        control_q1, 
                        carrier_freqs_1
                    )
        control_q2 = CarrierControl(
                        control_q2, 
                        carrier_freqs_2
                    )
    end

    max_control_amplitude = 0.2

    # include a single product gate
    IX = ProductGate(IdentityGate, PauliX)

    gates = [CNOT, SWAP, IX, CZ]
    N_gates = length(gates)

    q1_control_CNOT = QuditControl(control_q1)
    q2_control_CNOT = QuditControl(control_q2)
    # q1_control_SWAP = QuditControl(control_q1)
    # q2_control_SWAP = QuditControl(control_q2)
    # q1_control_QDT = QuditControl(base_control)
    # q2_control_QDT = QuditControl(base_control)
    Random.seed!(42)
    push!(q1_control_CNOT.coeffs, 0, max_control_amplitude*(0.5 .- rand(control_q1.N_coeff)))
    push!(q2_control_CNOT.coeffs, 0, max_control_amplitude*(0.5 .- rand(control_q2.N_coeff)))

    # push!(q1_control_SWAP.coeffs, 0, max_control_amplitude*(0.5 .- rand(control_q1.N_coeff)))
    # push!(q2_control_SWAP.coeffs, 0, max_control_amplitude*(0.5 .- rand(control_q1.N_coeff)))

    q1_control_SWAP = deepcopy(q1_control_CNOT)
    q2_control_SWAP = deepcopy(q2_control_CNOT)

    q1_control_CZ = deepcopy(q1_control_CNOT)
    q2_control_CZ = deepcopy(q2_control_CNOT)

    q1_control_IX = deepcopy(q1_control_CNOT)
    q2_control_IX = deepcopy(q2_control_CNOT)

    iter_count = 1
    add_control(pair, CNOT, q1_control_CNOT, q2_control_CNOT)
    add_control(pair, SWAP, q1_control_SWAP, q2_control_SWAP)
    add_control(pair, CZ, q1_control_CZ, q2_control_CZ)
    add_control(pair, IX, q1_control_IX, q2_control_IX)

    # Create a PhysicalQudit instance used for simulating real device outcomes
    # Random.seed!(60)
    phys_q_1 = PhysicalQudit(
                        Ne, Ng, 
                        ω1, ωr, ξ1,
                        M_spam_order=M_spam_order
                    )

    phys_q_2 = PhysicalQudit(
                        Ne, Ng, 
                        ω2, ωr, ξ2,
                        M_spam_order = M_spam_order
    )

    physical_pair = PhysicalQuditPair(phys_q_1, phys_q_2, ξ12, J12, M_spam_order = M_spam_order)

    # initial data generation
    pcof_optimal_1 = get(q1_control_CNOT.coeffs)
    pcof_optimal_2 = get(q2_control_CNOT.coeffs)
    event_obs, _, event_state = forward_event_quantum_multi([Ne, Ne], [Ng, Ng], [ω1, ω2], [ωr, ωr], [ξ1, ξ2], J12, ξ12, degree, n_splines_init, Matrix(1.0*I, 4, 4), T_init, nsteps_init, [pcof_optimal_1, pcof_optimal_2], carrier_freqs)
    # physical_pair.M_spam = Matrix(1.0*I, 4, 4)
    M_spam = physical_pair.M_spam
    # M_spam = Matrix(1.0*I, 4, 4)

    event_obs_SPAM = sample_quantum_state_history(100000, M_spam, event_obs)
    # event_obs_SPAM = event_obs

    save_data = true
    load_data = false
    plot_results = true
    run_loop = true
    use_initial_samples = false

    experiment_name = "results/characterization_2qubits"
    if save_data 
        folder, tag = make_unique_folder(experiment_name)
        pcof_dir = joinpath(folder, "pcof_optimal")
        mkpath(pcof_dir)
        @save joinpath(pcof_dir, "initial_pcof.jld2") pcof_optimal_1 pcof_optimal_2
        data_folder = joinpath(folder, "data")
        mkpath(data_folder)
        chain_data = joinpath(data_folder, "chain_data")
        mkpath(chain_data)
        event_obs_folder = joinpath(data_folder, "event_obs")
        mkpath(event_obs_folder)
        # event_obs_folder = joinpath(folder, "event_obs")
        # mkpath(event_obs_folder)
        folder_histogram = joinpath(folder, "histogram")
        mkpath(folder_histogram)
        folder_wasserstein = joinpath(folder, "wasserstein")
        mkpath(folder_wasserstein)
    end

    epsilon = 1E-4

    θ_init = [4.4, 4.7, 0.0] .* 2pi
    ωmin = [ω1_min, ω2_min, ξmin]
    ωmax = [ω1_max, ω2_max, ξmax]

    initial_dataset_size = 1
    event_obs_init = Vector{Array{Float64}}(undef, initial_dataset_size)
    event_obs_init[1] = event_obs_SPAM

    event_obs_total = Vector{Array{Float64}}(undef, max_characterizations)
    save_object(joinpath(event_obs_folder, "event_obs_0.jld2"), event_obs_init)

    nsteps_total = Vector{Real}(undef, initial_dataset_size)
    T_total = Vector{Real}(undef, initial_dataset_size)
    n_splines_total = Vector{Real}(undef, initial_dataset_size)
    degree_total = Vector{Real}(undef, initial_dataset_size)



    q_pred_infidelity_total = OrderedDict{Int, Dict{Union{GateType,ProductGate}, Float64}}()
    q_p_meas_infidelity_total = OrderedDict{Int, Dict{Union{GateType,ProductGate}, Float64}}()
    control_dict_total = OrderedDict{Int, Dict{Union{GateType,ProductGate}, Vector{QuditControl}}}()
    pcof_initial_total = Vector{Vector{Vector{Float64}}}(undef, initial_dataset_size)
    pcof_initial_total[1] = [pcof_optimal_1, pcof_optimal_2]

    covariance_list = [] 
    mean_list = []
    rand_seed_list = []
    param_init_list = []


    q_pred_infidelity_total[0] =
        Dict{Union{GateType,ProductGate}, Float64}(g => 1.0 for g in gates)

    q_p_meas_infidelity_total[0] =
        Dict{Union{GateType,ProductGate}, Float64}(g => 1.0 for g in gates)

    n_splines_total = fill(n_splines, N_gates)
    T_total = fill(T, N_gates)
    nsteps_total = fill(nsteps, N_gates)
    degree_total = fill(degree, N_gates)
    total_data_count = N_gates

    control_params_init = [degree_init, n_splines_init, T_init, nsteps_init]
    control_params_total = [degree_total, n_splines_total, T_total, nsteps_total]


    downweight_power = 0.7
    iter_count = 1

    iterations = 9000
    burnin = 8000
    thin = 10
    t0_adapt = 1000

    MHG_params = [iterations, burnin, thin, t0_adapt, ωmin, ωmax]

    U0 = Matrix(1.0*I, 4, 4)

    if isnothing(initial_guess)
        ω1_init = rand(Uniform(ω1_min, ω1_max))
        ω2_init = rand(Uniform(ω2_min, ω2_max))
        # J_init = rand(Uniform(Jmin, Jmax))
        ξ_init = rand(Uniform(ξmin, ξmax))
        θ_init = [ω1_init ω2_init ξ_init]
    else
        θ_init = initial_guess
    end
    # θ_init = [4.49 4.61 0.055] .* 2pi
    # θ_init = [4.5 4.6 0.01] .* 2pi
    event_obs = Vector{Array{Float64}}(undef, N_gates)
    pcof_optimal_total = Vector{Vector{Vector{Float64}}}(undef, N_gates)
    downgrade_prior = nothing 
    covariance_downgrade = nothing
    iter_count = 1
    for i in 1:max_characterizations
        push!(param_init_list, θ_init)
        # perform characterization and then optimal control 
        control_dict = OrderedDict{Union{GateType,ProductGate}, Vector{QuditControl}}()
        q_pred_infidelity = OrderedDict{Union{GateType,ProductGate}, Float64}()
        q_p_meas_infidelity = OrderedDict{Union{GateType,ProductGate}, Float64}()
        rand_seed = rand(RandomDevice(), UInt64)
        push!(rand_seed_list, rand_seed)
        if i == 1
            w2_chain_multi = run_w2_chain_quantum_multi_adaptive(
                    event_obs_init;
                    θ = θ_init,
                    ωr = ωr_vec,
                    degree = [degree],
                    n_splines = [n_splines_init],
                    U0 = U0,
                    T = [T_init],
                    λ_log0 = log(10.0),
                    nsteps = [nsteps_init],
                    pcof_optimal_total = pcof_initial_total,
                    carrier_freqs = carrier_freqs,
                    total_data_count = initial_dataset_size,
                    ωmin = ωmin,
                    ωmax = ωmax,
                    iterations = iterations,
                    scale_factor = nsteps_init + 1,
                    burnin = burnin,
                    thin = thin,
                    rng = Random.seed!(rand_seed),
                    t0_adapt = t0_adapt, 
                    verbose = false
                    )
        else
            # println("pcof_optimal total")
            # println(pcof_optimal_total)
            w2_chain_multi = run_w2_chain_quantum_multi_adaptive(
                    event_obs;
                    θ = θ_init,
                    ωr = ωr_vec,
                    degree = degree_total,
                    n_splines = n_splines_total,
                    prior = downgrade_prior,
                    U0 = U0,
                    T = T_total,
                    λ_log0 = log(10.0),
                    nsteps = nsteps_total,
                    pcof_optimal_total = pcof_optimal_total,
                    carrier_freqs = carrier_freqs,
                    total_data_count = N_gates,
                    ωmin = ωmin,
                    ωmax = ωmax,
                    iterations = iterations,
                    scale_factor = nsteps_total[1] + 1,
                    burnin = burnin,
                    thin = thin,
                    initial_cov_p = covariance_downgrade, 
                    rng = Random.seed!(rand_seed),
                    t0_adapt = t0_adapt
                    )
        end
        
        display(w2_chain_multi.diagnostic_chain[:,1,:]./2pi)

        

        # single qubit parameters
        add_param_samples(q1, w2_chain_multi.diagnostic_chain[:,1,1], zeros(length(w2_chain_multi.diagnostic_chain[:,1,1])), iter = iter_count)
        add_param_samples(q2,  w2_chain_multi.diagnostic_chain[:,1,2], zeros(length(w2_chain_multi.diagnostic_chain[:,1,2])), iter = iter_count)
        
        # coupling parameters 
        # use this for cross-kerr coupling
        add_param_samples(pair, w2_chain_multi.diagnostic_chain[:,1,3], zeros(length(w2_chain_multi.diagnostic_chain[:,1,3])), iter = iter_count)
        # use this for dipole coupling
        # add_param_samples(pair, zeros(length(w2_chain_multi.diagnostic_chain[:,1,1])), w2_chain_multi.diagnostic_chain[:,1,1],iter = iter_count)


        # add_param_samples(q1, [4.5 * 2pi], zeros(1), iter = iter_count)
        # add_param_samples(q2, [4.6 * 2pi], zeros(1), iter = iter_count)
        # add_param_samples(pair, zeros(1), [0.005*2pi], iter = iter_count)
        q1.omega_rot = ωr 
        q2.omega_rot = ωr

        
        
        for j in 1:N_gates

            
            pred_infidel = predicted_infidelity(pair, gates[j], pair.controls[gates[j]][1], pair.controls[gates[j]][2], dt = dt, iter = iter_count)
            if pred_infidel > epsilon || (q_p_meas_infidelity_total[iter_count - 1][gates[j]]) > epsilon 
                println("re-optimizing,  measured infidelity: ", q_p_meas_infidelity_total[iter_count-1][gates[j]])
                println("re-optimizing,  predicted infidelity: ", pred_infidel)
                pcof_optimal = optimize_control(pair, gates[j], dt = dt,
                        options = ["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250],
                        iter = iter_count, max_amplitude = max_control_amplitude)
                # println("optimized coeffs for $(gates[j]) gate")
                # println(pcof_optimal.x)
                
            end

            inf_s, inf_p, pop_history, psi_final = measure_infidelity(physical_pair, gates[j], pair.controls[gates[j]][1], pair.controls[gates[j]][2], 100000, dt = dt)
            event_obs[j] = pop_history
            q_p_meas_infidelity[gates[j]] = clamp(inf_p, 1E-6, 1) 
            control_dict[gates[j]] = pair.controls[gates[j]]
            pred_inf = predicted_infidelity(pair, gates[j], pair.controls[gates[j]][1], pair.controls[gates[j]][2], dt = dt, iter = iter_count)
            q_pred_infidelity[gates[j]] = pred_inf
            println("Predicted Infidelity: ", pred_inf)
            println("Measured Infidelity: ", clamp(inf_p, 1E-6, 1))
            pcof_optimal_total[j] = [get(pair.controls[gates[j]][1].coeffs), get(pair.controls[gates[j]][2].coeffs)]
            # println("pcof gate $(gates[j]) after: ")
            # println(pcof_optimal_total[j])
            save_object(joinpath(pcof_dir, "pcof_optimal_$i.jld2"), pcof_optimal_total)
        end
        control_dict_total[iter_count] = control_dict 
        q_pred_infidelity_total[iter_count] = q_pred_infidelity 
        q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity

        if save_data 
            save_object(joinpath(chain_data, "chain_data_$i.jld2"), w2_chain_multi)
            save_object(joinpath(event_obs_folder, "event_obs_$i.jld2"), event_obs)
        end


        # downgrade_prior = nothing
        # θ_init = [4.5 4.6 0.005] .* 2pi
        # downgrade_prior = nothing
        covariance_mat = cov(w2_chain_multi.diagnostic_chain[:,1,:])
        covariance_downgrade = (1 / downweight_power) * covariance_mat
        mean_data = mean.(eachcol(w2_chain_multi.diagnostic_chain[:,1,:]))
        if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
            println("Loop terminated, measured infidelity and predicted infidelity small")
            break
            
            # break
        else
            if i == max_characterizations
                println("Max # of characterizations reached, loop terminated")
                break 
            end
            println("Re-run characterization")
            prior_tol = 3E-4
            if all(values(q_p_meas_infidelity) .< prior_tol)
                downgrade_prior = MvNormal(mean_data, covariance_downgrade)
                θ_init = reshape(mean_data, 1, length(mean_data))
            else
                println("Measured infidelity of a single gate was larger than $prior_tol, setting prior to be uniform to explore more parameter space.")
                downgrade_prior = nothing
                # ω1_init = rand(Uniform(ω1_min, ω1_max))
                # ω2_init = rand(Uniform(ω2_min, ω2_max))
                # J_init = rand(Uniform(Jmin, Jmax))
                # ξ_init = rand(Uniform(ξmin, ξmax))
                # θ_init = [ω1_init ω2_init ξ_init]
                θ_init = w2_chain_multi.ω_best
                # θ_init = [4.4 4.7 0.015]
                # θ_init = [4.45 4.65 0.008]
                # θ_init = [4.49 4.61 0.055] .* 2pi
                
            end
            push!(mean_list, mean_data)
            push!(covariance_list, covariance_mat)
        end


        iter_count += 1

    end

    if save_data 
        @save joinpath(data_folder, "infidelity_data.jld2") q_pred_infidelity_total q_p_meas_infidelity_total mean_list covariance_list rand_seed_list
        @save joinpath(data_folder, "characterization_params.jld2") true_params param_init_list MHG_params mean_list covariance_list downweight_power rand_seed_list 
        @save joinpath(data_folder, "control_params.jld2") control_dict_total gates control_params_init control_params_total carrier_freqs
    end
    return iter_count
end

iter_count = 2
max_runs = 20
initial_guess =  [27.351048846350423 29.101220514620582 0.0599950992431776]
initial_guess = 
for i in 1:max_runs
    println("--------------------------------------")
    println("-------------- Run #$i ---------------")
    println("--------------------------------------")
    if i % 2 == 0
        iter_count = characterization_control_2qubits(6, initial_guess = initial_guess)
    else
        iter_count = characterization_control_2qubits(6)
    end
end
#=
if run_characterization

    ω_init = [4.4 4.7 0.2]
    ωr_vec = [ωr,  ωr]
    degree = [2]
    n_splines = [10]
    U0 = Matrix(1.0*I, 4, 4)
    T = [50]
    nsteps = [400]
    pcof_optimal_total = [[pcof_optimal_1, pcof_optimal_2]]
    carrier_freqs = [0, ξ12, 2*ξ12]
    total_data_count = 1
    ωmin = [4.0,4.0,0.0]
    ωmax = [5.0,5.0,0.2]
    event_obs_total = Vector{Array{Float64}}(undef, 1)
    event_obs_total[1] = event_obs_SPAM
    w2_chain_multi = run_w2_chain_quantum_multi_adaptive(
        event_obs_total;
        θ = ω_init, 
        ωr = ωr_vec,
        degree = degree,
        n_splines = n_splines,
        U0 = U0,
        T = T,
        λ_log0 = log(10.0),
        nsteps = nsteps,
        pcof_optimal_total = pcof_optimal_total,
        carrier_freqs = carrier_freqs,
        total_data_count = total_data_count,
        ωmin = ωmin, 
        ωmax = ωmax, 
        iterations = iterations,
        scale_factor = nsteps[1] + 1,
        burnin = burnin,
        thin = thin
    )

    # now calculate the wasserstein distance from 100 equispaced points in the parameter space to the true parameters, and see how it changes as we add more data and re-run inference.
    if calculate_landscape
        ω1_vec = LinRange(4.0, 5.0, 101)
        ω2_vec = LinRange(4.0, 5.0, 101)
        ξ12_vec = LinRange(0.0, 0.2, 101)


        loss = Array{Float64}(undef, 101, 101, 101)

        global count = 0

        for (i, ω1) in enumerate(ω1_vec)
            for (j, ω2) in enumerate(ω2_vec)
                for (k, ξ12) in enumerate(ξ12_vec)

                    ω_pt = [ω1, ω2, ξ12]

                    loss[i,j,k] = true_posterior_multi(
                        event_obs_total,
                            [
                        Uniform(ωmin[1], ωmax[1]),
                        Uniform(ωmin[2], ωmax[2]),
                        Uniform(ωmin[3], ωmax[3])
                        ],
                        ω_pt,
                        ωr_vec,
                        degree,
                        n_splines,
                        U0,
                        T,
                        nsteps,
                        pcof_optimal_total,
                        total_data_count
                    )[3]

                    global count += 1
                    if count % 1000 == 0
                        println("Calculated loss for $count points")
                    end
                end
            end
        end

        all_pts = collect(Iterators.product(ω1_vec, ω2_vec, ξ12_vec))
        loss_vec = zeros(length(all_pts))
        for i in 1:length(all_pts)
            ω_pt = [all_pts[i]...]
            loss_vec[i] = true_posterior_multi(event_obs_total, [Uniform(ωmin[1], ωmax[1]), Uniform(ωmin[2], ωmax[2]), Uniform(ωmin[3], ωmax[3])], ω_pt, ωr_vec, degree, n_splines, U0, T, nsteps, pcof_optimal_total, total_data_count)[3]
        end
        # now plot three slices of the loss landscape, ω1 and \omega2 at the true ξ12 value, ξ12 and ω1 at the true ω2 value, and ξ12 and ω2 at the true ω1 value

        # loss_plot_ω1_ω2 = heatmap(ω1_vec, ω2_vec, log10.(loss[:,:,26]'), xlabel = "ω1", ylabel = "ω2", title = "Loss landscape: ω1 vs ω2 at true ξ12")
        # loss_plot_ω1_ξ12 = heatmap(ω1_vec, ξ12_vec, log10.(loss[:,41,:]'), xlabel = "ω1", ylabel = "ξ12", title = "Loss landscape: ω1 vs ξ12 at true ω2")
        # loss_plot_ω2_ξ12 = surface(ω2_vec, ξ12_vec, log10.(loss[26,:,:]'), xlabel = "ω2", ylabel = "ξ12", title = "Loss landscape: ω2 vs ξ12 at true ω1")
    end
    if save_data 
        @save joinpath(data_folder, "data.jld2") pair physical_pair w2_chain_multi event_obs_total pcof_optimal_total
        @save joinpath(folder_wasserstein, "loss_data.jld2") loss ω1_vec ω2_vec ξ12_vec
    end
end

param_chain_1 = w2_chain_multi.diagnostic_chain


covariance_mat = cov(w2_chain_multi.diagnostic_chain[:,1,:])

# downgrade multivariate normal by multiplying covariance by (1/p)
downweight_power = 0.9
covariance_downgrade = (1/downweight_power)*covariance_mat 

mean_data = mean.(eachcol(w2_chain_multi.diagnostic_chain[:,1,:]))

downgrade_prior = MVNormal(mean_data, covariance_downgrade)

# the parameters have been obtained, now to run risk-neutral control with the parameters 
add_param_samples(q1, w2_chain_multi.diagnostic_chain[:,1,1], zeros(length(w2_chain_multi.diagnostic_chain[:,1,1])))
add_param_samples(q2, w2_chain_multi.diagnostic_chain[:,1,2], zeros(length(w2_chain_multi.diagnostic_chain[:,1,2])))
add_param_samples(pair, zeros(length(w2_chain_multi.diagnostic_chain[:,1,3])), w2_chain_multi.diagnostic_chain[:,1,3]) 

optimize_control(pair, CNOT, dt=dt_opt, 
                options=["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250])

pcof_optimal_1 = get(pair.controls[CNOT][1].coeffs)
pcof_optimal_2 = get(pair.controls[CNOT][2].coeffs)
if run_characterization

    ω_init = mean_data
    ωr_vec = [ωr,  ωr]
    degree = [2]
    n_splines = [10]
    U0 = Matrix(1.0*I, 4, 4)
    T = [50]
    nsteps = [400]
    pcof_optimal_total = [[pcof_optimal_1, pcof_optimal_2]]
    carrier_freqs = [0, ξ12, 2*ξ12]
    total_data_count = 1
    ωmin = [4.0,4.0,0.0]
    ωmax = [5.0,5.0,0.2]
    event_obs_total = Vector{Array{Float64}}(undef, 1)
    event_obs_total[1] = event_obs_SPAM
    w2_chain_multi = run_w2_chain_quantum_multi(
        event_obs_total;
        λ_log0 = log(10.0),
        prior = downgrade_prior,
        θ = ω_init, 
        ωr = ωr_vec,
        degree = degree,
        n_splines = n_splines,
        U0 = U0,
        T = T,
        nsteps = nsteps,
        pcof_optimal_total = pcof_optimal_total,
        carrier_freqs = carrier_freqs,
        total_data_count = total_data_count,
        ωmin = ωmin, 
        ωmax = ωmax, 
        iterations = iterations,
        scale_factor = nsteps[1] + 1,
        burnin = burnin,
        thin = thin
    )
end

param_chain_2 = w2_chain_multi.diagnostic_chain

=#


# The following preallocations assume we will fill up to max_data entries.
# Improvement: consider using Vector{Union{Nothing, T}}(...) or push! semantics instead of large undef arrays.

#=

pcof_init = Vector{}
event_obs_total = Vector{Array{Float64}}(undef, 1)
nsteps_total = Vector{Real}(undef, max_data)
T_total = Vector{Real}(undef, max_data)
n_splines_total = Vector{Real}(undef, max_data)
degree_total = Vector{Real}(undef, max_data)

# Need to create the qudit and the controls 
q = DigitalQudit(Ne, Ng)
q.omega_rot = ωr
iter_count = 0

# kernel density estimator parameters - these control the shape of the kernel density estimation that is used to create the prior for the next iteration of inference.

# determine downweighting parameters and kernel type for creating the kernel density estimation. 
# the two options are "Normal" and "Beta" kernels
downweight_power = 0.2
bandwidth = 0.1
dist_func = "GaussianFit"

# characterization iteration parameters 
iterations = 2000
burnin = 1000
thin = 2
total_samples = Int((iterations - burnin)/2)

degree_init = 0

experiment_name = "results/characterization_control_loop_data_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power)_samples_$(total_samples)_degree_$(degree_init)"



if save_data || plot_results
    folder, tag = make_unique_folder(experiment_name)
end 


ω0 = 4.38
ωmin = 4.0
ωmax = 5.0
xs = LinRange(ωmin, ωmax, 1000)

# set color palette for plotting - this is used for consistency across all plots, and to ensure that the same colors are used for the same gates/metrics across iterations.
colors = palette(:default)
colors_palette = [colors[1], colors[2], colors[4], colors[5], colors[6]]


if run_loop
    # println("Running characterization and control optimization loop with kernel: $dist_func, bandwidth: $bandwidth, downweight_power: $downweight_power")
    println("Running characterization and control optimization loop with downweighted Gaussian fit, downweight_power: $downweight_power")

    # Dictionaries indexed by iteration -> per-gate dictionaries
    control_dict_total = OrderedDict{Int, Dict{GateType, QuditControl}}()
    q_pred_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
    q_p_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
    q_s_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()

    # Create 1 pulse initially then use that for first characterization iteration
    control_dict = OrderedDict{GateType, QuditControl}()
    q_pred_infidelity = OrderedDict{GateType, Float64}()
    q_p_meas_infidelity = OrderedDict{GateType, Float64}()
    q_s_meas_infidelity = OrderedDict{GateType, Float64}()
    
    println("Creating controls")
    for j = 1:N_gates
        # Control for this gate with constant initial pulses
        qcontrol = FortranBSplineControl(degree_init, n_splines, T_gate)
        # pcof0 = 0.5*max_control_parameter*ones(qcontrol.N_coeff)
        # pcof0 = max_control_parameter*rand(qcontrol.N_coeff)
        pcof0[1:n_splines] .= rand()*max_control_parameter*ones(n_splines)
        pcof0[n_splines+1:end] .= rand()*max_control_parameter*ones(n_splines) 
        add_control(q, gates[j], qcontrol; coeffs = copy(pcof0))
        if plot_results
            control_plot = plot_controls(qcontrol, pcof0)
            savefig(control_plot, joinpath(folder, "initial_control.png"))
        end
        control_dict[gates[j]] = q.controls[gates[j]]
        # Infidelities for this gate
        q.infidelity[gates[j]] = History(Float64)
        # measure_infidelity returns (state-based infidelity, process-based infidelity, pop_history)
        q_s_inf, q_p_inf, pop_history = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
        # measure predicted infidelity under a slightly different physical model (phys_q_init)
        _, q_pred_init, _ = measure_infidelity(phys_q_init, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt; add_SPAM = false)
        q_p_meas_infidelity[gates[j]] = q_p_inf 
        q_s_meas_infidelity[gates[j]] = q_s_inf 
        q_pred_infidelity[gates[j]] = q_pred_init
    end
    control_dict_total[iter_count + 1] = control_dict
    q_pred_infidelity_total[iter_count] = q_pred_infidelity
    q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
    q_s_meas_infidelity_total[iter_count] = q_s_meas_infidelity

    # create physical qubit with initial parameter guess of 4.45 while keeping rotational frequency at 4.5 to get initial predicted infidelities 

    # Run control pulse on physical device (simulated)
    _, event_obs = run_control_physical(phys_q, control_dict_total[data_count][gates[1]]; dt = dt)
    event_obs = abs2.(event_obs)

    # Add measurement noise via SPAM matrix and sampling of quantum state history.
    # M_spam = column_stochastic(1E-4*rand(2))
    event_obs = sample_quantum_state_history(100000, phys_q.M_spam, event_obs)

    # Record dataset metadata for index 1
    nsteps_total[1] = 50
    T_total[1] = dt*nsteps_total[1]
    degree_total[1] = degree_init
    n_splines_total[1] = n_splines
    event_obs_total[data_count] = event_obs[:,1:nsteps_total[1]+1, :] 
    pcof_optimal_total[data_count] = get(control_dict_total[data_count][PauliX].coeffs)
    if save_data 
        save_object(joinpath(folder, "event_obs_iteration_$(iter_count).jld2"), event_obs_total)
    end
    # Initialize prior vector storage: storing discretized pdf values on xs grid per iteration
    prior_vec = Vector{Vector{Float64}}(undef, max_characterizations + 1) # +1 to account for initial prior before loop
    init_guess_vec = Vector{Float64}(undef, max_characterizations + 1)

    d = Uniform(ωmin, ωmax)
    init_guess_vec[1] = ω0
    ys = pdf.(d, xs)
    prior_vec[iter_count + 1] = ys

    
    if use_initial_samples
        omega_samples = load("omega_samples.jld2")
        xi_samples = zeros(length(omega_samples))
        add_param_samples(q, omega_samples, xi_samples; iter = iter_count + 1, average_omega_rot = false)
    else
        # Run an initial W2-chain inference over the first dataset (single run)
        w2_chain = run_w2_chain_quantum(
            event_obs_total;
            ω0= ω0, 
            ωr= 4.5, 
            degree = degree_total, 
            n_splines = n_splines_total, 
            U0 = U0, 
            T = T_total, 
            nsteps = nsteps_total, 
            pcof_optimal_total = pcof_optimal_total,
            total_data_count = data_count,
            λ= 10.0,
            iterations=iterations,
            burnin=burnin,
            thin=thin,
            ωmin=ωmin,
            ωmax=ωmax,
            δ=2.0,
            scale_factor=length(event_obs_total[1,:,1]),
            risk_scale=1.0,
            t0_adapt=100,
            target_accept=0.44,
            rng=Random.default_rng(),
        )

        # Extract samples for ω and create zero xi samples vector (only ω inferred here)
        omega_samples = w2_chain.chain_post[:,1]
        xi_samples = zeros(length(omega_samples))
        add_param_samples(q, omega_samples, xi_samples; iter = iter_count + 1, average_omega_rot = false)
    end
    

    # Add parameter samples to the DigitalQudit object for iteration 1
    

    # -----------------------------
    # Main characterization loop
    # -----------------------------
    # Risk-neutral optimization then characterization, repeated up to max_characterizations.



    for i in 1:max_characterizations
        global iter_count += 1
        # set ω0 to be the average of all previous samples
        omega_samples = get(q.omega, iter_count)
        
        # create kernel estimation that will be downweighted and used as prior for the next inference (NOT USED)
        # kernel_output = kernel_downweighting(omega_samples, bandwidth, downweight_power, kernel_func = dist_func, sample_min = ωmin, sample_max = ωmax)
        # ω0 = expected_value_piecewise_gauss(kernel_output.f_pdf, kernel_output.x_grid)
        # prior_vec[iter_count + 1] = kernel_output.f_pdf(xs)
        # println("PDF Area: ", gauss_integral(kernel_output.f_pdf, xs))
        # println("ω₀ = $ω0")


        # do a downgraded gaussian fit for the next prior
        posterior = gaussian_fit(omega_samples)
        prior_downgrade = downgrade_gaussian(omega_samples, 1/downweight_power)
        prior_vec[i + 1] = pdf.(truncated_downgrade_gaussian(omega_samples, 1/downweight_power, ωmin, ωmax), xs)
        # set ω0 to be the mean of the downgraded (both downgrading and truncating don't affect this)
        ω0 = mean(prior_downgrade)
        init_guess_vec[iter_count + 1] = ω0
        println("ω₀ = $ω0")
        current_samples_plot = histogram(omega_samples, xlabel = "ω", ylabel = "Density", title = "Posterior Samples, Iteration $iter_count", dpi = 250, label = "Samples", alpha = 0.5, normalize = true)
        plot!(xs, prior_vec[i + 1], label = "Truncated Prior", lw = 2, alpha = 0.7)
        plot!(xs, pdf.(posterior, xs), label = "Posterior", lw = 2, alpha = 0.7, linestyle =:dash)
        plot!(xs, pdf.(prior_downgrade, xs), label = "Downweighted Posterior", lw = 2, alpha = 0.7, linestyle =:dash)
        if plot_results
            savefig(joinpath(folder, "histogram_iteration_$(iter_count).png"))
        end
        

        control_dict = OrderedDict{GateType, QuditControl}()
        q_pred_infidelity = OrderedDict{GateType, Float64}()
        q_p_meas_infidelity = OrderedDict{GateType, Float64}()
        q_s_meas_infidelity = OrderedDict{GateType, Float64}()
        # WARNING/NOTE: These reassignments shadow the earlier pcof_optimal_total/event_obs_total variables
        # that were sized for max_data. Here they are recreated with length N_gates. This is legal, but may be
        # a source of confusion / bugs if you expected a consistent shape across code.
        pcof_optimal_total = Vector{Vector{Float64}}(undef, N_gates)
        event_obs_total = Vector{Array{Float64}}(undef, N_gates)
        nsteps_total = Vector{Real}(undef, N_gates)
        T_total = Vector{Real}(undef, N_gates)
        n_splines_total = Vector{Real}(undef, N_gates)
        degree_total = Vector{Real}(undef, N_gates)

        println("Performing risk-neutral optimization and measuring infidelity.")
        for j = 1:N_gates
            # if degree of controls is changing from initial data then need to update controls
            if get(q.controls[gates[j]].objs).degree != degree
                println("Degree of control ($(get(q.controls[gates[j]].objs).degree)) for $(gates[j]) does not match desired degree ($degree)")
                # pcof = get(q.controls[gates[j]].coeffs)
                qcontrol = FortranBSplineControl(degree, n_splines, T)
                add_control(q, gates[j], qcontrol, coeffs = copy(pcof0), overwrite_control = true)
                println("New degree: $(get(q.controls[gates[j]].objs).degree).")
            end
            println("Gate: ", gates[j])
            global data_count += 1
            # optimize controls
            # only need to re-optimize controls if previous measured infidelity was too large 
            #(q_p_meas_infidelity_total[iter_count - 1][gates[j]]) > epsilon ||
            # println("Coeffs before: ", get(q.controls[gates[j]].coeffs))
            if  (q_p_meas_infidelity_total[iter_count - 1][gates[j]]) > epsilon || (q_pred_infidelity_total[iter_count - 1][gates[j]] > epsilon)
                println("re-optimizing,  measured infidelity: ", q_p_meas_infidelity_total[iter_count-1][gates[j]])
                println("re-optimizing,  predicted infidelity: ", q_pred_infidelity_total[iter_count-1][gates[j]])
                optimize_control(q, gates[j], options=["max_iter" => 100, "print_level" => 5], dt = dt, iter = iter_count)
                # println("Coeffs after: ", get(q.controls[gates[j]].coeffs))
            end
            # measure infidelity and add values to respective dictionary
            q_s_inf, q_p_inf, pop_history = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
            println("Measured infidelity of $(gates[j]) at iteration $iter_count: ", q_s_inf)
            q_p_meas_infidelity[gates[j]] = q_p_inf 
            q_s_meas_infidelity[gates[j]] = q_s_inf 
            control_dict[gates[j]] = q.controls[gates[j]]
            q_pred_infidelity[gates[j]] = abs(get(q.infidelity[gates[j]]))
            # add pcof and data to total amount of data for characterization
            event_obs_total[j] = pop_history
            iter, pcof_optimal = last(control_dict[gates[j]].coeffs)
            pcof_optimal_total[j] = copy(get(q.controls[gates[j]].coeffs))
            # pcof_optimal_total[j] = pcof_optimal
            n_splines_total[j] = n_splines
            T_total[j] = T_gate 
            degree_total[j] = degree 
            nsteps_total[j] = nsteps

        end  
        if save_data 
            save_object(joinpath(folder, "event_obs_iteration_$(iter_count).jld2"), event_obs_total)
            save_object(joinpath(folder, "pcof_optimal_total_$(iter_count).jld2"), pcof_optimal_total)
        end
        

        # THIS IS THE REASON NOTHING IS WORKING!!!
        # control_dict_total has coefficients which are the same, not distinguishing between gate or even iteration
        
        control_dict_total[iter_count + 1] = control_dict
        q_pred_infidelity_total[iter_count] = q_pred_infidelity
        q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
        q_s_meas_infidelity_total[iter_count] = q_s_meas_infidelity

        # check the coefficients of control_dict 
        # println("Pauli X coeffs: ")
        # display(pcof_optimal_total[1])
        # println("Pauli Y coeffs: ")
        # display(pcof_optimal_total[2])
        # println("Pauli Z coeffs: ")
        # display(pcof_optimal_total[3])
        # println("Hadamard coeffs: ")
        # display(pcof_optimal_total[4])
        # Termination condition: both measured and predicted infidelities below epsilon for all gates
        if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
            println("Loop terminated, measured infidelity and predicted infidelity small")
            break 
        else
            println("Re-run characterization")
        end

        

        # Run W2-chain inference on the new accumulated data from all gates in this iteration
        w2_chain = run_w2_chain_quantum(
            event_obs_total;
            prior = prior_downgrade, 
            ω0= ω0, 
            ωr= 4.5, 
            degree = degree_total, 
            n_splines = n_splines_total, 
            U0 = U0, 
            T = T_total, 
            nsteps = nsteps_total, 
            pcof_optimal_total = pcof_optimal_total,
            total_data_count = N_gates,
            λ=10.0,
            iterations=iterations,
            burnin=burnin,
            thin=thin,
            ωmin=4.0,
            ωmax=5.0,
            δ=2.0,
            scale_factor=length(event_obs_total[1,:,1]),
            risk_scale=1.0,
            t0_adapt=100,
            target_accept=0.44,
            rng=Random.seed!(30),
        )

        omega_samples = w2_chain.chain_post[:,1]
        # plot histogram of omega_samples for this iteration
        add_param_samples(q, omega_samples, xi_samples; iter = iter_count + 1, average_omega_rot = false)

    end

    

    delete!(q_pred_infidelity_total, 0)
    delete!(q_p_meas_infidelity_total, 0)
    delete!(q_s_meas_infidelity_total, 0)

else
    println("Skipping characterization and control optimization loop, setting up data structures for plotting only.")
end

if save_data
    @save joinpath(folder, "data.jld2") q control_dict_total q_pred_infidelity_total q_p_meas_infidelity_total q_s_meas_infidelity_total prior_vec
end

if load_data 
    tag = "1" # default tag if loading data, will be overwritten by actual tag in loaded data
    experiment_folder = experiment_name * "_" * tag
    @load joinpath(experiment_folder, "data.jld2") q control_dict_total q_pred_infidelity_total q_p_meas_infidelity_total q_s_meas_infidelity_total prior_vec
    iter_count = length(keys(q_pred_infidelity_total))
end

# -------------------------
# Plotting and visualization
# -------------------------
# I want to plot the predicted and measured infidelities across iterations, as well as the histograms of the parameter samples across iterations, and also the control pulses across iterations.

# get default color palette and make it so that the colors cycle through the palette for each iteration

# trim unused parts of vectors and dictionaries

q_pred_infidelity_reduced = Dict(k => v for (k,v) in q_pred_infidelity_total if k <= iter_count)
q_p_meas_infidelity_reduced = Dict(k => v for (k,v) in q_p_meas_infidelity_total if k <= iter_count)
q_s_meas_infidelity_reduced = Dict(k => v for (k,v) in q_s_meas_infidelity_total if k <= iter_count)
prior_vec_reduced = prior_vec[1:iter_count + 1]


n = length(q_pred_infidelity_total)
if n > 5 
    num_to_plot = 5
    inds = round.(Int, range(1, n, length=num_to_plot))
    println(inds)
    prior_inds = round.(Int, range(1, length(prior_vec_reduced), length=num_to_plot))
    println(prior_inds)
    q_pred_infidelity_reduced = Dict(k => q_pred_infidelity_reduced[k] for k in inds)
    q_p_meas_infidelity_reduced = Dict(k => q_p_meas_infidelity_reduced[k] for k in inds)
    q_s_meas_infidelity_reduced = Dict(k => q_s_meas_infidelity_reduced[k] for k in inds)
    prior_vec_reduced = prior_vec_reduced[prior_inds]
else
    inds = round.(Int, range(1, n, length=n))
    prior_inds = collect(1:length(prior_vec_reduced))
end

if plot_results
    println("Plotting results")


    # Histogram of first-iteration samples
    hist_plot = histogram(get(q.omega, 1), xlabel = "ω", ylabel = "Density", title = "Parameter Distribution", dpi = 250, label = "Iteration 1", color = colors[2], alpha = 0.7, linewidth = 0.5)
    for i in 2:length(inds)
        println(inds[i])
        hist_plot = histogram!(get(q.omega, inds[i]), label = "Iteration $(inds[i])", color = colors[i + 1], alpha = 0.7, linewidth = 0.5)
    end
    vline!([ω], label = "True ω", color = :black, linestyle=:dash, alpha = 1.0, lw = 2)
    savefig(joinpath(folder, "histogram_iterations.png")) 

    # Helper to convert gate type to string for xtick labels
    gate_to_str(g) = g == PauliX ? "X" :
                    g == PauliY ? "Y" :
                    g == PauliZ ? "Z" :
                    g == Hadamard ? "H" :
                    string(g)

    jitter = 0.05
    keys_vec = collect(keys(q_p_meas_infidelity_total))
    total_keys = length(keys_vec)

    # measured infidelity plot 
    for i in 1:length(inds) # remove first key since that is just the initial characterization with the same pulse across all gates, which is not very interesting to plot since all infidelities are going to be large and the same across gates

        gates = collect(keys(q_p_meas_infidelity_total[i]))
        x = (1:N_gates) .+ (i - (length(inds)+1)/2) * jitter
        meas_vals = collect(values(q_p_meas_infidelity_total[inds[i]]))
        pred_vals = collect(values(q_pred_infidelity_total[inds[i]]))
        xtick_labels = gate_to_str.(gates)
        p_str = "p"*int_to_subscript(inds[i])
        if i == 1
            meas_infidelity_scatter = scatter(x, meas_vals, 
                yscale = :log10,
                xlabel = "Gate",
                ylabel = "Measured Infidelity",
                title = "Measured Infidelity for gate set across re-characterization iterations",
                titlefontsize = 8,
                xticks = (1:N_gates, xtick_labels),
                yticks = 10.0 .^ (-7:0),
                dpi = 300,
                marker = :circle,
                markersize = 6,
                alpha = 0.9,
                color = colors[i + 1],
                label = p_str,
                legend = :outertop,
                legend_background_color = :transparent,
                legendfontsize = 8,
                legendcolumns = 4,
            )
        else
            scatter!(x, meas_vals, 
                marker = :circle,
                markersize = 6,
                alpha = 0.9,
                color = colors[i + 1],
                label = p_str,
            )
        end
    end
    hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)

    savefig(joinpath(folder, "measured_infidelity.png"))

    # do the same above for predicted infidelity
    for i in 1:length(inds)
        gates = collect(keys(q_pred_infidelity_total[inds[i]]))
        x = (1:N_gates) .+ (i - (length(inds)+1)/2) * jitter
        meas_vals = collect(values(q_p_meas_infidelity_total[inds[i]]))
        pred_vals = collect(values(q_pred_infidelity_total[inds[i]]))
        xtick_labels = gate_to_str.(gates)
        p_str = "p"*int_to_subscript(inds[i])
        if i == 1
            pred_infidelity_scatter = scatter(x, pred_vals, 
                yscale = :log10,
                xlabel = "Gate",
                ylabel = "Predicted Infidelity",
                title = "Predicted Infidelity for gate set across re-characterization iterations",
                titlefontsize = 8,
                xticks = (1:N_gates, xtick_labels),
                yticks = 10.0 .^ (-7:0),
                dpi = 300,
                marker = :circle,
                markersize = 6,
                alpha = 0.9,
                color = colors[i + 1],
                label = p_str,
                legend = :outertop,
                legend_background_color = :transparent,
                legendfontsize = 8,
                legendcolumns = 4,
            )
        else
            scatter!(x, pred_vals, 
                marker = :circle,
                markersize = 6,
                alpha = 0.9,
                color = colors[i + 1],
                label = p_str,
            )
        end
    end
    hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)
    savefig(joinpath(folder, "predicted_infidelity.png"))

    # plot priors if using Kernel density estimation, not using this!
    # prior_plot = plot(xs, prior_vec_reduced[1], label = "Prior 0", dpi = 250, title = "Kernel: $dist_func, bandwidth = $bandwidth, downgrade power = $downweight_power", xlabel = "ω", ylabel = "Density", titlefontsize = 8)
    # for i in 2:length(prior_vec_reduced)
    #     plot!(xs, prior_vec_reduced[i], label = "Prior $(prior_inds[i]-1)", color = colors[i], lw = 1)
    # end

    # plot priors using Gaussian fitting
    prior_plot = plot(xs, prior_vec_reduced[1], label = "Prior 0", dpi = 250, title = "Downweighted Gaussian Fit, downweight power = $downweight_power", xlabel = "ω", ylabel = "Density", titlefontsize = 8)
    for i in 2:length(prior_vec_reduced)
        plot!(xs, prior_vec_reduced[i], label = "Prior $(prior_inds[i]-1)", color = colors[i], lw = 1)
    end

    savefig(joinpath(folder, "priors.png"))
    # now plot histograms along with priors for each iterations to see how much the priors are capturing the histograms
    for i in 1:iter_count
        if i + 1 > iter_count
            break
        end
        down_grade_plot = histogram(get(q.omega, i), xlabel = "ω", ylabel = "Density", title = "Parameter Distribution with Prior, Iteration $i", dpi = 250, label = "Samples", color = colors[2], alpha = 0.7, normalize = true)
        plot!(xs, prior_vec[i + 1], label = "Prior", color = colors[3], lw = 3)
        # vline!([ω], label = "True ω", color = :pink, alpha = 1.0, lw = 3)
        savefig(joinpath(folder, "hist_prior_overlay_iter_$(i).png"))
    end
end

=#