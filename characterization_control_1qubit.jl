include("src/QDT_src.jl")


# save_data = true
# load_data = false
# plot_results = false
# run_loop = true
# use_initial_samples = false

# experiment_name = "results/characterization_1qubits"
# if save_data 
#     folder, tag = make_unique_folder(experiment_name)
#     pcof_dir = joinpath(folder, "pcof_optimal")
#     mkpath(pcof_dir)
#     data_folder = joinpath(folder, "data")
#     mkpath(data_folder)
#     chain_data = joinpath(data_folder, "chain_data")
#     mkpath(chain_data)
#     event_obs_folder = joinpath(data_folder, "event_obs")
#     mkpath(event_obs_folder)
#     folder_histogram = joinpath(folder, "histogram")
#     mkpath(folder_histogram)
#     folder_wasserstein = joinpath(folder, "wasserstein")
#     mkpath(folder_wasserstein)
# end


# # -------------------------
# # Basic model and timescale
# # -------------------------

# ω = 4.6 * 2pi
# ωr = 4.5 * 2pi
# ξ = 0.0

# true_params = [ω]

# delta = ω - ωr
# Ne = 2 
# Ng = 0
# N_tot = Ne + Ng 
# M_spam_order = 1E-4 
# n_readout_samples = 100000

# # Simple 2-level control Hamiltonian definitions (X-like real, Y-like imaginary)
# H_control_real = [0.0 1;
# 1 0]
# H_control_imag = [0.0 1;
# -1 0]
# real_control_ops = [H_control_real]
# imag_control_ops = [H_control_imag]

# n_iters_opt = 100
# dt_opt = 0.2
# ωmin = 4.0 * 2pi
# ωmax = 5.0 * 2pi 

# degree = 2 
# degree_init = 2
# epsilon = 1E-4

# n_splines_init = 3
# n_splines = 15 
# T = 50.0
# T_init = T/4 
# nsteps = 200 

# dt = T/nsteps 
# nsteps_init = Int(T_init/dt)

# # create a control object 
# q_control = FortranBSplineControl(degree_init, n_splines_init, T)
# max_control_amplitude = 0.3
# pcof_l = -max_control_amplitude
# pcof_u = max_control_amplitude


# U0 = [1 0; 0 1]



# # Gate set and timing for per-gate control optimization
# gates = [PauliX, PauliY, PauliZ, Hadamard, Tgate]
# N_gates = length(gates)

# #################################################################
# # SETUP
# #################################################################
# q = DigitalQudit(Ne, Ng)
# q.omega_rot = ωr

# pcof_init = (0.5 .- rand(q_control.N_coeff)) .* max_control_amplitude

# q_control_X = QuditControl(q_control)
# q_control_Y = QuditControl(q_control)
# q_control_Z = QuditControl(q_control)
# q_control_H = QuditControl(q_control)
# q_control_T = QuditControl(q_control)

# for i in 1:N_gates
#     add_control(q, gates[i], q_control, coeffs = copy(pcof_init))
#     q.infidelity[gates[i]] = History(Float64)
# end



# if save_data
#     save_object(joinpath(pcof_dir, "initial_pcof.jld2"), pcof_init)
# end

# prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
# phys_q = PhysicalQudit(
#                     Ne, Ng, 
#                     ω, ωr, ξ,
#                     M_spam_order=M_spam_order
#                 )


# #######################################################
# # Create storage for desired quantities 
# #######################################################

# max_characterizations = 10
# initial_dataset_size = 1

# event_obs = forward_event_quantum(ω, ωr, degree_init, n_splines_init, U0, T_init, nsteps_init, pcof_init)
# M_spam = phys_q.M_spam 
# event_obs_SPAM = sample_quantum_state_history(n_readout_samples, M_spam, event_obs)

# event_obs_init = Vector{Array{Float64}}(undef, initial_dataset_size)
# event_obs_init[1] = event_obs_SPAM

# pcof_initial_total = Vector{Vector{Float64}}(undef, initial_dataset_size)
# pcof_initial_total[1] = pcof_init

# event_obs_total = Vector{Array{Float64}}(undef, initial_dataset_size)

# variance_list = [] 
# mean_list = []
# rand_seed_list = []
# param_init_list = []

# q_pred_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
# q_p_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()

# n_splines_total = fill(n_splines, N_gates)
# T_total = fill(T, N_gates)
# nsteps_total = fill(nsteps, N_gates)
# degree_total = fill(degree, N_gates)
# total_data_count = N_gates 

# control_params_init = [degree_init, n_splines_init, T_init, nsteps_init]
# control_params_total = [degree_total, n_splines_total, nsteps_total]

# downweight_power = 0.8
# iter_count = 1 

# iterations = 5000 
# burnin = 4000
# thin = 4
# t0_adapt = 200 

# MHG_params = [iterations, burnin, thin, t0_adapt, ωmin, ωmax]

function characterization_control_1qubit(max_characterizations; initial_guess::Union{AbstractVecOrMat, Nothing} = nothing)
    save_data = true
    load_data = false
    plot_results = false
    run_loop = true
    use_initial_samples = false

    experiment_name = "results/characterization_1qubits"
    if save_data 
        folder, tag = make_unique_folder(experiment_name)
        pcof_dir = joinpath(folder, "pcof_optimal")
        mkpath(pcof_dir)
        data_folder = joinpath(folder, "data")
        mkpath(data_folder)
        chain_data = joinpath(data_folder, "chain_data")
        mkpath(chain_data)
        event_obs_folder = joinpath(data_folder, "event_obs")
        mkpath(event_obs_folder)
        folder_histogram = joinpath(folder, "histogram")
        mkpath(folder_histogram)
        folder_wasserstein = joinpath(folder, "wasserstein")
        mkpath(folder_wasserstein)
    end


    # -------------------------
    # Basic model and timescale
    # -------------------------

    ω = 4.6 * 2pi
    ωr = 4.5 * 2pi
    ξ = 0.0

    true_params = [ω, ωr]

    delta = ω - ωr
    Ne = 2 
    Ng = 0
    N_tot = Ne + Ng 
    M_spam_order = 1E-4
    n_readout_samples = 100000

    # Simple 2-level control Hamiltonian definitions (X-like real, Y-like imaginary)
    H_control_real = [0.0 1;
    1 0]
    H_control_imag = [0.0 1;
    -1 0]
    real_control_ops = [H_control_real]
    imag_control_ops = [H_control_imag]

    n_iters_opt = 100
    dt_opt = 0.2
    ωmin = 4.0 * 2pi
    ωmax = 5.0 * 2pi 

    degree = 2 
    degree_init = 2
    epsilon = 1E-4

    n_splines_init = 10
    n_splines = 15 
    T = 50.0
    T_init = T/8
    nsteps = 200 

    dt = T/nsteps 
    nsteps_init = Int(T_init/dt)

    # create a control object 
    q_control = FortranBSplineControl(degree_init, n_splines_init, T)
    max_control_amplitude = 0.3
    pcof_l = -max_control_amplitude
    pcof_u = max_control_amplitude


    U0 = [1 0; 0 1]



    # Gate set and timing for per-gate control optimization
    gates = [PauliX, PauliY, PauliZ, Hadamard, Tgate]
    N_gates = length(gates)

    #################################################################
    # SETUP
    #################################################################
    q = DigitalQudit(Ne, Ng)
    q.omega_rot = ωr

    q_history = Vector{Any}()
    push!(q_history, q)
    Random.seed!(42)
    pcof_init = (0.5 .- rand(q_control.N_coeff)) .* max_control_amplitude

    q_control_X = QuditControl(q_control)
    q_control_Y = QuditControl(q_control)
    q_control_Z = QuditControl(q_control)
    q_control_H = QuditControl(q_control)
    q_control_T = QuditControl(q_control)

    for i in 1:N_gates
        add_control(q, gates[i], q_control, coeffs = copy(pcof_init))
        q.infidelity[gates[i]] = History(Float64)
    end



    if save_data
        save_object(joinpath(pcof_dir, "initial_pcof.jld2"), pcof_init)
    end

    prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
    phys_q = PhysicalQudit(
                        Ne, Ng, 
                        ω, ωr, ξ,
                        M_spam_order=M_spam_order
                    )


    #######################################################
    # Create storage for desired quantities 
    #######################################################

    initial_dataset_size = 1

    event_obs = forward_event_quantum(ω, ωr, degree_init, n_splines_init, U0, T_init, nsteps_init, pcof_init)
    M_spam = phys_q.M_spam 
    # eps_1 = 0.0001s
    # eps_2 = 0.0002
    # M_spam = [1.0 - eps_1 eps_2; eps_1, 1 - eps_2]
    # M_spam = Matrix(1.0*I, 2, 2)
    event_obs_SPAM = sample_quantum_state_history(n_readout_samples, M_spam, event_obs)

    event_obs_init = Vector{Array{Float64}}(undef, initial_dataset_size)
    event_obs_init[1] = event_obs_SPAM

    save_object(joinpath(event_obs_folder, "event_obs_0.jld2"), event_obs_init)

    pcof_initial_total = Vector{Vector{Float64}}(undef, initial_dataset_size)
    pcof_initial_total[1] = pcof_init

    event_obs_total = Vector{Array{Float64}}(undef, initial_dataset_size)

    variance_list = [] 
    mean_list = []
    rand_seed_list = []
    param_init_list = []

    q_pred_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
    q_p_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()

    n_splines_total = fill(n_splines, N_gates)
    T_total = fill(T, N_gates)
    nsteps_total = fill(nsteps, N_gates)
    degree_total = fill(degree, N_gates)
    total_data_count = N_gates 

    control_params_init = [degree_init, n_splines_init, T_init, nsteps_init]
    control_params_total = [degree_total, n_splines_total, T_total, nsteps_total]

    downweight_power = 0.8
    iter_count = 1 

    iterations = 5000
    burnin = 4000
    thin = 4
    t0_adapt = 200 

    MHG_params = [iterations, burnin, thin, t0_adapt, ωmin, ωmax]
    if isnothing(initial_guess)
        ω_init = [rand(Uniform(ωmin, ωmax))]
    else
        ω_init = initial_guess
    end
    event_obs = Vector{Array{Float64}}(undef, N_gates)
    pcof_optimal_total = Vector{Vector{Float64}}(undef, N_gates)
    downgrade_prior = nothing 
    control_dict_total = OrderedDict{Int, OrderedDict{GateType, QuditControl}}()
    iter_count = 1
    for i in 1:max_characterizations
        push!(param_init_list, ω_init)
        q_pred_infidelity = OrderedDict{GateType, Float64}()
        q_p_meas_infidelity = OrderedDict{GateType, Float64}()
        rand_seed = rand(RandomDevice(), UInt64)
        push!(rand_seed_list, rand_seed)
    
        if i == 1
            w2_chain = run_w2_chain_quantum_adaptive(
                    event_obs_init; 
                    ω0_vec = ω_init,
                    ωr = ωr, 
                    λ_log0 = log(10.0),
                    degree = [degree_init],
                    n_splines = [n_splines_init],
                    U0 = U0, 
                    T = [T_init],
                    nsteps = [nsteps_init],
                    pcof_optimal_total = pcof_initial_total, 
                    total_data_count = initial_dataset_size,
                    iterations = iterations, 
                    burnin = burnin, 
                    thin = thin, 
                    ωmin = ωmin,
                    ωmax = ωmax, 
                    δ = 2.0, 
                    scale_factor = nsteps_init + 1,
                    risk_scale = 1.0, 
                    t0_adapt = t0_adapt, 
                    target_accept = 0.44, 
                    rng = Random.seed!(rand_seed)
            )
        else 
            w2_chain = run_w2_chain_quantum_adaptive(
                event_obs; 
                ω_prior = downgrade_prior, 
                ω0_vec = ω_init, 
                λ_log0 = log(10.0),
                ωr = ωr, 
                degree = degree_total, 
                n_splines = n_splines_total, 
                U0 = U0,
                T = T_total,
                nsteps = nsteps_total, 
                pcof_optimal_total = pcof_optimal_total,
                total_data_count = N_gates,
                iterations = iterations, 
                burnin = burnin, 
                thin = thin, 
                ωmin = ωmin, 
                ωmax = ωmax, 
                δ = 2.0, 
                scale_factor = nsteps_total[1] + 1, 
                risk_scale = 1.0, 
                t0_adapt = 100, 
                target_accept = 0.44, 
                rng = Random.seed!(rand_seed)

            )
        end
        omega_samples = w2_chain.diagnostic_chain[:,1]
        add_param_samples(q, omega_samples, zeros(length(omega_samples)), iter = iter_count)
        q.omega_rot = ωr
        display(omega_samples./2pi)
        control_dict = OrderedDict{GateType, QuditControl}()
        for j in 1:N_gates
            println("Gate: $([gates[j]])")
            # Make sure degree and number of splines match the initial and new controls 
            if get(q.controls[gates[j]].objs).degree != degree_total[j]
                println("Degree of control ($(get(q.controls[gates[j]].objs).degree)) for $(gates[j]) does not match desired degree ($degree)")
                # pcof = get(q.controls[gates[j]].coeffs)
                qcontrol = FortranBSplineControl(degree, n_splines, T)
                add_control(q, gates[j], qcontrol, coeffs = pcof0, overwrite_control = true)
                println("New degree: $(get(q.controls[gates[j]].objs).degree).")
            end
            if Int(length(get(q.controls[gates[j]].coeffs))/2) != n_splines_total[j] 
                println("Number of splines ($(Int(length(get(q.controls[gates[j]].coeffs))/2))) does not match desired number of splines ($n_splines)")
                qcontrol = FortranBSplineControl(degree, n_splines, T)
                add_control(q, gates[j], qcontrol, coeffs = (0.5 .- rand(qcontrol.N_coeff)) .* max_control_amplitude, overwrite_control = true)
                println("New number of splines: ", Int(length(get(q.controls[gates[j]].coeffs)))/2)
            end

            pred_infidel = predicted_infidelity(q, gates[j], q.controls[gates[j]], dt = dt)
            control_dict[gates[j]] = q.controls[gates[j]]
            if !haskey(q_p_meas_infidelity_total,iter_count - 1)
                _, meas_infidel, _ = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
            else
                meas_infidel = q_p_meas_infidelity_total[iter_count - 1][gates[j]]
            end
            
            if pred_infidel > epsilon || meas_infidel > epsilon 

                println("re-optimizing, measured infidelity: ", meas_infidel)
                println("re-optimizing, predicted infidelity: ", pred_infidel)
                optimize_control(q, gates[j], options = ["max_iter" => n_iters_opt, "print_level" => 5], dt = dt, iter = iter_count, max_amplitude = max_control_amplitude)
            end
            inf_s, inf_p, pop_history = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
            event_obs[j] = pop_history
            q_p_meas_infidelity[gates[j]] = clamp(inf_p, 1E-6, 1)
            pred_inf = predicted_infidelity(q, gates[j], q.controls[gates[j]], dt = dt)
            q_pred_infidelity[gates[j]] = pred_inf
            pcof_optimal_total[j] = get(q.controls[gates[j]].coeffs)
            println("Measured Infidelity: ", clamp(inf_p, 1E-6, 1))
            println("Predicted Infidelity: ", pred_inf)
            if save_data 
                save_object(joinpath(pcof_dir, "pcof_optimal_$i.jld2"), pcof_optimal_total)
            end
        end

        q_pred_infidelity_total[iter_count] = q_pred_infidelity 
        q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
        control_dict_total[iter_count] = control_dict
        if save_data 
            save_object(joinpath(chain_data, "chain_data_$i.jld2"), w2_chain)
            save_object(joinpath(event_obs_folder, "event_obs_$i.jld2"), event_obs)
        end
        stdev = std(omega_samples)
        mean_omega = mean(omega_samples)
        if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
            println("Loop terminated, measured infidelity and predicted infidelity small")
            break
        else
            println("Re-run characterization")
            prior_tol = 1E-1 
            if all(values(q_p_meas_infidelity) .< prior_tol)
                downgrade_prior = downgrade_gaussian(omega_samples, 1/downweight_power)
                ω_init = [mean_omega]
            else
                println("Measured infidelity of a single gate was larger than $prior_tol")
                downgrade_prior = nothing 
                ω_init = w2_chain.ω_best
            end
            push!(mean_list, mean_omega)
            push!(variance_list, stdev)
        end
        iter_count += 1
        push!(q_history, q)
    end
    if save_data 
        @save joinpath(data_folder, "infidelity_data.jld2") q_pred_infidelity_total q_p_meas_infidelity_total mean_list variance_list rand_seed_list 
        @save joinpath(data_folder, "characterization_params.jld2") true_params param_init_list MHG_params mean_list variance_list downweight_power rand_seed_list 
        @save joinpath(data_folder, "control_params.jld2") gates control_params_init control_params_total control_dict_total
        @save joinpath(data_folder, "qubit_data.jld2") q_history phys_q
    end
    return iter_count
end

max_runs = 20
initial_guess = [4.55 * 2pi]
for i in 1:max_runs
    println("--------------------------------------")
    println("-------------- Run #$i ---------------")
    println("--------------------------------------")
    if i % 2 == 0
        iter_count = characterization_control_1qubit(6, initial_guess = initial_guess)
    else
        iter_count = characterization_control_1qubit(6)
    end
end