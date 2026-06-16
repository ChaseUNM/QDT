# File: /Users/chase/QDT/characterization_control.jl
# Purpose: End-to-end characterization + control optimization loop for a single qudit.
# Notes:
# - This file runs an iterative loop: generate/optimize controls, run physical experiments (simulated),
#   infer parameter posteriors, update priors, and repeat until infidelity tolerance is met.
# - The script is written as a script (top-level); consider refactoring into functions for testability
#   and reusability (see improvement notes below).

using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, OrderedCollections, Dates, MCMCDiagnosticTools
include("src/DistributionFit.jl")
include("src/digital_qudit.jl")
include("src/digital_device.jl")
include("src/physical_device.jl")
include("src/util.jl")
include("src/wasserstein_inference.jl")
include("src/postprocessing.jl")
include("src/forward_model_quantum.jl")


rand_numbers = 1
# Random_seed_list = rand(1:100000, rand_numbers)
# Random_seed_list_2 = collect(101:101)



λ = 0.1


for k in 1:rand_numbers
    # flags for saving and loading data, plotting, and for running the full characterization loop
    save_data = true
    load_data = false
    plot_results = true
    run_loop = true
    use_initial_samples = false

    # -------------------------
    # Basic model and timescale
    # -------------------------
    ω = 4.7
    ωr = 4.5
    ξ = 0.0
    Ne = 2
    Ng = 0
    N_ess = 2
    N_guard = 0
    N_tot = N_ess + N_guard

    # Simple 2-level control Hamiltonian definitions (X-like real, Y-like imaginary)
    H_control_real = [0.0 1;
    1 0]
    H_control_imag = [0.0 1;
    -1 0]
    real_control_ops = [H_control_real]
    imag_control_ops = [H_control_imag]

    # Control parametrization: B-splines
    degree = 2
    n_splines = 16
    T = 50
    nsteps = 200
    dt = T/nsteps

    # Create a control object (Fortran BSpline wrapper)
    control = FortranBSplineControl(degree, n_splines, T)
    max_control_parameter = 0.1
    pcof_l = -max_control_parameter
    pcof_u = max_control_parameter
    # initial coefficients drawn uniformly in [-max, max] scaled by (0.5 - rand)
    # pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter

    # Detuning for rotating frame Schrodinger problem
    delta = ω - ωr
    U0 = [1 0; 0 1]
    # Note: SchrodingerProb is created twice in the file. The first creation here is unused later.
    prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)

    # Gate set and timing for per-gate control optimization
    gates = [PauliX, PauliY, PauliZ, Hadamard]
    gate_set = [PauliX_gate(), PauliY_gate(), PauliZ_gate(), Hadamard_gate()]
    N_gates = length(gates)
    T_gate = 50

    # Measurement/SPAM settings
    M_spam_order = 1e-4
    n_readout_samples = 100000

    # Create a PhysicalQudit instance used for simulating real device outcomes
    phys_q = PhysicalQudit(
                        Ne, Ng, 
                        ω, ωr, ξ,
                        M_spam_order=M_spam_order
                    )

    # Recreate SchrodingerProb again (duplicate of above) - harmless but redundant.
    delta = ω - ωr
    U0 = [1 0; 0 1]
    prob = SchrodingerProb(Float64[0 0; 0 delta], real_control_ops, imag_control_ops, U0, T, nsteps)
    gates = [PauliX, PauliY, PauliZ, Hadamard]
    epsilon = 1E-4

    # This initial physical qudit is used for initial measured infidelities, but this is not necessary at all, can completely ignore as measured infidelities aren't even plotted until after the first characterization. 

    # create maximum amount of data for using characterization
    max_characterizations = 10
    global data_count = 1

    # The following preallocations assume we will fill up to max_data entries.
    # Improvement: consider using Vector{Union{Nothing, T}}(...) or push! semantics instead of large undef arrays.
    initial_dataset_size = 1
    pcof_optimal_total = Vector{Vector{Float64}}(undef, initial_dataset_size)
    event_obs_total = Vector{Array{Float64}}(undef, initial_dataset_size)
    nsteps_total = Vector{Real}(undef, initial_dataset_size)
    T_total = Vector{Real}(undef, initial_dataset_size)
    n_splines_total = Vector{Real}(undef, initial_dataset_size)
    degree_total = Vector{Real}(undef, initial_dataset_size)

    # Need to create the qudit and the controls 
    q = DigitalQudit(Ne, Ng)
    q.omega_rot = ωr
    global iter_count = 0

    # kernel density estimator parameters - these control the shape of the kernel density estimation that is used to create the prior for the next iteration of inference.

    # determine downweighting parameters and kernel type for creating the kernel density estimation. 
    # the two options are "Normal" and "Beta" kernels
    downweight_power = 0.7
    bandwidth = 0.1
    dist_func = "GaussianFit_Seeded"

    # characterization iteration parameters, many iterations, 20% burn-in and thinning every 20 samples
    iterations = 10000
    burnin = 8000
    thin = 2
    total_samples = length(collect(burnin:thin:iterations))

    degree_init = 0


    experiment_name = "results/characterization_control_$(dist_func)_power_$(downweight_power)_samples_$(total_samples)_degree_$(degree_init)_λ_adaptive"



    if save_data || plot_results
        folder, tag = make_unique_folder(experiment_name)
        plot_dir = joinpath(folder, "pcof_optimal")
        mkpath(plot_dir)
        folder_data = joinpath(folder, "data")
        mkpath(folder_data)
        folder_event_obs = joinpath(folder, "data", "event_obs")
        mkpath(folder_event_obs)
        folder_chain_data = joinpath(folder, "data", "chain_data")
        mkpath(folder_chain_data)
        plot_dir_true = joinpath(folder, "true_distributions")
        mkpath(plot_dir_true)
        plot_dir_wasserstein = joinpath(folder, "wasserstein")
        mkpath(plot_dir_wasserstein)
        folder_histogram = joinpath(folder, "histograms")
        mkpath(folder_histogram)
        folder_infidelity = joinpath(folder, "infidelity")
        mkpath(folder_infidelity)
    end 

    
    
    ωmin = 4.0
    ωmax = 5.0
    n_pts = 1000
    ωs = LinRange(ωmin, ωmax, n_pts)
    xs = LinRange(ωmin, ωmax, n_pts)

    # set color palette for plotting - this is used for consistency across all plots, and to ensure that the same colors are used for the same gates/metrics across iterations.
    colors = palette(:default)
    colors_palette = [colors[1], colors[2], colors[4], colors[5], colors[6]]


    if run_loop
        ω0_history = []
        area_history = []
        ω0 = Vector(LinRange(ωmin + 0.1*(ωmax - ωmin), ωmax - 0.1*(ωmax - ωmin), 4))
        # ω0 = [4.5]
        push!(ω0_history, ω0)
        
        rand_seeds = []
        # println("Running characterization and control optimization loop with kernel: $dist_func, bandwidth: $bandwidth, downweight_power: $downweight_power")
        println("Running characterization and control optimization loop with downweighted Gaussian fit, downweight_power: $downweight_power")

        # Dictionaries indexed by iteration -> per-gate dictionaries
        control_dict_total = OrderedDict{Int, Dict{GateType, QuditControl}}()
        q_pred_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
        q_p_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()
        q_s_meas_infidelity_total = OrderedDict{Int, Dict{GateType, Float64}}()

        mean_list = []
        std_dev_list = []
        rhat_list = []
        ess_list = []

        # Create 1 pulse initially then use that for first characterization iteration
        control_dict = OrderedDict{GateType, QuditControl}()
        q_pred_infidelity = OrderedDict{GateType, Float64}()
        q_p_meas_infidelity = OrderedDict{GateType, Float64}()
        q_s_meas_infidelity = OrderedDict{GateType, Float64}()
        pcof0 = zeros(2*n_splines)
        pcof0[1:n_splines] .= 0.9*max_control_parameter*ones(n_splines)
        # pcof0[1:n_splines] .= 0.9*max_control_parameter*LinRange(-1, 1, n_splines)
        # pcof0[1:n_splines] .= 0.9*max_control_parameter*rand(n_splines)
        pcof0[n_splines+1:end] .= 0.1*max_control_parameter*ones(n_splines)
        # pcof0[n_splines+1:end] .= 0.1*max_control_parameter*LinRange(-1, 1, n_splines) 
        # pcof0[n_splines+1:end] .= 0.1*max_control_parameter*rand(n_splines)
        println("Creating controls")
        for j = 1:N_gates
            # Control for this gate with constant initial pulses
            qcontrol = FortranBSplineControl(degree_init, n_splines, T_gate)
            # pcof0 = 0.5*max_control_parameter*ones(qcontrol.N_coeff)
            # pcof0 = max_control_parameter*rand(qcontrol.N_coeff)
            
            add_control(q, gates[j], qcontrol; coeffs = copy(pcof0))
            if plot_results
                
                control_plot = plot_controls(qcontrol, pcof0)
                savefig(control_plot, joinpath(plot_dir, "initial_control.png"))
            end
            control_dict[gates[j]] = q.controls[gates[j]]
            # Infidelities for this gate
            q.infidelity[gates[j]] = History(Float64)
            # measure_infidelity returns (state-based infidelity, process-based infidelity, pop_history)
            q_s_inf, q_p_inf, pop_history = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
            # measure predicted infidelity under a slightly different physical model (phys_q_init)
            # _, q_pred_init, _ = measure_infidelity(phys_q_init, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt; add_SPAM = false)
            q_p_meas_infidelity[gates[j]] = q_p_inf 
            q_s_meas_infidelity[gates[j]] = q_s_inf 
            q_pred_infidelity[gates[j]] = 1.0
        end
        control_dict_total[iter_count + 1] = control_dict
        q_pred_infidelity_total[iter_count] = q_pred_infidelity
        q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
        q_s_meas_infidelity_total[iter_count] = q_s_meas_infidelity


        # Run control pulse on physical device (simulated)
        _, event_obs = run_control_physical(phys_q, control_dict_total[data_count][gates[1]]; dt = dt)
        event_obs = abs2.(event_obs)

        # Add measurement noise via SPAM matrix and sampling of quantum state history.
        # M_spam = column_stochastic(1E-4*rand(2))
        event_obs = sample_quantum_state_history(100000, phys_q.M_spam, event_obs)

        # Record dataset metadata for index 1
        nsteps_total[1] = nsteps
        T_total[1] = dt*nsteps_total[1]
        degree_total[1] = degree_init
        n_splines_total[1] = n_splines
        event_obs_total[data_count] = event_obs[:,1:nsteps_total[1]+1, :] 
        pcof_optimal_total[data_count] = get(control_dict_total[data_count][PauliX].coeffs)
        if save_data 
            save_object(joinpath(folder_event_obs, "event_obs_iteration_$(iter_count).jld2"), event_obs_total)
            save_object(joinpath(plot_dir, "pcof_optimal_total_$(iter_count).jld2"), pcof_optimal_total)
        end
        # Initialize prior vector storage: storing discretized pdf values on xs grid per iteration
        prior_vec = Vector{Vector{Float64}}(undef, max_characterizations + 1) # +1 to account for initial prior before loop
        # init_guess_vec = Vector{Float64}(undef, max_characterizations + 1)

        d = Uniform(ωmin, ωmax)
        # init_guess_vec[1] = ω0
        ys = pdf.(d, xs)
        prior_vec[iter_count + 1] = ys

        
        if use_initial_samples
            omega_samples = load("omega_samples.jld2")
            xi_samples = zeros(length(omega_samples))
            add_param_samples(q, omega_samples, xi_samples; iter = iter_count + 1, average_omega_rot = false)
        else
            rand_number = rand(1:10000)
            # Run an initial W2-chain inference over the first dataset (single run)
            w2_chain = run_w2_chain_quantum_adaptive(
                event_obs_total;
                ω0_vec= ω0, 
                ωr= 4.5, 
                λ_log0 = log(0.1),
                degree = degree_total, 
                n_splines = n_splines_total, 
                U0 = U0, 
                T = T_total, 
                nsteps = nsteps_total, 
                pcof_optimal_total = pcof_optimal_total,
                total_data_count = data_count,
                iterations=iterations,
                burnin=burnin,
                thin=thin,
                ωmin=ωmin,
                ωmax=ωmax,
                δ=2.0,
                scale_factor=length(event_obs_total[1][1,:,1]),
                risk_scale=1.0,
                t0_adapt=100,
                target_accept=0.44,
                rng=Random.seed!(rand_number),
            )
            push!(rand_seeds, rand_number)
            # Extract samples for ω and create zero xi samples vector (only ω inferred here)
            omega_samples = w2_chain.chain_post[:,1]
            omega_samples_single_chain = w2_chain.diagnostic_chain[:,1]
            # get R_hat and ESS for the chains
            R_hat = rhat(w2_chain.diagnostic_chain)
            ESS = ess(w2_chain.diagnostic_chain)
             println("R_hat: ", R_hat)
            println("ESS: ", ESS)
            
            push!(rhat_list, R_hat)
            push!(ess_list, ESS)
            # println(omega_samples)
            push!(mean_list, mean(omega_samples))
            push!(std_dev_list, std(omega_samples))
            #savefig of wasserstein distance function 
            if save_data
                save_object(joinpath(folder_chain_data, "w2_chain_$(iter_count).jld2"), w2_chain)
            end
            if plot_results 
                
                f_vec = zeros(n_pts)
                wasserstein_vec = zeros(n_pts)
                for i in 1:n_pts 
                    f_vec[i], _, wasserstein_vec[i], _ = true_posterior(event_obs_total, 
                    Uniform(ωmin, ωmax),
                    ωs[i], ωr, degree_total, n_splines_total, U0, T_total, 
                    nsteps_total, pcof_optimal_total, 1, λ)
                end
                f_normal, area_normal = normalize_function(ωs, f_vec)
                push!(area_history, area_normal)
                true_distribution = plot(ωs, f_normal.(ωs), linewidth = 2, label = "Posterior", dpi = 250)
                histogram!(omega_samples, normalize = true, alpha = 0.5, label = "Samples")
                savefig(joinpath(plot_dir_true, "true_distribution_$(iter_count).png"))
                wasserstein_plot = plot(ωs, wasserstein_vec, xlabel = "ω", ylabel = "Φ", yscale =:log10, dpi = 250, xlims = (ωmin, ωmax))
                savefig(joinpath(plot_dir_wasserstein, "wasserstein_landscape_$(iter_count).png"))
                omega_histogram = histogram(omega_samples, normalize = true, linewidth = 0.1, xlabel = "ω", label = "Samples", xlims = (ωmin, ωmax))
                combined_plot = plot(omega_histogram, wasserstein_plot, layout = (2,1), dpi = 250)
                savefig(joinpath(folder_histogram, "combined_plot_$(iter_count).png"))
            end
            xi_samples = zeros(length(omega_samples))
            if R_hat < 1.01 && ESS > 400
                println("Statistics look good just picking one of the chains")
            else
                println("Statistics are bad, still just picking one of the chains, but know that the statistics are bad.")
            end
            add_param_samples(q, omega_samples_single_chain, xi_samples; iter = iter_count + 1, average_omega_rot = false)
        end
        

        # Add parameter samples to the DigitalQudit object for iteration 1
        

        # -----------------------------
        # Main characterization loop
        # -----------------------------
        # Risk-neutral optimization then characterization, repeated up to max_characterizations.
       
            



        for i in 1:max_characterizations
            iterations = 10000
            burnin = 8000
            thin = 2

            rand_number = rand(1:10000)
            push!(rand_seeds, rand_number)
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
                    add_control(q, gates[j], qcontrol, coeffs = pcof0, overwrite_control = true)
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
                println("Measured infidelity of $(gates[j]) at iteration $iter_count: ", q_p_inf)
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
                save_object(joinpath(folder_event_obs, "event_obs_iteration_$(iter_count).jld2"), event_obs_total)
                save_object(joinpath(plot_dir, "pcof_optimal_total_$(iter_count).jld2"), pcof_optimal_total)
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
            
            x_min = minimum(omega_samples)
            x_max = maximum(omega_samples)
            xs = LinRange(x_min, x_max, 1000)
            posterior = gaussian_fit(omega_samples)
            gauss_fit_tol = 1E-1
            if all(values(q_p_meas_infidelity_total[iter_count]) .< 1E-1)
                prior_downgrade = downgrade_gaussian(omega_samples, 1/downweight_power)

                # set mean to either be mean of prior or midpoint of support.

                # ω0 = [mean(prior_downgrade)]
                ω0 = [mean([4.0,5.0])]
                prior_vec[i + 1] = pdf.(truncated_downgrade_gaussian(omega_samples, 1/downweight_power, ωmin, ωmax), xs)
            else
                println("Measured infidelity of a single gate was larger than $gauss_fit_tol, setting prior to be uniform to explore more parameter space.")
                prior_downgrade = Uniform(ωmin, ωmax)
                prior_vec[i + 1] = pdf.(Uniform(ωmin, ωmax), xs)
                ω0 = [mean([4.0,5.0])]
            end
            
            # set ω0 to be sampled from the truncated prior
            # ω0 = rand(truncated(prior_downgrade, ωmin, ωmax), 4) .+ 0.1*randn(4)
            
            push!(ω0_history, ω0)
            current_samples_plot = histogram(omega_samples, xlabel = "ω", ylabel = "Density", title = "Posterior Samples, Iteration $iter_count", dpi = 250, label = "Samples", alpha = 0.5, normalize = true)
            plot!(xs, prior_vec[i + 1], label = "Truncated Prior", lw = 2, alpha = 0.5)
            plot!(xs, pdf.(posterior, xs), label = "Posterior", lw = 2, alpha = 0.7, linestyle =:dash)
            plot!(xs, pdf.(prior_downgrade, xs), label = "Downweighted Posterior", lw = 2, alpha = 0.7, linestyle =:dash)
            λ_samples = w2_chain.chain_post[:,2]
            
            if plot_results
                savefig(joinpath(folder_histogram, "histogram_iteration_$(iter_count).png"))
            end
            λ_histogram = histogram(λ_samples, xlabel = "λ", ylabel = "Density", title = "λ Samples, Iteration $iter_count", dpi = 250, label = "Samples", alpha = 0.5, normalize = true)
            
            if plot_results
                savefig(joinpath(folder_histogram, "lambda_histogram_iteration_$(iter_count).png"))
            end


            if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
                println("Loop terminated, measured infidelity and predicted infidelity small")
                break 
            else
                println("Re-run characterization")
            end
            # Run W2-chain inference on the new accumulated data from all gates in this iteration
            w2_chain = run_w2_chain_quantum_adaptive(
                event_obs_total;
                ω_prior = prior_downgrade, 
                λ_log_prior = nothing,
                λ_log0 = log(0.1),
                ω0_vec= ω0, 
                ωr= 4.5, 
                degree = degree_total, 
                n_splines = n_splines_total, 
                U0 = U0, 
                T = T_total, 
                nsteps = nsteps_total, 
                pcof_optimal_total = pcof_optimal_total,
                total_data_count = N_gates,
                iterations=iterations,
                burnin=burnin,
                thin=thin,
                ωmin=4.0,
                ωmax=5.0,
                δ=2.0,
                scale_factor=length(event_obs_total[1][1,:,1]),
                risk_scale=1.0,
                t0_adapt=100,
                target_accept=0.44,
                rng=Random.seed!(rand_number),
            )
            R_hat = rhat(w2_chain.diagnostic_chain)
            ESS = ess(w2_chain.diagnostic_chain)
            push!(rhat_list, R_hat)
            push!(ess_list, ESS)
            omega_samples = w2_chain.chain_post[:,1]
            λ_samples = w2_chain.chain_post[:,2]
            if save_data
                save_object(joinpath(folder_chain_data, "w2_chain_$(iter_count).jld2"), w2_chain)
            end
            if plot_results 
                
                f_vec = zeros(n_pts)
                wasserstein_vec = zeros(n_pts)
                for i in 1:n_pts 
                    f_vec[i], _, wasserstein_vec[i], _ = true_posterior(event_obs_total, 
                    truncated(prior_downgrade, ωmin, ωmax),
                    ωs[i], ωr, degree_total, n_splines_total, U0, T_total, 
                    nsteps_total, pcof_optimal_total, N_gates, λ)
                end
                f_normal, area_normal = normalize_function(ωs, f_vec)
                true_distribution = plot(ωs, f_normal.(ωs), linewidth = 2, label = "Posterior", dpi = 250)
                push!(area_history, area_normal)
                histogram!(omega_samples, normalize = true, alpha = 0.5, label = "Samples")
                savefig(joinpath(plot_dir_true, "true_distribution_$(iter_count).png"))
                wasserstein_plot = plot(ωs, wasserstein_vec, xlabel = "ω", ylabel = "Φ", yscale =:log10, dpi = 250)
                savefig(joinpath(plot_dir_wasserstein, "wasserstein_landscape_$(iter_count).png"))
                omega_histogram = histogram(omega_samples, normalize = true, linewidth = 0.1, xlabel = "ω", label = "Samples", xlims = (ωmin, ωmax))
                combined_plot = plot(omega_histogram, wasserstein_plot, layout = (2,1), dpi = 250)
                savefig(joinpath(folder_histogram, "combined_plot_$(iter_count).png"))
            end
            push!(mean_list, mean(omega_samples))
            push!(std_dev_list, std(omega_samples))
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
        
        @save joinpath(folder_data, "data.jld2") q control_dict_total q_pred_infidelity_total q_p_meas_infidelity_total q_s_meas_infidelity_total prior_vec mean_list std_dev_list rand_seeds ω0_history rhat_list ess_list area_history
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
        savefig(joinpath(folder_histogram, "histogram_iterations.png")) 

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

        savefig(joinpath(folder_infidelity, "measured_infidelity.png"))

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
        savefig(joinpath(folder_infidelity, "predicted_infidelity.png"))

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

        savefig(joinpath(folder_histogram, "priors.png"))
        # # now plot histograms along with priors for each iterations to see how much the priors are capturing the histograms
        # for i in 1:iter_count
        #     if i + 1 > iter_count
        #         break
        #     end
        #     down_grade_plot = histogram(get(q.omega, i), xlabel = "ω", ylabel = "Density", title = "Parameter Distribution with Prior, Iteration $i", dpi = 250, label = "Samples", color = colors[2], alpha = 0.7, normalize = true)
        #     plot!(xs, prior_vec[i + 1], label = "Prior", color = colors[3], lw = 3)
        #     # vline!([ω], label = "True ω", color = :pink, alpha = 1.0, lw = 3)
        #     savefig(joinpath(folder, "hist_prior_overlay_iter_$(i).png"))
        # end
    end
end