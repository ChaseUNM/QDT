using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, OrderedCollections, Dates, MCMCDiagnosticTools
include("src/DistributionFit.jl")
include("src/digital_qudit.jl")
include("src/digital_device.jl")
include("src/physical_device.jl")
include("src/util.jl")
include("src/wasserstein_inference.jl")
include("src/postprocessing.jl")
include("src/forward_model_quantum.jl")

samples = 1001
downweight_power = 0.8
λ_adaptive = true
λ = 0.1

tag = 3
degree_init = 2
characterization_iter = 1

# get path used to store characterization data for this set of parameters, and load the characterization data

folder = λ_adaptive ? "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_adaptive_$tag" : "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_$(λ)_$tag"
# get w2_chain data
w2_chain_data = load(joinpath(folder, "data", "chain_data", "w2_chain_$(characterization_iter).jld2"))["single_stored_object"]


if λ_adaptive
    λ_samples = w2_chain_data.chain_post[:,2]
    λ_hist = histogram(λ_samples)
    λ_trace = plot(λ_samples)
end



omega_samples = w2_chain_data.chain_post[:,1]


posterior = gaussian_fit(omega_samples)
downweight_power = 1.0
prior_downgrade = downgrade_gaussian(omega_samples, 1/downweight_power)


ω0 = mean(prior_downgrade)
ωr = 4.5



ωs = LinRange(4.0, 5.0, 1001)

prior_plot = histogram(omega_samples, normalize = true)
plot!(ωs, pdf.(truncated(prior_downgrade, 4.0, 5.0), ωs))



diagnostic_chain_histograms = histogram()
for i in 1:size(w2_chain_data.diagnostic_chain)[2]
    histogram!(w2_chain_data.diagnostic_chain[:,i], alpha = 0.5, label = "Chain $i")
end

println("Rhat: ", rhat(w2_chain_data.diagnostic_chain))


# pcof_optimal_total[1] = get(control_dict_total[characterization_iter][PauliX].coeffs)
# pcof_optimal_total[2] = get(control_dict_total[characterization_iter][PauliY].coeffs)
# pcof_optimal_total[3] = get(control_dict_total[characterization_iter][PauliZ].coeffs)
# pcof_optimal_total[4] = get(control_dict_total[characterization_iter][Hadamard].coeffs)





run_chain = false
normalize = false
if run_chain
    ωmin = 4.0
    ωmax = 5.0
    ω0_vec = Vector(LinRange(ωmin + 0.1*(ωmax - ωmin), ωmax - 0.1*(ωmax - ωmin), 5))
    # ω0_vec = [mean(prior_downgrade)]
    ω0_vec = [4.1]
    # ω0_vec = [4.5]
    rand_seed = rand(1:100000)
    println("Rand seed: ", rand_seed)
    pcof_optimal_total = load(joinpath(folder, "pcof_optimal", "pcof_optimal_total_$(characterization_iter + 1).jld2"))["single_stored_object"]
    event_obs = load(joinpath(folder, "data", "event_obs", "event_obs_iteration_$(characterization_iter + 1).jld2"))["single_stored_object"]
    n_data = length(event_obs)
    gates = [PauliX, PauliY, PauliZ, Hadamard]
    N_gates = length(gates)
    T_gate = 50
    # degree = characterization_iter == 0 ? 0 : 2
    degree = 2
    degree_total = fill(degree, n_data)
    T_total = fill(T_gate, n_data)
    nsteps_total = fill(200, n_data) 
    
    U0 = [1 0; 0 1]

    iterations = 10000
    burnin = 8000
    thin = 2
    n_splines_total = fill(16, n_data)
    
    # w2_chain = run_w2_chain_quantum(
    #     event_obs;
    #     prior = prior_downgrade, 
    #     ω0_vec = ω0_vec, 
    #     ωr= ωr, 
    #     degree = degree_total, 
    #     n_splines = n_splines_total, 
    #     U0 = U0, 
    #     T = T_total, 
    #     nsteps = nsteps_total, 
    #     pcof_optimal_total = pcof_optimal_total,
    #     total_data_count = n_data,
    #     λ= 0.1,
    #     iterations=iterations,
    #     burnin=burnin,
    #     thin=thin,
    #     ωmin=ωmin,
    #     ωmax=ωmax,
    #     δ=2.0,
    #     scale_factor=length(event_obs[1][1,:,1]),
    #     risk_scale=1.0,
    #     t0_adapt=1,
    #     target_accept=0.44,
    #     rng=Random.seed!(rand_seed),
    #         )
    w2_chain_adaptive = run_w2_chain_quantum_adaptive(
        event_obs;
        # prior = prior_downgrade, 
        ω0_vec = ω0_vec, 
        ωr= ωr, 
        λ_log0 = log(0.1),
        degree = degree_total, 
        n_splines = n_splines_total, 
        U0 = U0, 
        T = T_total, 
        nsteps = nsteps_total, 
        pcof_optimal_total = pcof_optimal_total,
        total_data_count = n_data,
        iterations=iterations,
        burnin=burnin,
        thin=thin,
        ωmin=ωmin,
        ωmax=ωmax,
        δ=2.0,
        scale_factor=length(event_obs[1][1,:,1]),
        risk_scale=1.0,
        t0_adapt=100,
        target_accept=0.44,
        rng=Random.seed!(1708)
            )
    # omega_samples_avg = w2_chain.chain_post[:,1]
    omega_samples_adaptive = w2_chain_adaptive.chain_post[:,1]

    diagnostic_chain_histograms = histogram()
    for i in 1:size(w2_chain_adaptive.diagnostic_chain)[2]
        histogram!(w2_chain_adaptive.diagnostic_chain[:,i], alpha = 0.5, label = "Chain $i")
    end
    
    hyperparam_history_ω = w2_chain_adaptive.hyperparam_history_ω

    dθ_plot = plot(hyperparam_history_ω[1,:,1], ylabel = "dθ")
    μθ_plot = plot(hyperparam_history_ω[1,:,2], ylabel = "μ")
    Σθ_plot = plot(hyperparam_history_ω[1,:,3], ylabel = "Σ")
    ηθ_plot = plot(hyperparam_history_ω[1,:,4], ylabel = "η")
    hyperparam_plot = plot(dθ_plot, μθ_plot, Σθ_plot, ηθ_plot, layout = (2,2))
    # println("Standard Deviation of non-adaptive samples: ", std(omega_samples_avg))
    println("Standard Deviation of adaptive samples: ", std(omega_samples_adaptive))
    all_samples = w2_chain_adaptive.chain
    omega_hist = histogram(omega_samples_adaptive, label = "New samples", alpha = 0.5, normalize = normalize, xlims = (4.0, 5.0), linewidth = 0.5)
    # histogram!(omega_samples, label = "Old samples", normalize = normalize, linewidth = 0.5)

    # histogram!(all_samples[:,1], label = "All samples", alpha = 0.1, normalize = normalize)
    # vline!([4.8], label = "True parameter", color=:black, linestyle=:dash, linewidth = 2)

    n_pts = 1000
    ωmin = 4.0
    ωmax = 5.0
    ωs = LinRange(ωmin, ωmax, n_pts)
    wasserstein_vec = zeros(n_pts)
    for i in 1:n_pts
        _, _, wasserstein_vec[i], _ = true_posterior(event_obs, 
        # truncated(prior_downgrade, ωmin, ωmax), 
        Uniform(ωmin, ωmax),
        ωs[i], ωr, degree_total, n_splines_total, U0, T_total, 
        nsteps_total, pcof_optimal_total, n_data, λ)
    end
    wasserstein_plot = plot(ωs, wasserstein_vec, xlabel = "ω", ylabel = "Φ", yscale=:log10)
    total_plot = plot(omega_hist, wasserstein_plot, layout = (2, 1), xlims = (ωmin, ωmax))
    # calculate mean and standard deviation of all samples 
    mean_omega = mean(omega_samples_adaptive)
    std_omega = std(omega_samples_adaptive)
    println("Mean of new samples: ", mean_omega)
    println("Standard deviation of new samples: ", std_omega)
    mean_omega_all = mean(all_samples[:,1])
    std_omega_all = std(all_samples[:,1])
    # println("Mean of all samples: ", mean_omega_all)
    # println("Standard deviation of all samples: ", std_omega_all)
    
    println("Mean of prior samples: ", mean(omega_samples))
    println("Standard deviation of prior samples: ", std(omega_samples))

    # # forward event giving strange data results... 
    # # test explicitly here 
    # U0 = [1 0; 0 1]
    # forward_event_output = forward_event_quantum(4.62, 4.5, 2, 10, U0, 50, 200, pcof_optimal_total[4])

    λ = 0.1
    # println("minimum lp: ", minimum(w2_chain.chain[:,4]))
    # println("maximum lp: ", maximum(w2_chain.chain[:,4]))
    # println("minimum scaled wasserstein: ", -λ*length(event_obs[1][1,:,1]) * minimum(w2_chain.chain[:,3]))
    # println("maximum scaled wasserstein: ", -λ*length(event_obs[1][1,:,1]) * maximum(w2_chain.chain[:,3]))
    chains = w2_chain_adaptive.diagnostic_chain
    n = length(chains)
    rhat_val = rhat(chains)
    ess_val  = ess(chains)

    println("=== Single iid N(0,1) chain (10000 draws) ===")
    println("R-hat : ", round(rhat_val[1], digits=5))
    println("ESS   : ", round(ess_val[1],  digits=1))
    println()
    println("R-hat < 1.01 indicates convergence (rule of thumb)")
    println("ESS/n = ", round(ess_val[1] / n, digits=3), "  (close to 1.0 for iid)")
end

get_other_data = true
    colors = palette(:default)
    # now load the data which consists of data.jld2, /chain_data/ and /event_obs/
    if get_other_data
    if isfile(joinpath(folder, "data", "data.jld2"))
        println("Folder exists")
        @load joinpath(folder, "data", "data.jld2") q control_dict_total q_pred_infidelity_total q_pred_infidelity_total_updated q_p_meas_infidelity_total q_s_meas_infidelity_total prior_vec mean_list std_dev_list rand_seeds ω0_history rhat_list ess_list area_history
        # plot the standard deviation, mean, R_hat, ESS, and area_history 
        mean_plot = plot(mean_list, xlabel = "Iteration", ylabel = "Mean", title = "Mean History", label = "Mean", dpi = 250, marker = :circle)
        hline!([4.62], label = "True Parameter")
        std_dev_plot = plot(std_dev_list, xlabel = "Iteration", ylabel = "Standard Deviation", title = "Standard Deviation History", label = "Standard Deviation", dpi = 250, marker = :circle, yscale=:log10)
        rhat_plot = plot(rhat_list, xlabel = "Iteration", ylabel = "R_hat", title = "R_hat History", label = "R_hat", dpi = 250, marker = :circle)
        ess_plot = plot(ess_list, xlabel = "Iteration", ylabel = "ESS", title = "ESS History", label = "ESS", dpi = 250, marker = :circle)
        area_plot = plot(area_history, xlabel = "Iteration", ylabel = "Area", title = "Area History", label = "Area", dpi = 250, marker = :circle)
        xvals = vcat([fill(i, length(ω0_history[i])) for i in eachindex(ω0_history)]...)
        yvals = vcat(ω0_history...)


       
        

        init_plot = scatter(
            xvals,
            yvals,
            xlabel="Iteration",
            ylabel="ω₀",
            title="ω₀ History",
            label=false,
            dpi=250,
            marker=:circle,
        )


        rand_seed = rand_seeds[characterization_iter + 1]
        omega_samples = get(q.omega, characterization_iter)

        ωmin = 4.0
        ωmax = 5.0
        x_min = minimum(omega_samples)
        x_max = maximum(omega_samples)
        xs = LinRange(x_min, x_max, 1000)
        prior_vec = prior_vec[characterization_iter]
        sample_plots = histogram(xs, omega_samples, label = "Parameter Samples", alpha = 0.8, normalize = normalize)
        plot!(xs, pdf.(posterior, xs), label = "Gaussian Fit")
        plot!(xs, pdf.(prior_downgrade, xs), label = "Prior Downgrade")
        plot!(xs, prior_vec, label = "Prior")
        iter_count = length(keys(q_pred_infidelity_total))
        # calculate posterior distributions directly 

        # plot densities of histograms using KDE
        density_folder = joinpath(folder, "density")
        if !isdir(density_folder)
            mkpath(density_folder)
        end
        density_all = plot(x_label = "ω", ylabel = "Density", dpi = 250)
        density_subset = plot(xlabel = "ω", ylabel = "Density", dpi = 250)
        density_first = plot(xlabel = "ω", ylabel = "Density", dpi = 250)
        for i in 1:iter_count 
            omega_samples = get(q.omega, i)
            k = kde(omega_samples)
            plot!(density_all, k.x, k.density, label = "Iteration $i", color = colors[i + 1])
            if i != 1
                plot!(density_subset, k.x, k.density, label = "Iteration $i", color = colors[i + 1])
            end
            if i == 1
                plot!(density_first, k.x, k.density, label = "Iteration $i", color = colors[i + 1])
            end
        end
        savefig(density_all, joinpath(density_folder, "density_all.png"))
        savefig(density_subset, joinpath(density_folder,"density_subset.png"))
        savefig(density_first, joinpath(density_folder, "density_first.png"))
        # now get a vector of these values 
        n_pts = 1000
        ωmin = 4.0
        ωmax = 5.0
        ωs = LinRange(ωmin, ωmax, n_pts)
        f_vec = zeros(n_pts)
        f_vec_sum = zeros(n_pts)
        wasserstein_vec = zeros(n_pts)
        wasserstein_vec_sum = zeros(n_pts)
        for i in 1:n_pts 
            f_vec[i], f_vec_sum[i], wasserstein_vec[i], wasserstein_vec_sum[i] = true_posterior(event_obs, 
            truncated(prior_downgrade, ωmin, ωmax), 
            # Uniform(ωmin, ωmax),
            ωs[i], ωr, degree_total, n_splines_total, U0, T_total, 
            nsteps_total, pcof_optimal_total, N_gates, λ)
        end
        mean_omega_prior = mean(omega_samples)
        std_omega_prior = std(omega_samples)


        # now interpolate and then normalize 
        wasserstein_plot = plot(ωs, wasserstein_vec, xlabel = "ω", ylabel = "Φ", yscale=:log10, label = "Averaged Wasserstein")
        plot!(ωs, wasserstein_vec_sum, label = "Summed Wasserstein")

        f_non_normal = interpolate_function(ωs, f_vec)
        f_normal, area_avg = normalize_function(ωs, f_vec)
        f_normal_sum, area_sum = normalize_function(ωs, f_vec_sum)
        true_distribution = plot(ωs, f_normal.(ωs), linewidth = 2, label = "Averaged Wasserstein", legend=:topleft)
        # plot!(ωs, f_non_normal.(ωs), label = "Non-normalized wasserstein")
        # plot!(ωs, f_normal_sum.(ωs), linewidth = 2, label = "Summed Wasserstein")
        plot!(ωs, pdf.(truncated(prior_downgrade), ωs), label = "Truncated Prior")
        # plot!(ωs, exp.(-λ*200 .*wasserstein_vec), label = "exponentiated wasserstein")
        # plot!(ωs, exp.(-λ*200 .*wasserstein_vec) .* pdf.(truncated(prior_downgrade), ωs), label = "product")
        # hline!([1.0], label = "1.0", linestyle=:dash, color=:black)
        # histogram!(omega_samples_avg, normalize = true, alpha = 0.5, label = "New Samples", linewidth = 0.1)
        histogram!(omega_samples, normalize = true, alpha = 0.5, label = "Old Samples")
        vline!([4.62], label = "True parameter", linestyle =:dash, linewidth = 0.1)

        f_mean = expected_value_piecewise_gauss(f_normal, ωs)
        f_stdev = std_dev_piecewise_gauss(f_normal, ωs, f_mean)

        println("mean previous samples: ", mean_omega_prior)
        println("standard deviation previous samples: ", std_omega_prior)

        println("prior mean: ", mean(truncated(prior_downgrade)))
        println("prior standard deviation: ", std(truncated(prior_downgrade)))

        println("function mean: ", f_mean)
        println("function standard deviation: ", f_stdev)




        loss_plot = plot(yscale=:log10, dpi = 250)
        # plot loss plot throughout iterations 
        
        # calculate predicted infidelity for samples with with previous control pulses 
        total_iter = length(q_pred_infidelity_total)
        q_pred_infidelity_updated = zeros(N_gates, total_iter)
        for i in 1:total_iter
            pcof_optimal_total = load(joinpath(folder, "pcof_optimal", "pcof_optimal_total_$i.jld2"))["single_stored_object"]
            ω = get(q.omega, i)
            q_new = DigitalQudit(2, 0)
            add_param_samples(q_new, ω, zeros(length(ω)))
            q_new.omega_rot = 4.5
            for j in 1:N_gates 
                pcof = pcof_optimal_total[j]
                qcontrol = FortranBSplineControl(2, 16, 50.0)
                add_control(q_new, gates[j], qcontrol; coeffs = pcof)
                gate = gates[j]
                q_pred_infidelity_updated[j,i] = predicted_infidelity(q, gate, q_new.controls[gate], dt = 0.25)
            end
        end

    else
        println("Folder doesn't exist")
    end
end

calculate_loss = false

if calculate_loss
    plt_labels = ["Constant B-Splines", "Quadratic B-Splines"]
    n_pts = 1000
    ωmin = 4.0
    ωmax = 5.0
    ωs = LinRange(ωmin, ωmax, n_pts)
    wasserstein_vec = zeros(n_pts)
    wasserstein_vec = zeros(n_pts)
    max_loss_tag = Int(max_tag(joinpath(folder, "data", "event_obs")))
    loss_plot = plot(yscale=:log10, xlabel = "ω₁", ylabel = "Wasserstein loss", yticks = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5], title = "Wasserstein loss as a function of ω₁", dpi = 250, legend =:outertop, legend_columns = 2)
    for i in 0:1
        event_obs = load(joinpath(folder, "data", "event_obs", "event_obs_iteration_$i.jld2"))["single_stored_object"]
        pcof_optimal_total = load(joinpath(folder, "pcof_optimal", "pcof_optimal_total_$i.jld2"))["single_stored_object"]
        n_data = length(event_obs)
        println(n_data)
        degree = i == 0 ? 0 : 2
        degree_total = fill(degree, n_data)
        println(degree_total)
        T_total = fill(50.0, n_data)
        nsteps_total = fill(200, n_data) 
        nsplines_total = fill(16, n_data)
        for i in 1:n_pts 
        _, _, wasserstein_vec[i], _ = true_posterior(event_obs, 
            truncated(prior_downgrade, ωmin, ωmax), 
            # Uniform(ωmin, ωmax),
            ωs[i], ωr, degree_total, nsplines_total, U0, T_total, 
            nsteps_total, pcof_optimal_total, n_data, λ)
        end
        
        plot!(loss_plot, ωs, wasserstein_vec, label = plt_labels[i + 1])
    end
end



