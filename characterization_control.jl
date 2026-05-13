# File: /Users/chase/QDT/characterization_control.jl
# Purpose: End-to-end characterization + control optimization loop for a single qudit.
# Notes:
# - This file runs an iterative loop: generate/optimize controls, run physical experiments (simulated),
#   infer parameter posteriors, update priors, and repeat until infidelity tolerance is met.
# - The script is written as a script (top-level); consider refactoring into functions for testability
#   and reusability (see improvement notes below).

using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, OrderedCollections
include("src/KernelDensity.jl")
include("src/digital_qudit.jl")
include("src/physical_device.jl")
include("src/util.jl")
include("src/wasserstein_inference.jl")
include("src/postprocessing.jl")
include("src/forward_model_quantum.jl")

# -------------------------
# Basic model and timescale
# -------------------------
ω = 4.5
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
n_splines = 10 
T = 50
nsteps = 200
dt = T/nsteps

# Create a control object (Fortran BSpline wrapper)
control = FortranBSplineControl(degree, n_splines, T)
max_control_parameter = 0.1
pcof_l = -max_control_parameter
pcof_u = max_control_parameter
# initial coefficients drawn uniformly in [-max, max] scaled by (0.5 - rand)
pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter

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
phys_q_init = PhysicalQudit(
                    Ne, Ng, 
                    4.45, ωr, ξ,
                    M_spam_order=M_spam_order
                )

# create maximum amount of data for using characterization
max_data = 1000
max_characterizations = 100
data_count = 1

# The following preallocations assume we will fill up to max_data entries.
# Improvement: consider using Vector{Union{Nothing, T}}(...) or push! semantics instead of large undef arrays.
pcof_optimal_total = Vector{Vector{Float64}}(undef, max_data)
event_obs_total = Vector{Array{Float64}}(undef, max_data)
nsteps_total = Vector{Real}(undef, max_data)
T_total = Vector{Real}(undef, max_data)
n_splines_total = Vector{Real}(undef, max_data)
degree_total = Vector{Real}(undef, max_data)

# Need to create the qudit and the controls 
q = DigitalQudit(Ne, Ng)
q.omega_rot = ωr
iter_count = 1

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
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    pcof0 = 0.5*max_control_parameter*ones(qcontrol.N_coeff)
    
    add_control(q, gates[j], qcontrol; iter = iter_count, coeffs = pcof0)
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
control_dict_total[iter_count] = control_dict
q_pred_infidelity_total[iter_count] = q_pred_infidelity
q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
q_s_meas_infidelity_total[iter_count] = q_s_meas_infidelity

# create physical qubit with initial parameter guess of 4.45 while keeping rotational frequency at 4.5 to get initial predicted infidelities 

# Run control pulse on physical device (simulated)
_, event_obs = run_control_physical(phys_q, control_dict_total[data_count][gates[1]]; dt = dt)
event_obs = abs2.(event_obs)

# Add measurement noise via SPAM matrix and sampling of quantum state history.
M_spam = column_stochastic(1E-4*rand(2))
event_obs = sample_quantum_state_history(100000, M_spam, event_obs)

# Record dataset metadata for index 1
nsteps_total[1] = 50
T_total[1] = dt*nsteps_total[1]
degree_total[1] = degree
n_splines_total[1] = n_splines
event_obs_total[data_count] = event_obs[:,1:nsteps_total[1]+1, :] 
pcof_optimal_total[data_count] = get(control_dict_total[data_count][PauliX].coeffs)

# Initialize prior vector storage: storing discretized pdf values on xs grid per iteration
prior_vec = Vector{Vector{Float64}}(undef, max_characterizations)
ωmin = 4.0
ωmax = 5.0
d = Uniform(ωmin, ωmax)
xs = LinRange(4.0, 5.0, 1000)
ys = pdf.(d, xs)
prior_vec[iter_count] = ys

# Run an initial W2-chain inference over the first dataset (single run)
w2_chain = run_w2_chain_quantum(
    event_obs_total;
    ω0= 4.3, 
    ωr= 4.5, 
    degree = degree_total, 
    n_splines = n_splines_total, 
    U0 = U0, 
    T = T_total, 
    nsteps = nsteps_total, 
    pcof_optimal_total = pcof_optimal_total,
    total_data_count = data_count,
    λ= 10.0,
    iterations=800,
    burnin=400,
    thin=2,
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

# Add parameter samples to the DigitalQudit object for iteration 1
add_param_samples(q, omega_samples, xi_samples; iter = iter_count, average_omega_rot = false)

# -----------------------------
# Main characterization loop
# -----------------------------
# Risk-neutral optimization then characterization, repeated up to max_characterizations.
downweight_power = 0.2
bandwidth = 0.1
dist_func = "Beta"

for i in 1:max_characterizations

    global iter_count += 1

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
        global data_count += 1
        # optimize controls
        # only need to re-optimize controls if previous measured infidelity was too large 
        if (q_p_meas_infidelity_total[iter_count - 1][gates[j]]) > epsilon || (q_pred_infidelity_total[iter_count - 1][gates[j]] > epsilon)
            println("re-optimizing,  measured infidelity: ", q_p_meas_infidelity_total[iter_count-1][gates[j]])
            println("re-optimizing,  predicted infidelity: ", q_pred_infidelity_total[iter_count-1][gates[j]])
            optimize_control(q, gates[j], options=["max_iter" => 100, "print_level" => 5], dt = dt, iter = iter_count)
        end
        # measure infidelity and add values to respective dictionary
        q_s_inf, q_p_inf, pop_history = measure_infidelity(phys_q, gates[j], q.controls[gates[j]], n_readout_samples, dt = dt)
        q_p_meas_infidelity[gates[j]] = q_p_inf 
        q_s_meas_infidelity[gates[j]] = q_s_inf 
        control_dict[gates[j]] = q.controls[gates[j]]
        q_pred_infidelity[gates[j]] = get(q.infidelity[gates[j]])
        # add pcof and data to total amount of data for characterization
        event_obs_total[j] = pop_history
        iter, pcof_optimal = last(control_dict[gates[j]].coeffs)
        pcof_optimal_total[j] = pcof_optimal
        n_splines_total[j] = n_splines
        T_total[j] = T_gate 
        degree_total[j] = degree 
        nsteps_total[j] = nsteps

    end  

    control_dict_total[iter_count] = control_dict
    q_pred_infidelity_total[iter_count] = q_pred_infidelity
    q_p_meas_infidelity_total[iter_count] = q_p_meas_infidelity
    q_s_meas_infidelity_total[iter_count] = q_s_meas_infidelity

    # Termination condition: both measured and predicted infidelities below epsilon for all gates
    if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
        println("Loop terminated, measured infidelity and predicted infidelity small")
        break 
    else
        println("Re-run characterization")
    end

    # set ω0 to be the average of all previous samples
    omega_samples = get(q.omega, iter_count - 1)
    
    # create kernel estimation that will be downweighted and used as prior for the next inference
    kernel_output = kernel_downweighting(omega_samples, bandwidth, downweight_power, kernel_func = dist_func, sample_min = ωmin, sample_max = ωmax)
    ω0 = expected_value_piecewise_gauss(kernel_output.f_pdf, kernel_output.x_grid)
    prior_vec[iter_count] = kernel_output.f_pdf(xs)
    println("PDF Area: ", gauss_integral(kernel_output.f_pdf, xs))
    println("ω₀ = $ω0")

    # Run W2-chain inference on the new accumulated data from all gates in this iteration
    w2_chain = run_w2_chain_quantum(
        event_obs_total;
        prior = kernel_output, 
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
        iterations=800,
        burnin=400,
        thin=2,
        ωmin=4.0,
        ωmax=5.0,
        δ=2.0,
        scale_factor=length(event_obs_total[1,:,1]),
        risk_scale=1.0,
        t0_adapt=100,
        target_accept=0.44,
        rng=Random.default_rng(),
    )

    omega_samples = w2_chain.chain_post[:,1]

    add_param_samples(q, omega_samples, xi_samples; iter = iter_count, average_omega_rot = false)

end

# Save data if desired
save_data = true

if save_data
    @save "characterization_control_loop_data_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).jld2" q control_dict_total q_pred_infidelity_total q_p_meas_infidelity_total q_s_meas_infidelity_total prior_vec
end

# -------------------------
# Plotting and visualization
# -------------------------
# I want to plot the predicted and measured infidelities across iterations, as well as the histograms of the parameter samples across iterations, and also the control pulses across iterations.

# get default color palette and make it so that the colors cycle through the palette for each iteration
colors = palette(:default)

# Histogram of first-iteration samples
hist_plot = histogram(get(q.omega, 1), xlabel = "ω", ylabel = "Density", title = "Parameter Distribution", dpi = 250, label = "Iteration 1", color = colors[2], alpha = 0.7)
for i in 2:iter_count-1
    hist_plot = histogram!(get(q.omega, i), label = "Iteration $i", color = colors[i + 1], alpha = 0.5)
end
vline!([ω], label = "True ω", color = :pink, alpha = 1.0, lw = 3)
if save_data
    savefig("hist_plot_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).png")
end

# Helper to convert gate type to string for xtick labels
gate_to_str(g) = g == PauliX ? "X" :
                 g == PauliY ? "Y" :
                 g == PauliZ ? "Z" :
                 g == Hadamard ? "H" :
                 string(g)

jitter = 0.1
keys_vec = collect(keys(q_p_meas_infidelity_total))
total_keys = length(keys_vec)

# measured infidelity plot 
for idx in keys_vec[2:end] # remove first key since that is just the initial characterization with the same pulse across all gates, which is not very interesting to plot since all infidelities are going to be large and the same across gates

    gates = collect(keys(q_p_meas_infidelity_total[keys_vec[idx]]))
    x = (1:N_gates) .+ (idx - (length(keys_vec)+1)/2) * jitter
    meas_vals = collect(values(q_p_meas_infidelity_total[keys_vec[idx]]))
    pred_vals = collect(values(q_pred_infidelity_total[keys_vec[idx]]))
    xtick_labels = gate_to_str.(gates)
    p_str = "p"*int_to_subscript(idx)
    if idx == 2
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
            color = colors[idx],
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
            color = colors[idx],
            label = p_str,
        )
    end
end
hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)

if save_data
    savefig("measured_infidelity_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).png")
end

# do the same above for predicted infidelity
for idx in keys_vec[2:end]
    gates = collect(keys(q_pred_infidelity_total[keys_vec[idx]]))
    x = (1:N_gates) .+ (idx - (length(keys_vec)+1)/2) * jitter
    meas_vals = collect(values(q_p_meas_infidelity_total[keys_vec[idx]]))
    pred_vals = collect(values(q_pred_infidelity_total[keys_vec[idx]]))
    xtick_labels = gate_to_str.(gates)
    p_str = "p"*int_to_subscript(idx)
    if idx == 2
        pred_infidelity_scatter = scatter(x, pred_vals, 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Predicted Infidelity",
            title = "Measured Infidelity for gate set across re-characterization iterations",
            titlefontsize = 8,
            xticks = (1:N_gates, xtick_labels),
            yticks = 10.0 .^ (-7:0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = colors[idx],
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
            color = colors[idx],
            label = p_str,
        )
    end
end
hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)
if save_data
    savefig("predicted_infidelity_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).png")
end

# lastly want to plot priors 
prior_plot = plot(xs, prior_vec[1], label = "Prior 1", dpi = 250, title = "Kernel: $dist_func, bandwidth = $bandwidth, downweight_power = $downweight_power", xlabel = "ω", ylabel = "Density", titlefontsize = 8)
for i in 2:iter_count-1
    plot!(xs, prior_vec[i], label = "Prior $i")
end
if save_data
    savefig("priors_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).png")
end

# now plot histograms along with priors for each iterations to see how much the priors are capturing the histograms
for i in 1:iter_count-2
    hist_plot = histogram(get(q.omega, i), xlabel = "ω", ylabel = "Density", title = "Parameter Distribution with Prior, Iteration $i", dpi = 250, label = "Samples", color = colors[2], alpha = 0.7, normalize = true)
    plot!(xs, prior_vec[i + 1], label = "Prior", color = colors[3], lw = 3)
    # vline!([ω], label = "True ω", color = :pink, alpha = 1.0, lw = 3)
    if save_data
        savefig("hist_prior_overlay_iter_$(i)_dist_func_$(dist_func)_bandwidth_$(bandwidth)_power_$(downweight_power).png")
    end
end
