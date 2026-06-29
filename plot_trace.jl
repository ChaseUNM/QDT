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
downweight_power = 0.7
λ_adaptive = true
λ = 0.1

tag = 6
degree_init = 0
characterization_iter = 0

# get path used to store characterization data for this set of parameters, and load the characterization data

folder = λ_adaptive ? "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_adaptive_$tag" : "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_$(λ)_$tag"
# get w2_chain data
w2_chain_data = load(joinpath(folder, "data", "chain_data", "w2_chain_$(characterization_iter).jld2"))["single_stored_object"]
event_obs = load(joinpath(folder, "data", "event_obs", "event_obs_iteration_$(characterization_iter).jld2"))["single_stored_object"]
pcof_optimal_total = load(joinpath(folder, "pcof_optimal", "pcof_optimal_total_$(characterization_iter).jld2"))["single_stored_object"]

# get trace plot of eta, Sigma, λ



λ_hyper = w2_chain_data.hyperparam_history_λ_log
λ_samples = w2_chain_data.chain[:,2]
window_l = 1
window_r = 150
η_trace = λ_hyper[1,:,3]
iterations = length(η_trace)
η_trace_plot = plot(collect(window_l:window_r), η_trace[window_l:window_r], xlabel = "Iteration", ylabel = "η")
Σ_trace = λ_hyper[1,:,2]
Σ_trace_plot = plot(collect(window_l:window_r), Σ_trace[window_l:window_r], xlabel = "Iteration", ylabel = "Σ")
λ_trace_plot = plot(collect(window_l:window_r), (λ_samples[window_l:window_r]), xlabel = "Iteration", ylabel = "λ")
ω_trace_plot = plot(collect(window_l:window_r), w2_chain_data.chain[:,1][window_l:window_r], xlabel = "Iterations", ylabel = "ω")

λ_trace_plots = plot(λ_trace_plot, η_trace_plot, Σ_trace_plot, layout = (3,1), dpi = 250)

ω_hyper = w2_chain_data.hyperparam_history_ω
ηω_trace = ω_hyper[1,:,3]
Σω_trace = ω_hyper[1,:,2]
ηω_trace_plot = plot(collect(window_l:window_r), ηω_trace[window_l:window_r], xlabel = "Iteration", ylabel = "η")
Σω_trace_plot = plot(collect(window_l:window_r), (Σω_trace[window_l:window_r]), xlabel = "Iteration", ylabel = "Σ", yscale=:log10)
ω_trace_plots = plot(ω_trace_plot, ηω_trace_plot, Σω_trace_plot, layout = (3,1), dpi = 250)

