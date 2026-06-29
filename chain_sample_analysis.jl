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
downweight_power = 0.3
λ_adaptive = true
tag = 3
λ = 0.1

degree_init = 2

folder = λ_adaptive ? "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_adaptive_$tag" : "results/characterization_control_GaussianFit_Seeded_power_$(downweight_power)_samples_$(samples)_degree_$(degree_init)_λ_$(λ)_$tag"


plot_chain = true
hist_plot_number = 1
for i in 0:10
    if isfile(joinpath(folder, "data", "chain_data", "w2_chain_$i.jld2"))
        w2_chain_data = load(joinpath(folder, "data", "chain_data", "w2_chain_$i.jld2"))["single_stored_object"]

        println("Rhat: ", rhat(w2_chain_data.diagnostic_chain))
    else
        println("folder doesn't exist")
    end

end

if plot_chain

    w2_chain_data = load(joinpath(folder, "data", "chain_data", "w2_chain_$(hist_plot_number).jld2"))["single_stored_object"]

    diagnostic_chain_histograms = histogram()
    for i in 1:size(w2_chain_data.diagnostic_chain)[2]
        histogram!(w2_chain_data.diagnostic_chain[:,i], alpha = 0.5, label = "Chain $i")
    end
end