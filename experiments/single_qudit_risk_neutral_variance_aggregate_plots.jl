using LinearAlgebra, Printf, Random, Distributions
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
using JLD2

###################################################################
# PARAMETERS
###################################################################

Ne      = 2
Ng      = 2
omega   = 4.5
xi      = 0.2

gate_names = ["X", "Y", "Z"]
n_gates = length(gate_names)


##############################################################
# LOAD DATA FROM FILE
##############################################################

data_dir = "./data/single_qudit_risk_neutral_variance"

n_samples = []
sample_stdev = []
infidelity_mean  = Array{Float64}(undef, n_gates, 0)
infidelity_stdev = Array{Float64}(undef, n_gates, 0)

for filename in readdir(data_dir)

    full_path = joinpath(data_dir, filename)
    
    # Skip subdirectories
    if isdir(full_path)
        continue
    end

    # Load the data from file
    local data = load(full_path)

    # Save data needed for the plots
    infidelities = data["infidelity_samples"]
    n_trials = size(infidelities, 1)
    for t = 1:n_trials
        push!(n_samples, data["n_samples"])
        push!(sample_stdev, data["omega_stdev"])
        imean = zeros(n_gates)
        istdev = zeros(n_gates)
        for i = 1:n_gates
            imean[i]  = mean(infidelities[t,i,:])
            istdev[i] = std(infidelities[t,i,:])
        end
        global infidelity_mean  = [infidelity_mean imean]
        global infidelity_stdev = [infidelity_stdev istdev]
    end
end


##############################################################
# PLOT MEAN INFIDELITY VS OMEGA STDEV
###############################################################

# subplots = Any[]
# for i = 1:n_gates
#     title_str = @sprintf(
#                     "Infidelity vs. \$\\omega\$ StDev, \$%s\$ Gate", 
#                     gate_names[i]
#                 )
#     p_i = plot(
#             title= LaTeXString(title_str), 
#             titlefontsize=16, 
#             xlabel=L"StDev in $\omega$", xlabelfontsize=14,
#             ylabel="Digital Gate Infidelity", ylabelfontsize=14, 
#             size=(600,500), dpi=512, 
#             xscale=:log10, yscale=:log10,
#             left_margin = 10mm, 
#             right_margin = 10mm
#         )
    
    
#     scatter!(
#         sample_stdev, infidelity_mean[i,:], 
#         group=n_samples, alpha=0.5, 
#         legend=false
#     )

#     push!(subplots, p_i)

# end

# f = plot(
#     subplots[1], subplots[2], subplots[3],
#     layout=(1,3), size=(1500,500), dpi=512, 
#     bottom_margin=10mm, top_margin=10mm
# )


##############################################################
# PLOT MEAN INFIDELITY VS # SAMPLES
###############################################################

# unique_n_samples    = unique(n_samples)
# unique_sample_stdev = unique(sample_stdev)

# subplots = Any[]
# for i = 1:n_gates
#     title_str = @sprintf(
#                     "\$%s\$ Gate", 
#                     gate_names[i]
#                 )
#     p_i = plot(
#             title= LaTeXString(title_str), 
#             titlefontsize=16, 
#             xlabel="Number of Parameter Samples", xlabelfontsize=14,
#             ylabel="Digital Gate Infidelity", ylabelfontsize=14, 
#             size=(600,500), 
#             xticks=(unique_n_samples,unique_n_samples),
#             xscale=:log10, yscale=:log10,
#             dpi=512, 
#             left_margin = 8mm, 
#             right_margin = 8mm
#         )
    
    
#     scatter!(
#         n_samples, infidelity_mean[i,:], 
#         group=sample_stdev, alpha=0.5, 
#         legend=false
#     )

#     push!(subplots, p_i)

# end

# f = plot(
#     subplots[1], subplots[2], subplots[3],
#     layout=(1,3), size=(1500,500), dpi=512, 
#     bottom_margin=10mm, top_margin=10mm
# )


##############################################################
# PLOT MEAN INFIDELITY VS # SAMPLES
# (for a single gate)
###############################################################

unique_n_samples    = unique(n_samples)
unique_sample_stdev = sort(unique(sample_stdev), rev=true)

for i = 1:n_gates

    title_str = @sprintf(
                    "\$%s\$ Gate", 
                    gate_names[i]
                )
    f = plot(
            title= LaTeXString(title_str), 
            titlefontsize=20,
            tickfontsize=12, 
            xlabel="Number of Parameter Samples", xlabelfontsize=14,
            ylabel="Digital Gate Infidelity", ylabelfontsize=14, 
            size=(1000,500), 
            xticks=(unique_n_samples,unique_n_samples),
            xscale=:log10, yscale=:log10,
            dpi=512, 
            legend=:outerright, legendfontsize=12,
            left_margin = 8mm, 
            top_margin = 8mm, 
            bottom_margin = 8mm,
            right_margin = 8mm
        )


    # my_cg = cgrad(
    #             :viridis, 
    #             length(unique_sample_stdev), 
    #             categorical=true
    #         )
    J = length(unique_sample_stdev)
    for j = 1:J
        local idx = (sample_stdev .== unique_sample_stdev[j])
        label=@sprintf("\$\\sigma_\\omega =\$%.2e", unique_sample_stdev[j])
        scatter!(
            f,
            n_samples[idx], infidelity_mean[i,idx], 
            markersize=8, alpha=0.5,
            label=LaTeXString(label)
        )
    end

    fig_dir = "./figures/single_qudit_risk_neutral_variance"
    fname = @sprintf("%s_gate_agg.svg",gate_names[i]);
    savefig(f, joinpath(fig_dir, fname))

end
