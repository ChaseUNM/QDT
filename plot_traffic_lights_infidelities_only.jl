using JLD2, Plots, Printf
include("src/QDT_src.jl")

#= ========================================================================
    LOAD DATA
======================================================================== =#

qubits_tag = 478
qubit_tag  = 113

qubits_folder = "results/characterization_2qubits_$qubits_tag"
qubit_folder  = "results/characterization_1qubits_$qubit_tag"


#= ========================================================================
    TRAFFIC LIGHT PLOTTING FUNCTION
======================================================================== =#

plot_settings = Dict(
    :size => (550,600),
    :framestyle => :box,
    :yscale => :log10,
    :xlabel => "Gate",
    :ylabel => "Infidelity",
    :titlefontsize => 25,
    :xguidefontsize => 22, 
    :yguidefontsize => 22, 
    :xtickfontsize => 20, 
    :grid => :y,
    :ytickfontsize => 16, 
    :dpi => 300,
    :legendfontsize => 19,
    :legendcolumns => 2,
    :fg_legend => false,
    :bg_legend => false
)

scatter_settings = Dict(
    :markersize => 8,
    :alpha => 0.8,
)

function plot_light_logic(
            folder::String, 
            red_light_inds::Vector{Int}, 
            yellow_light_inds::Vector{Int}, 
            green_light_inds::Vector{Int};
            log10_ymin::Int=-6    
    )

    # Load data from file
    predicted_infidelity, measured_infidelity = load(
        joinpath(folder, "data", "infidelity_data.jld2"),
        "q_pred_infidelity_total", "q_p_meas_infidelity_total"
    )
    gates = collect(keys(measured_infidelity[1]))

    if !isdir(joinpath(folder, "figures"))
        mkpath(joinpath(folder, "figures"))
    end

    epsilon = 1E-4
    N_gates = length(gates)
    @printf("N_gates = %d\n", N_gates)

    x_sep = 0.4
    jitter = 0.08
    x_vals = x_sep * collect(1:N_gates)
    
    plot_inds   = [red_light_inds, yellow_light_inds, green_light_inds]
    plot_colors = [:red, :yellow, :green]
    plot_names  = ["red", "yellow", "green"]
    legend_locs = [:outertop, :outertop, :outertop]

    # Loop over light color r/y/g
    for t = 1:3
        # Loop over indices with this light color
        for idx = plot_inds[t]
            
            # Initialize the plot with the desired settings
            f = plot(
                    title="Learning Cycle $(idx)",
                    xlims=(x_vals[1]-x_sep/2,x_vals[end]+x_sep/2),
                    xticks = (x_vals, gate_to_str.(gates)),
                    legend = legend_locs[t],
                    yticks = 10.0 .^ (log10_ymin:0),
                    ylims = (10.0 ^ log10_ymin, 2*10^0);
                    plot_settings...
                )
            
            # Plot measured infidelities
            scatter!(
                x_vals .- jitter/2, 
                collect(values(measured_infidelity[idx])),
                marker = :circle, color=plot_colors[t], label=" Measured ";
                scatter_settings ...
            )
            
            # Plot predicted infidelities
            scatter!(
                x_vals .+ jitter/2, 
                collect(values(predicted_infidelity[idx])),
                marker = :square, color=plot_colors[t], label=" Predicted"; 
                scatter_settings ...
            )

            # Plot target infidelity
            hline!([epsilon], label = "", color=:black, linestyle=:dash)

            # Lines separating each gate
            vline!(
                x_vals[1:end-1] .+ x_sep/2, 
                label="", color=:black, alpha=0.15
            )
            
            # Save data to file
            light_type = plot_names[t]
            figname = "$(light_type)_light_infidelity_idx_$(idx).pdf"
            fullpath = joinpath(folder, "figures", figname)
            @printf("Created figure %s\n", fullpath)
            savefig(f, fullpath)
        end
    end

    return gates
end



#= ========================================================================
    RUN PLOTTING FUNCTION
======================================================================== =#

g1 = plot_light_logic(qubit_folder, [1],[2],[3])
g2 = plot_light_logic(qubits_folder, [1], [3], [6], log10_ymin=-5)
