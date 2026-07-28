using JLD2, Plots

include("src/QDT_src.jl")

qubits_tag = 453
qubit_tag = 113

qubits_folder = "results/characterization_2qubits_$qubits_tag"
qubit_folder = "results/characterization_1qubits_$qubit_tag"


function plot_light_logic_1qubit(folder::String, red_light_inds::Vector{Int}, yellow_light_inds::Vector{Int}, green_light_inds::Vector{Int})

    @load joinpath(folder, "data", "infidelity_data.jld2") q_pred_infidelity_total q_p_meas_infidelity_total mean_list variance_list rand_seed_list
    @load joinpath(folder, "data", "characterization_params.jld2") true_params param_init_list MHG_params mean_list variance_list downweight_power rand_seed_list 
    @load joinpath(folder, "data", "control_params.jld2") gates control_params_init control_params_total control_dict_total
    @load joinpath(folder, "data", "qubit_data.jld2") q_history phys_q

    if !isdir(joinpath(folder, "figures"))
        mkpath(joinpath(folder, "figures"))
    end

    epsilon = 1E-4
    N_gates = length(gates)
    true_params_GHz = true_params./2pi

    jitter = 0.05
    x_vals = collect(1:N_gates)
    x_vals_jitter = x_vals .+ jitter
    # plot red light scatter plot with measured infidelity values and predicted infidelities for the red light round
    for red_light_idx in red_light_inds
        red_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[red_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $red_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12, 
            xtickfontsize = 11, 
            ytickfontsize = 11, 
            xticks = (1:N_gates, gate_to_str.(collect(keys(q_p_meas_infidelity_total[red_light_idx])))),
            yticks = 10.0 .^ (-7:0),
            ylims = (10^-6, 10^0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :red,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[red_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :red,
            label = "Predicted",
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "red_light_infidelity_idx_$(red_light_idx).png"))
    end

    # plot yellow light scatter plot with measured infidelity values and predicted infidelities for the yellow light round
    for yellow_light_idx in yellow_light_inds
        yellow_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[yellow_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $yellow_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12,
            xtickfontsize = 11, 
            ytickfontsize = 11,
            xticks = (x_vals, gate_to_str.(collect(keys(q_p_meas_infidelity_total[yellow_light_idx])))),
            yticks = 10.0 .^(-7:0),
            ylims = (10^-6, 10^0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :yellow,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[yellow_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :yellow,
            label = "Predicted"
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "yellow_light_infidelity_idx_$(yellow_light_idx).png"))
    end

    # plot green light scatter plot with measured infidelity values and predicted infidelities for the green light round
    for green_light_idx in green_light_inds

        green_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[green_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $green_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12, 
            xtickfontsize = 11, 
            ytickfontsize = 11, 
            xticks = (1:N_gates, gate_to_str.(collect(keys(q_p_meas_infidelity_total[green_light_idx])))),
            yticks = 10.0 .^(-7:0),
            ylims = (10^-6, 10^0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :green,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[green_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :green,
            label = "Predicted",
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "green_light_infidelity_idx_$(green_light_idx).png"))
    end

    # now plot the density of the posterior distribution for each light, plot the yellow and green lights together and the red separately since the red light is so far away from the other two

    red_colors = [:red, :firebrick, :indianred, :salmon]

    yellow_colors = [:goldenrod, :darkgoldenrod, :darkorange, :khaki]
    green_colors = [:green, :forestgreen, :limegreen, :seagreen]

    red_light_posterior_ω1 = nothing

    for (i, red_light_idx) in enumerate(red_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$red_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1] ./ 2pi

        if red_light_posterior_ω1 === nothing
            red_light_posterior_ω1 = density(
                samples,
                # xticks = [4.499, 4.5, 4.501],
                xformatter = :plain,
                xlabel = "ω₁ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ω₁ for red light",
                label = "Red cycle $red_light_idx",
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                linewidth = 3,
                color = red_colors[i],
                dpi = 250
            )
        else
            density!(
                red_light_posterior_ω1,
                samples,
                label = "Red cycle $red_light_idx",
                color = red_colors[i],
                linewidth = 3
            )
        end
    end

    vline!(
        red_light_posterior_ω1,
        [true_params_GHz[1]],
        label = "True ω₁",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        red_light_posterior_ω1,
        joinpath(folder, "figures", "red_light_posterior_ω1.png")
    )


    yellow_green_posteriors_ω1 = nothing

    # Plot yellow curves
    for (i, yellow_light_idx) in enumerate(yellow_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$yellow_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1] ./ 2pi

        if yellow_green_posteriors_ω1 === nothing
            yellow_green_posteriors_ω1 = density(
                samples,
                xticks = [4.5996, 4.5998, 4.6, 4.6002],
                xformatter = :plain,
                xlabel = "ω₁ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ω₁",
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3,
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                dpi = 250
            )
        else
            density!(
                yellow_green_posteriors_ω1,
                samples,
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3
            )
        end
    end


    # Plot green curves
    for (i, green_light_idx) in enumerate(green_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$green_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1] ./ 2pi

        density!(
            yellow_green_posteriors_ω1,
            samples,
            label = "Green cycle $green_light_idx",
            color = green_colors[i],
            linewidth = 3
        )
    end


    # True value
    vline!(
        yellow_green_posteriors_ω1,
        [true_params_GHz[1]],
        label = "True ω₁",
        color = :black,
        linestyle = :dash,
        lw = 4
    )

    savefig(
        yellow_green_posteriors_ω1,
        joinpath(folder, "figures", "yellow_green_posteriors_ω1.png")
    )
end

function plot_light_logic_2qubits(folder::String, red_light_inds::Vector{Int}, yellow_light_inds::Vector{Int}, green_light_inds::Vector{Int})

    @load joinpath(folder, "data", "infidelity_data.jld2") q_pred_infidelity_total q_p_meas_infidelity_total mean_list covariance_list rand_seed_list
    @load joinpath(folder, "data", "characterization_params.jld2") true_params param_init_list MHG_params mean_list covariance_list downweight_power rand_seed_list 
    @load joinpath(folder, "data", "control_params.jld2") control_dict_total gates control_params_init control_params_total carrier_freqs

    epsilon = 1E-4
    N_gates = length(gates)
    true_params_GHz = true_params./2pi

    if !isdir(joinpath(folder, "figures"))
        mkpath(joinpath(folder, "figures"))
    end
    jitter = 0.05
    x_vals = collect(1:N_gates)
    x_vals_jitter = x_vals .+ jitter
    # plot red light scatter plot with measured infidelity values and predicted infidelities for the red light round
    for red_light_idx in red_light_inds
        red_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[red_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $red_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12, 
            xtickfontsize = 11, 
            ytickfontsize = 11, 
            xticks = (1:N_gates, gate_to_str.(collect(keys(q_p_meas_infidelity_total[red_light_idx])))),
            yticks = 10.0 .^ (-7:0),
            ylims = (10^-5, 10^-0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :red,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[red_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :red,
            label = "Predicted",
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "red_light_infidelity_idx_$(red_light_idx).png"))
    end

    # plot yellow light scatter plot with measured infidelity values and predicted infidelities for the yellow light round
    for yellow_light_idx in yellow_light_inds
        yellow_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[yellow_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $yellow_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12,
            xtickfontsize = 11, 
            ytickfontsize = 11,
            xticks = (x_vals, gate_to_str.(collect(keys(q_p_meas_infidelity_total[yellow_light_idx])))),
            ylims = (10^-5, 10^-0),
            yticks = 10.0 .^(-7:0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :yellow,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[yellow_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :yellow,
            label = "Predicted"
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "yellow_light_infidelity_idx_$(yellow_light_idx).png"))
    end

    # plot green light scatter plot with measured infidelity values and predicted infidelities for the green light round
    for green_light_idx in green_light_inds

        green_light_scatter = scatter(x_vals, collect(values(q_p_meas_infidelity_total[green_light_idx])), 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Infidelity",
            title = "Measured Infidelity for gate set at QDT learning cycle $green_light_idx",
            titlefontsize = 13,
            xguidefontsize = 12, 
            yguidefontsize = 12, 
            xtickfontsize = 11, 
            ytickfontsize = 11, 
            xticks = (1:N_gates, gate_to_str.(collect(keys(q_p_meas_infidelity_total[green_light_idx])))),
            ylims = (10^-5, 10^-0),
            yticks = 10.0 .^(-7:0),
            dpi = 300,
            marker = :circle,
            markersize = 6,
            alpha = 0.9,
            color = :green,
            label = "Measured",
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        scatter!(x_vals_jitter, collect(values(q_pred_infidelity_total[green_light_idx])),
            marker = :square,
            markersize = 6,
            alpha = 0.9,
            color = :green,
            label = "Predicted",
        )
        hline!([epsilon], label = "Tolerance", color=:black, linestyle=:dash)
        savefig(joinpath(folder, "figures", "green_light_infidelity_idx_$(green_light_idx).png"))
    end

    # now plot the density of the posterior distribution for each light, plot the yellow and green lights together and the red separately since the red light is so far away from the other two

    red_colors = [:red, :firebrick, :indianred, :salmon]

    yellow_colors = [:goldenrod, :darkgoldenrod, :darkorange, :khaki]
    green_colors = [:green, :forestgreen, :limegreen, :seagreen]

    red_light_posterior_ω1 = nothing

    for (i, red_light_idx) in enumerate(red_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$red_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,1] ./ 2pi

        if red_light_posterior_ω1 === nothing
            red_light_posterior_ω1 = density(
                samples,
                # xticks = [4.499, 4.5, 4.501],
                xformatter = :plain,
                xlabel = "ω₁ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ω₁ for red light",
                label = "Red cycle $red_light_idx",
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                linewidth = 3,
                color = red_colors[i],
                dpi = 250
            )
        else
            density!(
                red_light_posterior_ω1,
                samples,
                label = "Red cycle $red_light_idx",
                color = red_colors[i],
                linewidth = 3
            )
        end
    end

    vline!(
        red_light_posterior_ω1,
        [true_params_GHz[1]],
        label = "True ω₁",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        red_light_posterior_ω1,
        joinpath(folder, "figures", "red_light_posterior_ω1.png")
    )


    yellow_green_posteriors_ω1 = nothing

    # Plot yellow curves
    for (i, yellow_light_idx) in enumerate(yellow_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$yellow_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,1] ./ 2pi

        if yellow_green_posteriors_ω1 === nothing
            yellow_green_posteriors_ω1 = density(
                samples,
                xticks = [4.4998, 4.5, 4.5002],
                xformatter = :plain,
                xlabel = "ω₁ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ω₁",
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3,
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                dpi = 250
            )
        else
            density!(
                yellow_green_posteriors_ω1,
                samples,
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3
            )
        end
    end


    # Plot green curves
    for (i, green_light_idx) in enumerate(green_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$green_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,1] ./ 2pi

        density!(
            yellow_green_posteriors_ω1,
            samples,
            label = "Green cycle $green_light_idx",
            color = green_colors[i],
            linewidth = 3
        )
    end


    # True value
    vline!(
        yellow_green_posteriors_ω1,
        [true_params_GHz[1]],
        label = "True ω₁",
        color = :black,
        linestyle = :dash,
        lw = 4
    )

    savefig(
        yellow_green_posteriors_ω1,
        joinpath(folder, "figures", "yellow_green_posteriors_ω1.png")
    )


    red_light_posterior_ω2 = nothing

    for (i, red_light_idx) in enumerate(red_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$red_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,2] ./ 2pi

        if red_light_posterior_ω2 === nothing
            red_light_posterior_ω2 = density(
                samples,
                # xticks = range(minimum(samples), maximum(samples), length=3),
                xlabel = "ω₂ (GHz)",
                xformatter = :plain,
                ylabel = "Density",
                title = "Posterior density of ω₂ for red light",
                label = "Red cycle $red_light_idx",
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                linewidth = 3,
                color = red_colors[i],
                dpi = 250
            )
        else
            density!(
                red_light_posterior_ω2,
                samples,
                label = "Red cycle $red_light_idx",
                color = red_colors[i],
                linewidth = 3
            )
        end
    end

    vline!(
        red_light_posterior_ω2,
        [true_params_GHz[2]],
        label = "True ω₂",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        red_light_posterior_ω2,
        joinpath(folder, "figures", "red_light_posterior_ω2.png")
    )

    yellow_green_posteriors_ω2 = nothing

    # Plot yellow curves
    for (i, yellow_light_idx) in enumerate(yellow_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$yellow_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,2] ./ 2pi

        if yellow_green_posteriors_ω2 === nothing
            yellow_green_posteriors_ω2 = density(
                samples,
                xticks = [4.59995, 4.6, 4.60005],
                xformatter = :plain,
                xlabel = "ω₂ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ω₂ for yellow and green lights",
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3,
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                dpi = 250
            )
        else
            density!(
                yellow_green_posteriors_ω2,
                samples,
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3
            )
        end
    end


    # Plot green curves
    for (i, green_light_idx) in enumerate(green_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$green_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,2] ./ 2pi

        density!(
            yellow_green_posteriors_ω2,
            samples,
            label = "Green cycle $green_light_idx",
            color = green_colors[i],
            linewidth = 3
        )
    end


    vline!(
        yellow_green_posteriors_ω2,
        [true_params_GHz[2]],
        label = "True ω₂",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        yellow_green_posteriors_ω2,
        joinpath(folder, "figures", "yellow_green_posteriors_ω2.png")
    )


    red_light_posterior_ξ = nothing

    for (i, red_light_idx) in enumerate(red_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$red_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,3] ./ 2pi

        if red_light_posterior_ξ === nothing
            red_light_posterior_ξ = density(
                samples,
                xticks = [0.00925, 0.01, 0.01075],
                xformatter = :plain,
                xlabel = "ξ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ξ for red light",
                label = "Red cycle $red_light_idx",
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                linewidth = 3,
                color = red_colors[i],
                dpi = 250
            )
        else
            density!(
                red_light_posterior_ξ,
                samples,
                label = "Red cycle $red_light_idx",
                color = red_colors[i],
                linewidth = 3
            )
        end
    end

    vline!(
        red_light_posterior_ξ,
        [true_params_GHz[4]],
        label = "True ξ",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        red_light_posterior_ξ,
        joinpath(folder, "figures", "red_light_posterior_ξ.png")
    )

    yellow_green_posteriors_ξ = nothing

    # Plot yellow curves
    for (i, yellow_light_idx) in enumerate(yellow_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$yellow_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,3] ./ 2pi

        if yellow_green_posteriors_ξ === nothing
            yellow_green_posteriors_ξ = density(
                samples,
                xticks = [0.0099, 0.01, 0.0101],
                xformatter = :plain,
                xlabel = "ξ (GHz)",
                ylabel = "Density",
                title = "Posterior density of ξ for yellow and green lights",
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3,
                titlefontsize = 12,
                xguidefontsize = 11,
                yguidefontsize = 11,
                xtickfontsize = 10,
                ytickfontsize = 10,
                dpi = 250
            )
        else
            density!(
                yellow_green_posteriors_ξ,
                samples,
                label = "Yellow cycle $yellow_light_idx",
                color = yellow_colors[i],
                linewidth = 3
            )
        end
    end


    # Plot green curves
    for (i, green_light_idx) in enumerate(green_light_inds)

        chain = load_object(
            joinpath(folder, "data", "chain_data", "chain_data_$green_light_idx.jld2")
        )

        samples = chain.diagnostic_chain[:,1,3] ./ 2pi

        density!(
            yellow_green_posteriors_ξ,
            samples,
            label = "Green cycle $green_light_idx",
            color = green_colors[i],
            linewidth = 3
        )
    end


    vline!(
        yellow_green_posteriors_ξ,
        [true_params_GHz[4]],
        label = "True ξ",
        color = :black,
        linestyle = :dash,
        alpha = 0.8,
        lw = 4.0
    )

    savefig(
        yellow_green_posteriors_ξ,
        joinpath(folder, "figures", "yellow_green_posteriors_ξ.png")
    )

end

plot_light_logic_2qubits(qubits_folder, [1], [2], [3])
plot_light_logic_1qubit(qubit_folder, [1],[2],[3])