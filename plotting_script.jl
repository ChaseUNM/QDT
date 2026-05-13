using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, OrderedCollections
include("src/digital_qudit.jl")
include("src/physical_device.jl")
include("src/util.jl")


# Device 
Ne = 2
Ng = 0
N = Ne + Ng
n_samples = 200


# Controls
degree = 2
n_splines = 8
n_iters_opt = 100


#Create gate set
gates = [PauliX, PauliY, PauliZ, Hadamard]
N_gates = length(gates)
T_gate = 50

# Physical Qudit
M_spam_order = 1e-4
n_readout_samples = 100000
# n_pts = 1001
true_omega = 4.5
true_xi = 0.0
omega_rot = 4.5
phys_q = PhysicalQudit(
                    Ne, Ng, 
                    true_omega, omega_rot, true_xi,
                    M_spam_order=M_spam_order
                )

epsilon = 1E-4

experiment = 4
# Load the data if necessary 
data = load("bias_stdev_data_$experiment.jld2")
q_meas_infidelities_state = data["q_meas_infidelities_state"]
q_meas_infidelities_pop = data["q_meas_infidelities_pop"]
q_pred_infidelities = data["q_pred_infidelities"]
control_history = data["control_history"]
large_predicted_infidelity_flag = data["large_predicted_infidelity_flag"]

q_data = load("qudit_$experiment.jld2")
q = q_data["q"]


# Get keys of dictionary as vector of tuples 
keys_vec = collect(keys(q_meas_infidelities_pop))
total_keys = length(keys_vec)


# Now to plot the data
# Plot each pair of bias and standard deviation with the predicted and measured infidelities for each gate type as a scatter plot 

# Plot the first pair of bias and standard deviation with the predicted and measured infidelities for each gate type as a scatter plot

#Only choose subset of keys to plot, include first, last and get a total of 9
if total_keys <= 9 
    m = total_keys - 2
else
    m = 7
end

key_plot = round.(Int, range(1, total_keys, length = m + 2))


jitter = 0.4/total_keys

#plot the standard deviation and bias as characterization occurs

#get first element of the tuple from keys vec
std_dev_arr = [keys_vec[i][1] for i in 1:total_keys]
bias_arr = [keys_vec[i][2] for i in 1:total_keys]

colors = palette(:viridis, total_keys)
# x = collect(1:total_keys)
colors = palette(:viridis, total_keys)
x = 1:total_keys

param_scatter = scatter(x, [std_dev_arr bias_arr],
    marker = [:circle :star],
    markersize = 6,
    alpha = 0.9,
    marker_z = 1:length(x),
    color = cgrad(colors, categorical = true),   
    colorbar = false, # <-- per-point coloring
    labels = ["Standard Deviation" "Bias"], xticks = [2*i - 1 for i in 1:20],
    xlabel = "Recharacterization iteration", yscale = :log10, yticks = [10^-1, 10^-2, minimum(bias_arr), minimum(std_dev_arr)], dpi = 250
)
savefig("param_scatter_$experiment.png")


gate_to_str(g) = g == PauliX ? "X" :
                 g == PauliY ? "Y" :
                 g == PauliZ ? "Z" :
                 g == Hadamard ? "H" :
                 string(g)

# meas_infidelity_scatter = scatter()  
for idx in key_plot
    gates = collect(keys(q_meas_infidelities_pop[keys_vec[idx]]))
    x = (1:N_gates) .+ (idx - (length(key_plot)+1)/2) * jitter
    meas_vals = collect(values(q_meas_infidelities_pop[keys_vec[idx]]))
    pred_vals = collect(values(q_pred_infidelities[keys_vec[idx]]))
    xtick_labels = gate_to_str.(gates)
    p_str = "p"*int_to_subscript(idx)
    if idx == 1
        meas_infidelity_scatter = scatter(x, meas_vals, 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Measured Infidelity",
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


# grid! isn't defined in this scope; enable the grid via plot! on the existing plot.
# Use a try/catch in case the scatter object isn't available.
try
    plot!(meas_infidelity_scatter, grid = true)
catch
    plot!(grid = true)
end

savefig("meas_infidelity_scatter_$experiment.png")
# pred_infidelity_scatter = scatter()
for idx in key_plot
    gates = collect(keys(q_meas_infidelities_pop[keys_vec[idx]]))
    x = (1:N_gates) .+ (idx - (length(key_plot)+1)/2) * jitter
    meas_vals = collect(values(q_meas_infidelities_pop[keys_vec[idx]]))
    pred_vals = collect(values(q_pred_infidelities[keys_vec[idx]]))
    xtick_labels = gate_to_str.(gates)
    p_str = "p"*int_to_subscript(idx)
    if idx == 1
        pred_infidelity_scatter = scatter(x, pred_vals, 
            yscale = :log10,
            xlabel = "Gate",
            ylabel = "Predicted Infidelity",
            xticks = (1:N_gates, xtick_labels),
            yticks = 10.0 .^ (-7:0),
            dpi = 300,
            marker = :star,
            markersize = 6,
            alpha = 0.8,
            color = colors[idx],
            label = p_str,
            legend = :outertop,
            legend_background_color = :transparent,
            legendfontsize = 8,
            legendcolumns = 4,
        )
        # predicted values for the same key (different marker/color)
    else
        scatter!(x, pred_vals,
            marker = :star,
            markersize = 6,
            alpha = 0.8,
            color = colors[idx],
            label = p_str,
        )
    end
end

hline!([epsilon], label = "Tolerance", color = :black, linestyle = :dash, alpha = 0.8)

# grid! isn't defined in this scope; enable the grid via plot! on the existing plot.
# Use a try/catch in case the scatter object isn't available.
try
    plot!(pred_infidelity_scatter, grid = true)
catch
    plot!(grid = true)
end
savefig("pred_infidelity_scatter_$experiment.png")
#Let's extract the histograms of each sampled parameter 
hist_plotting_indices = [1,Int(floor(total_keys/2)),total_keys]
histogram_plot = histogram(get(q.omega, hist_plotting_indices[1]), xlabel = "ω", ylabel = "counts", dpi = 250, label = "p₁", legend_background_color=:transparent, legend_columns = 4, legend=:outertop, color = colors[hist_plotting_indices[1]], alpha = 0.9,linecolor = :transparent, linewidth = 0)





for i in collect(2:length(hist_plotting_indices))
    # println(typeof(q.omega))
    # println("i: $i")
    # println("plotting_indices[i]: $(plotting_indices[i])")
    param_sample_data = get(q.omega, hist_plotting_indices[i])
    # println(param_sample_data)
    label_str = "p" * int_to_subscript(hist_plotting_indices[i])
    # println(label_str)
    histogram!(param_sample_data, xlabel = "ω", ylabel = "counts", dpi = 250, label = label_str, alpha = 0.9, fillcolor = colors[hist_plotting_indices[i]],linecolor = :transparent, linewidth = 0)
end
vline!([true_omega], label = "true ω₁", lw = 2, color =:black, linestyle =:dash)
savefig("histogram_$experiment.png")
