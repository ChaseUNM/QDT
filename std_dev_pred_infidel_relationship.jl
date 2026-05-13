using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
include("src/qdt.jl")
include("src/measurement.jl")

# Device
N = 1 
Ne = [2]
Ng = [0]
ω_true = 4.5
ω_rot = ω_true
Mspam_order = 5E-3

# Gates
x_gate = [0 1.0; 1.0 0]
y_gate = [0 im; -im 0]
z_gate = [1.0 0; 0 -1.0]
gate_set = [x_gate, y_gate, z_gate]
N_gates = length(gate_set)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 3
n_iters_opt = 100

# Device "characterization"
n_ω_samples = 200
ω_stdev_list = 10.0 .^ -collect(1:6)
nsplines_list = [3, 6, 9, 12, 15, 18]

pred_infidelity_mat = zeros(N_gates, length(ω_stdev_list), length(nsplines_list))

#Loop through each stdev in stdev_list and each splines
for i in 1: length(ω_stdev_list) 
    # Make the QDT
    println("Standard Deviation: ", ω_stdev_list[i])
    for j in 1:length(nsplines_list)
        Twin = QDT( N, Ne, Ng, ω_rot, 
                gate_set, Mspam_order, 
                n_param_samples=n_ω_samples,
                T_gate=T_gate, 
                control_degree=degree, control_nsplines=nsplines_list[j])
        println("# of Splines: ", Twin.nsplines)
        # Generate parameter samples
        ω_samples = Normal(ω_true, ω_stdev_list[i])
        Twin.param_samples = rand(ω_samples, Twin.n_param_samples) 

        # Optimize the control for this sample set
        ipopt_options = ["max_iter" => n_iters_opt, "print_level" => 3]
        optimize_controls(Twin, ipopt_options=ipopt_options)
        pred_infidelity_mat[:,i,j] = Twin.predicted_infidelity
    end
end

pred_infidelity_mat = abs.(pred_infidelity_mat)


# plot_x = plot(ω_stdev_list, pred_infidelity_mat[1,:,1], label = "Splines: $(nsplines_list[1])", yscale=:log10, ylabel = "Infidelity", xlabel = "Standard Deviation", xscale=:log10, legend=:topleft)
# plot!(ω_stdev_list, pred_infidelity_mat[1,:,2], label = "Splines: $(nsplines_list[2])")
# plot!(ω_stdev_list, pred_infidelity_mat[1,:,3], label = "Splines: $(nsplines_list[3])")
# plot!(ω_stdev_list, pred_infidelity_mat[1,:,4], label = "Splines: $(nsplines_list[4])")
# plot!(ω_stdev_list, pred_infidelity_mat[1,:,5], label = "Splines: $(nsplines_list[5])")
# plot!(ω_stdev_list, pred_infidelity_mat[1,:,6], label = "Splines: $(nsplines_list[6])")

plot_x = plot(ω_stdev_list, pred_infidelity_mat[1,:,:], label = ["$(nsplines_list[1]) splines" "$(nsplines_list[2]) splines" "$(nsplines_list[3]) splines" "$(nsplines_list[4]) splines" "$(nsplines_list[5]) splines" "$(nsplines_list[6]) splines"], xscale=:log10, yscale=:log10, legend =:outertop, legend_columns = 3, ylabel = "Infidelity", xlabel = "Standard Deviation", title = "X Gate Predicted Infidelities", xticks = ω_stdev_list, yticks = [10E-6, 10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 1], xlims = (10^-6.1, 10^-0.9), ylims = (10^-6, 1), markersize = 1.5, marker=:circle, dpi = 250)

plot_y = plot(ω_stdev_list, pred_infidelity_mat[2,:,:], label = ["$(nsplines_list[1]) splines" "$(nsplines_list[2]) splines" "$(nsplines_list[3]) splines" "$(nsplines_list[4]) splines" "$(nsplines_list[5]) splines" "$(nsplines_list[6]) splines"], xscale=:log10, yscale=:log10, legend =:outertop, legend_columns = 3, ylabel = "Infidelity", xlabel = "Standard Deviation", title = "Y Gate Predicted Infidelities", xticks = ω_stdev_list, yticks = [10E-6, 10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 1], xlims = (10^-6.1, 10^-0.9), ylims = (10^-6, 1), markersize = 1.5, marker=:circle, dpi = 250)

plot_z = plot(ω_stdev_list, pred_infidelity_mat[3,:,:], label = ["$(nsplines_list[1]) splines" "$(nsplines_list[2]) splines" "$(nsplines_list[3]) splines" "$(nsplines_list[4]) splines" "$(nsplines_list[5]) splines" "$(nsplines_list[6]) splines"], xscale=:log10, yscale=:log10, legend =:outertop, legend_columns = 3, ylabel = "Infidelity", xlabel = "Standard Deviation", title = "Z Gate Predicted Infidelities", xticks = ω_stdev_list, yticks = [10E-6, 10E-5, 10E-4, 10E-3, 10E-2, 10E-1, 1], xlims = (10^-6.1, 10^-0.9), ylims = (10^-6, 1), markersize = 1.5, marker=:circle, dpi = 250)