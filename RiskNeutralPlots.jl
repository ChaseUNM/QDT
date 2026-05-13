using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, LaTeXStrings, JLD2
include("src/digital_device.jl")
include("src/physical_device.jl")
include("src/util.jl")

# Device 
Ne = 2
Ng = 0
omega = 4.5
omega_bias = 0.0
omega_stdev = 0.001
n_samples = 200
xi = 0.0
xi_bias = 0.0
xi_stdev = 0.0


# Gates
gates = [PauliX, PauliY, PauliZ, Hadamard]
N_gates = length(gates)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 6
n_iters_opt = 100

# Physical Qudit
M_spam_order = 1e-4
n_readout_samples = 100000

############################################################################# 
# SETUP
#############################################################################





omega_stdev = [0.1, 0.05, 0.005, 0.001, 0.0]
number_deviations = length(omega_stdev)


omega = 4.5
omega_rot = omega

h = 0.04
x_pts = 1001
omega_range = LinRange(omega - h, omega + h, x_pts)
xi_range = LinRange(xi - h, xi + h, x_pts)

meas_pop_infidelities_array = zeros(number_deviations, x_pts, N_gates)
meas_state_infidelities_array = zeros(number_deviations, x_pts, N_gates)

q_list = []

############################################################################# 
# CONTROL OPTIMIZATION
#############################################################################

# meas_pop_infidelities_array = zeros(x_pts, N_gates)
# meas_state_infidelities_array = zeros(x_pts, N_gates)


# # Create a Qudit
# q = DigitalQudit(Ne, Ng)
# N = Ne + Ng
# # Create a control for each gate
# for j = 1:N_gates
#     # Control for this gate
#     qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
#     add_control(q, gates[j], qcontrol)
#     # Infidelities for this gate
#     q.infidelity[gates[j]] = History(Float64)
# end

# # Create parameter sampler
# omega_sampler = Normal(omega + omega_bias, omega_stdev)
# xi_sampler    = Normal(xi + xi_bias, xi_stdev)

# # Initial parameter samples
# omega_samples = sort(rand(omega_sampler, n_samples))
# xi_samples = sort(rand(xi_sampler, n_samples))
# add_param_samples(q, omega_samples, xi_samples)
# # Optimize the controls
# for j = 1:N_gates
#     optimize_control(q, gates[j])
# end

############################################################################# 
# MEASURED FIDELITY
#############################################################################
# Create the "physical device" for each true parameter in omega_range 
# for j in 1:x_pts
#     phys_q = PhysicalQudit(
#                 Ne, Ng, 
#                 omega_range[j], omega, xi_range[j],
#                 M_spam_order=M_spam_order
#             )

#     dt = 0.01
#     # println("Omega value: ", get(phys_q.qudit.omega))
#     measured_state_infidelity = zeros(N_gates)
#     measured_population_infidelity = zeros(N_gates)
#     for k = 1:N_gates
#         s_inf, p_inf = measure_infidelity(
#                             phys_q,
#                             gates[k],
#                             q.controls[gates[k]],
#                             n_readout_samples,
#                             dt=dt
#                     ) 
#         measured_state_infidelity[k] = s_inf
#         measured_population_infidelity[k] = p_inf  
#         # println("This is p_inf: ", p_inf)
#         # println("This is s_inf: ", s_inf) 
#         meas_pop_infidelities_array[j,k] = p_inf 
#         meas_state_infidelities_array[j,k] = s_inf
#     end      
# end

for i in 1:length(omega_stdev)
    # Create a qudit[1]
    # n_samples = omega_stdev[i] == 0.0 ? 1 : samples
    # n_samples = 50
    q = DigitalQudit(Ne, Ng)
    N = Ne + Ng
    # Create a control for each gate
    for j = 1:N_gates
        # Control for this gate
        qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
        add_control(q, gates[j], qcontrol)
        # Infidelities for this gate
        q.infidelity[gates[j]] = History(Float64)
    end

    # Create parameter sampler

    omega_sampler = Normal(omega + omega_bias, omega_stdev[i])
    xi_sampler    = Normal(xi + xi_bias, xi_stdev)

    # Initial parameter samples
    omega_samples = sort(rand(omega_sampler, n_samples))
    if omega_stdev[i] == 0.0
        omega_sampler = Normal(omega + omega_bias, 0.001)
        omega_samples = sort(rand(omega_sampler, n_samples))
        omega_samples = [mean(omega_samples)]
    end
    xi_samples = sort(rand(xi_sampler, n_samples))
    hist = histogram(omega_samples, xlabel = L"\omega", ylabel = "Counts", dpi = 250)
    savefig(hist,"histogram_center_$(omega)_stdev_$(omega_stdev[i])_omega_rot_$(omega_rot)_bias_$(omega_bias)_samples_$(n_samples).png")
    add_param_samples(q, omega_samples, xi_samples)
    # println("Param samples: ", get(q.omega))
    # Optimize the controls
    for j = 1:N_gates
        optimize_control(q, gates[j])
    end
    push!(q_list, q)
    ############################################################################# 
    # MEASURED FIDELITY
    #############################################################################
    # Create the "physical device" for each true parameter in omega_range 
    for j in 1:x_pts

        phys_q = PhysicalQudit(
                    Ne, Ng, 
                    omega_range[j], omega_rot, xi_range[j],
                    M_spam_order=M_spam_order
                )

        dt = 0.01
        # println("Omega value: ", get(phys_q.qudit.omega))
        measured_state_infidelity = zeros(N_gates)
        measured_population_infidelity = zeros(N_gates)
        for k = 1:N_gates
            s_inf, p_inf = measure_infidelity(
                                phys_q,
                                gates[k],
                                q.controls[gates[k]],
                                n_readout_samples,
                                dt=dt
                        ) 
            # println("Omega value: ", omega_range[j])
            # println("Omega: ", omega)
            measured_state_infidelity[k] = s_inf
            measured_population_infidelity[k] = p_inf  
            # println("Population Infidelity: ", p_inf)
            # println("This is p_inf: ", p_inf)
            # println("This is s_inf: ", s_inf) 
            meas_pop_infidelities_array[i,j,k] = p_inf 
            meas_state_infidelities_array[i,j,k] = s_inf
        end      
    end
end

###########################################################################################
# PLOTTING DATA 
###########################################################################################




save_object("meas_pop_infidelities_$(omega)_omega_rot_$(omega_rot)_bias_$(omega_bias)_spam_order_$(M_spam_order)_n_samples_$(n_samples).jld2", meas_pop_infidelities_array)
save_object("meas_state_infidelities_$(omega)_omega_rot_$(omega_rot)_bias_$(omega_bias)_spam_order_$(M_spam_order)_n_samples_$(n_samples).jld2", meas_state_infidelities_array)
save_object("q_list_bias_$(omega)_omega_rot_$(omega_rot)_bias_$(omega_bias)_spam_order_$(M_spam_order)_n_samples_$(n_samples).jld2", q_list)

meas_pop_infidelities_array_clamped = clamp.(meas_pop_infidelities_array, 5E-6, 1)
meas_state_infidelities_array_clamped = clamp.(meas_state_infidelities_array, 5E-6, 1)

#X gate
risk_neutral_x_gate = plot(omega_range .- omega, meas_pop_infidelities_array_clamped[1,:,1], label = "StDev:$(omega_stdev[1])", yscale =:log10, xlabel = "ω₁ parameter drift (GHz)", ylabel = "Measured infidelity", dpi = 250, yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:bottomleft)
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[2,:,1], label = "StDev:$(omega_stdev[2])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[3,:,1], label = "StDev:$(omega_stdev[3])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[4,:,1], label = "StDev:$(omega_stdev[4])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[5,:,1], label = "Classical OC")
savefig("risk_neutral_x_gate_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")
#Y gate 
risk_neutral_y_gate = plot(omega_range .- omega,meas_pop_infidelities_array_clamped[1,:,2], label = "StDev:$(omega_stdev[1])", yscale =:log10, xlabel = "ω₁ parameter drift (GHz)", ylabel = "Measured infidelity", dpi = 250, yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:bottomleft)
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[2,:,2], label = "StDev:$(omega_stdev[2])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[3,:,2], label = "StDev:$(omega_stdev[3])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[4,:,2], label = "StDev:$(omega_stdev[4])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[5,:,2], label = "Classical OC")
savefig("risk_neutral_y_gate_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")

#Z gate
risk_neutral_z_gate = plot(omega_range .- omega, meas_pop_infidelities_array_clamped[1,:,3], label = "StDev:$(omega_stdev[1])", yscale=:log10,xlabel = "ω₁ parameter drift (GHz)", ylabel = "Measured infidelity", dpi = 250, yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:bottomleft)
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[2,:,3], label = "StDev:$(omega_stdev[2])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[3,:,3], label = "StDev:$(omega_stdev[3])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[4,:,3], label = "StDev:$(omega_stdev[4])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[5,:,3], label = "Classical OC")
savefig("risk_neutral_z_gate_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")

#Hadamard gate
risk_neutral_hadamard_gate = plot(omega_range .- omega, meas_pop_infidelities_array_clamped[1,:,4], label = "StDev:$(omega_stdev[1])", yscale=:log10, xlabel = "ω₁ parameter drift (GHz)", ylabel = "Measured infidelity", dpi = 250, yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:bottomleft)
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[2,:,4], label = "StDev:$(omega_stdev[2])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[3,:,4], label = "StDev:$(omega_stdev[3])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[4,:,4], label = "StDev:$(omega_stdev[4])")
plot!(omega_range .- omega, meas_pop_infidelities_array_clamped[5,:,4], label = "Classical OC")
savefig("risk_neutral_hadamard_gate_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")

#Getting predicted infidelities 
predicted_infidelity_arr = zeros(number_deviations, N_gates)
for i in 1:number_deviations
    for j in 1:N_gates 
        predicted_infidelity_arr[i, j] = get(q_list[i].infidelity[gates[j]]) 
    end
end

#Plot predicted infidelities

omega_mask = omega_stdev .> 0.0
predicted_infidelity_arr_mask = predicted_infidelity_arr[omega_mask, :]
omega_stdev_mask = omega_stdev[omega_mask]

predicted_infidelities_plot = plot(omega_stdev_mask, predicted_infidelity_arr_mask, label = ["X gate" "Y gate" "Z gate" "Hadamard gate"], xscale=:log10, yscale=:log10, xlabel = "Standard Deviation of ω₁ parameter samples (GHz)", ylabel = "Predicted infidelity", dpi = 250, xticks = [10^-3, 10^-2.5, 10^-2, 10^-1.5, 10^-1], yticks = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:topleft, marker = [:circle :star :diamond :square], alpha = 0.8)


#Fake location for x = 0 point 

#Get default color palette 
color_palette = palette(:default)

x_fake = 0.0005
y_zero = predicted_infidelity_arr[5,:]
scatter!([x_fake], [y_zero[1]], label = "Classical OC X gate", markershape=:utriangle, markersize=3, color=color_palette[1])
scatter!([x_fake], [y_zero[2]], label = "Classical OC Y gate", markershape=:utriangle, markersize=3, color=color_palette[2])
scatter!([x_fake], [y_zero[3]], label = "Classical OC Z gate", markershape=:utriangle, markersize=3, color=color_palette[3])
scatter!([x_fake], [y_zero[4]], label = "Classical OC Hadamard gate", markershape=:utriangle, markersize=3, color=color_palette[4])



savefig("predicted_infidelity_vs_omega_stdev_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")


predicted_infidelities_plot = plot(omega_stdev, predicted_infidelity_arr[:,1], label = "X gate", xscale=:log10, yscale=:log10, xlabel = "Standard Deviation of ω₁ parameter samples (GHz)", ylabel = "Predicted infidelity", dpi = 250, xticks = [0.001, 0.005, 0.01, 0.05, 0.1], yticks = [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 1], legend =:bottomright)
plot!(omega_stdev, predicted_infidelity_arr[:,2], label = "Y gate")
plot!(omega_stdev, predicted_infidelity_arr[:,3], label = "Z gate")
plot!(omega_stdev, predicted_infidelity_arr[:,4], label = "Hadamard gate")
savefig("predicted_infidelity_vs_omega_stdev_omega_$(omega)_bias_$(omega_bias)_omega_rot_$(omega_rot)_spam_order_$(M_spam_order)_n_samples_$(n_samples).png")