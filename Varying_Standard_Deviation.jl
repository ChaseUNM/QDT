using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2
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
#Loop for different standard deviations values to see how the predicted and measured infidelities change with different parameter distributions.

std_dev_all = vcat(10 .^ -LinRange(1,5,25), 0.0)
std_dev_init = std_dev_all[1]
omega_bias = 0.001

q_meas_infidelities_state = Vector{Dict{GateType, Float64}}(undef, length(std_dev_all))
q_meas_infidelities_pop = Vector{Dict{GateType, Float64}}(undef, length(std_dev_all))
q_pred_infidelities = Vector{Dict{GateType, Float64}}(undef, length(std_dev_all))
q_list = Vector{DigitalQudit}(undef, length(std_dev_all))
count = 1

omega_sampler = Normal(omega_center + omega_bias, std_dev_init)
xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
omega_samples = rand(omega_sampler, n_samples)
xi_samples    = rand(xi_sampler, n_samples)

q = DigitalQudit(Ne, Ng)
q.omega_rot = omega_center
add_param_samples(q, omega_samples, xi_samples; iter = count, average_omega_rot = false)
q_p_meas_infidelity = Dict{GateType, Float64}()
q_s_meas_infidelity = Dict{GateType, Float64}()

pred_recharacterization_flag = Dict{Float64, Bool}() 
reoptimization_flag = Dict{Float64, Dict}()
large_predicted_infidelity_flag = Dict{Float64, Dict}()

control_history = Dict{Float64, Dict}()

q_pred_infidelity = Dict{GateType, Float64}()
# Create a control for each gate
for j = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[j], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[j]] = History(Float64)
end
control_dict = Dict{GateType, QuditControl}()
for j = 1:N_gates
    optimize_control(q, gates[j], options=["max_iter" => 100, "print_level" => 3])
    control_dict[gates[j]] = q.controls[gates[j]]
    q_pred_infidelity[gates[j]] = get(q.infidelity[gates[j]])
end   
q_pred_infidelities[count] = q_pred_infidelity 
control_history[std_dev_init] = control_dict
q_list[count] = q

for i in 1:N_gates
    q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
    q_p_meas_infidelity[gates[i]] = q_p_inf 
    q_s_meas_infidelity[gates[i]] = q_s_inf 
end
q_meas_infidelities_state[count] = q_s_meas_infidelity
q_meas_infidelities_pop[count] = q_p_meas_infidelity

if all(values(q_meas_infidelities_pop[count]) .< epsilon)
    println("All measured fidelities sufficiently small")
    recharacterization_flag[omega_bias] = false
else
    println("Need more characterization")
    recharacterization_flag[omega_bias] = true
end

count += 1

#Care more about predicted infidelities here than measured, but will still measure at each iteration to see how it changes with different parameter distributions.
for i in 2:length(std_dev_all)
    std_dev = std_dev_all[i]
    println("Standard deviation: ", std_dev)

    omega_sampler = Normal(omega_center + omega_bias, std_dev)
    xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
    omega_samples = rand(omega_sampler, n_samples)
    xi_samples    = rand(xi_sampler, n_samples)
    add_param_samples(q, omega_samples, xi_samples; iter = count, average_omega_rot = false)
    q.omega_rot = omega_center

    q_pred_infidelity = Dict{GateType, Float64}()
    # If infidelity was small enough, then won't need to re-optimize
    optim_dict = Dict{GateType, Bool}()
    control_dict = Dict{GateType, QuditControl}()
    q_p_meas_infidelity = Dict{GateType, Float64}()
    q_s_meas_infidelity = Dict{GateType, Float64}()
    # if any(values(q_pred_infidelities[count-1]) .> epsilon)
    println("#######################################################################")
    println("predicted infidelity", q_pred_infidelities[count - 1])
    println("#######################################################################")
    for j = 1:N_gates
        optimize_control(q, gates[j]; iter = count, options=["max_iter" => 100, "print_level" => 3])
        q_pred_infidelity[gates[j]] = get(q.infidelity[gates[j]])
        control_dict[gates[j]] = q.controls[gates[j]]
    end
    # pred_recharacterization_flag[std_dev_all[count-1]] = true
    # else
    #     println("########################################################################")
    #     println("small enough predicted infidelity, no need to re-characterize (hopefully): ", q_pred_infidelities[count - 1])
    #     println("#########################################################################")
    #     recharacterization_flag[std_dev_all[count-1]] = false
    #     # break
    # end

    for i in 1:N_gates
        q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
        q_p_meas_infidelity[gates[i]] = q_p_inf 
        q_s_meas_infidelity[gates[i]] = q_s_inf
    end
    q_meas_infidelities_state[count] = q_s_meas_infidelity
    q_meas_infidelities_pop[count] = q_p_meas_infidelity
    q_pred_infidelities[count] = q_pred_infidelity
    control_history[count] = control_dict
    global count += 1
end
