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

#Loop both bias and standard deviation to see how the predicted and measured infidelities change with different parameter distributions.

#Ideally want it so if predicted infidelity is too large then change standard deviation, if measured infidelity is too large then change bias 

#Need to start with some initial standard deviation and some initial bias, this will be imprecise and inaccurate 

omega_center = 4.5
xi_center = 4.5
xi_bias = 0.0
xi_stdev = 0.0

stdev_all = vcat(10 .^ -LinRange(1,5,25), 0.0)
bias_all = vcat(10 .^ -LinRange(1,5,25), 0.0)

max_number_of_simulations = length(stdev_all)*length(bias_all)

#Ensure global variables work inside loop

global std_dev_init = stdev_all[1]
global bias_init = bias_all[1]

#Count variable is for add_param_samples function
global characterization_count = 1

#Create dictionaries to store information

q_meas_infidelities_state = OrderedDict{Tuple{Float64, Float64}, Dict{GateType, Float64}}()
q_meas_infidelities_pop = OrderedDict{Tuple{Float64, Float64}, Dict{GateType, Float64}}()
q_pred_infidelities = OrderedDict{Tuple{Float64, Float64}, Dict{GateType, Float64}}()

control_history = OrderedDict{Tuple{Float64, Float64}, Dict}()

large_predicted_infidelity_flag = OrderedDict{Tuple{Float64,Float64}, Dict}()


omega_sampler = Normal(omega_center + bias_init, std_dev_init)
xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
omega_samples = rand(omega_sampler, n_samples)
xi_samples    = rand(xi_sampler, n_samples)

q = DigitalQudit(Ne, Ng)
q.omega_rot = omega_center
add_param_samples(q, omega_samples, xi_samples; iter = characterization_count, average_omega_rot = false)

q_p_meas_infidelity = OrderedDict{GateType, Float64}()
q_s_meas_infidelity = OrderedDict{GateType, Float64}()
q_pred_infidelity = OrderedDict{GateType, Float64}()
control_dict = OrderedDict{GateType, QuditControl}()

for j = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[j], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[j]] = History(Float64)
end

for j = 1:N_gates
    optimize_control(q, gates[j], options=["max_iter" => 100, "print_level" => 3])
    control_dict[gates[j]] = q.controls[gates[j]]
    q_pred_infidelity[gates[j]] = get(q.infidelity[gates[j]])
end   

q_pred_infidelities[(std_dev_init, bias_init)] = q_pred_infidelity 
control_history[(std_dev_init, bias_init)] = control_dict

for i in 1:N_gates
    q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
    q_p_meas_infidelity[gates[i]] = q_p_inf 
    q_s_meas_infidelity[gates[i]] = q_s_inf 
end
q_meas_infidelities_state[(std_dev_init, bias_init)] = q_s_meas_infidelity
q_meas_infidelities_pop[(std_dev_init, bias_init)] = q_p_meas_infidelity

experiment = 1


for i in 1:max_number_of_simulations
    #Increase count for iterations in ValueHistories
    global characterization_count += 1
    # Now to recharacterize 
    # A large predicted infidelity will be handled by decreasing standard deviations 
    
    # If any predicted infidelity is too large, decrease standard deviation 
    # This is not the only option, could also increase time of the gate
    # Count how many predicted and measured infidelities are larger than epsilon
    pred_array = collect(values(q_pred_infidelities[(std_dev_init, bias_init)]))
    meas_pop_array = collect(values(q_meas_infidelities_pop[(std_dev_init, bias_init)]))

    pred_larger = count(x -> x > epsilon, pred_array)
    meas_pop_larger = count(x -> x > epsilon, meas_pop_array)

    if experiment == 1
        if any(values(q_pred_infidelities[(std_dev_init, bias_init)]) .> epsilon)
            std_dev = std_dev_init * 10^(-1/6)
        else
            std_dev = std_dev_init
        end
        

        # If all predicted infidelities are good, but measured infidelities are bad then the issue is with the accuracy of the model 
        # Therefore decrease bias 
        if all(values(q_pred_infidelities[(std_dev_init, bias_init)]) .< epsilon) && any(values(q_meas_infidelities_pop[(std_dev_init, bias_init)]) .> epsilon)
            bias = bias_init * 10^(-1/6)
        else
            bias = bias_init
        end

    elseif experiment == 2
        std_dev = std_dev_init * 10^(-1/6)
        bias = bias_init * 10^(-1/6)

    elseif experiment == 3
        if any(values(q_pred_infidelities[(std_dev_init, bias_init)]) .> epsilon)
            std_dev = std_dev_init * 10^(-1/6)
            bias = bias_init * 10^(-1/12)
        end

        if all(values(q_pred_infidelities[(std_dev_init, bias_init)]) .< epsilon) && any(values(q_meas_infidelities_pop[(std_dev_init, bias_init)]) .> epsilon)
            bias = bias_init * 10^(-1/6)
            std_dev = std_dev_init * 10^(-1/12)
        end

    elseif experiment == 4
        std_dev = std_dev_init * 10^(pred_larger * -1/12)
        bias = bias_init * 10^(meas_pop_larger * -1/12)
    end

    println("Standard deviation: $std_dev")
    println("Bias: $bias")

    omega_sampler = Normal(omega_center + bias, std_dev)
    xi_sampler    = Normal(xi_center + xi_bias, xi_stdev)
    omega_samples = rand(omega_sampler, n_samples)
    xi_samples    = rand(xi_sampler, n_samples)
    add_param_samples(q, omega_samples, xi_samples; iter = characterization_count, average_omega_rot = false)
    q.omega_rot = omega_center


    q_pred_infidelity = OrderedDict{GateType, Float64}()
    q_p_meas_infidelity = OrderedDict{GateType, Float64}()
    q_s_meas_infidelity = OrderedDict{GateType, Float64}()
    control_dict = OrderedDict{GateType, QuditControl}()

    # Re-optimize if measured infidelity was too low or predicted infidelity was too low, we want both of these values to be small
    # This should only really have an impact if the optimzier gets lucky at some point and starts off with a very small predicted infidelity 
    # If this doesn't happen then can just keep the same control pulses
    for j = 1:N_gates
        if q_meas_infidelities_pop[(std_dev_init, bias_init)][gates[j]] > epsilon || q_pred_infidelities[(std_dev_init, bias_init)][gates[j]] > epsilon
        
            println("#######################################################################")
            println("re-optimizing: ")
            println("measured infidelity: ", q_meas_infidelities_pop[(std_dev_init, bias_init)][gates[j]])
            println("predicted infidelity: ", q_pred_infidelities[(std_dev_init, bias_init)][gates[j]])
            println("#######################################################################")
            optimize_control(q, gates[j]; iter = characterization_count, options=["max_iter" => 100, "print_level" => 3])
        else
            println("########################################################################")
            println("no re-optimizing")
            println("measured infidelity: ", q_meas_infidelities_pop[(std_dev_init, bias_init)][gates[j]])
            println("predicted infidelity: ", q_pred_infidelities[(std_dev_init, bias_init)][gates[j]])
            println("#########################################################################")
        end
        q_pred_infidelity[gates[j]] = get(q.infidelity[gates[j]])
        control_dict[gates[j]] = q.controls[gates[j]]
    end
    control_history[(std_dev, bias)] = control_dict 

    # Now obtain the measured infidelity

    for i in 1:N_gates
        q_s_inf, q_p_inf = measure_infidelity(phys_q, gates[i], q.controls[gates[i]], n_readout_samples)
        q_p_meas_infidelity[gates[i]] = q_p_inf 
        q_s_meas_infidelity[gates[i]] = q_s_inf 
    end

    # Add data within loop to outside data
    q_meas_infidelities_state[(std_dev, bias)] = q_s_meas_infidelity
    q_meas_infidelities_pop[(std_dev, bias)] = q_p_meas_infidelity
    q_pred_infidelities[(std_dev, bias)] = q_pred_infidelity

    if all(values(q_p_meas_infidelity) .< epsilon) && all(values(q_pred_infidelity) .< epsilon)
        println("Loop completed, all predicted and measured infidelities less than tolerance")
        break
    end

    global std_dev_init = std_dev 
    global bias_init = bias
    
end

# Now save all data if there is save_data flag is true
save_data = true
if save_data
    @save "bias_stdev_data_$experiment.jld2" q_meas_infidelities_state q_meas_infidelities_pop q_pred_infidelities control_history large_predicted_infidelity_flag
    @save "bias_stdev_parameters_$experiment.jld2" stdev_all bias_all
    @save "qudit_$experiment.jld2" q
end


