using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
#Want to create QDT object 

function column_stochastic(epsilon_list::Array{Float64})
    # Ident = Matrix(1.0*I, N, N)
    N = length(epsilon_list)
    M = zeros(N, N)

    for i in 1:N
        col_vec = zeros(N)
        col_vec[1] = 1 - epsilon_list[i]
        remaining_elements = rand(N - 1)
        remaining_sum = sum(remaining_elements)
        remaining_elements *= epsilon_list[i]/remaining_sum 
        col_vec[2:N] = remaining_elements 
        col_vec[1], col_vec[i] = col_vec[i], col_vec[1]
        M[:,i] = col_vec 
    end
    return M 
end

mutable struct QDT
    #inputs 

    #Model description
    N::Int64 
    Ne::Vector{Int64}
    Ng::Vector{Int64}
    param_samples::Vector{Float64}
    number_of_samples::Int64
    real_control_op::AbstractMatrix
    imag_control_op::AbstractMatrix

    #Control parameters
    degree::Vector{Int64}
    nsplines::Vector{Int64}
    T_gate::Vector{Float64}
    steps::Vector{Int64}
    N_gates::Int64
    gate_set::Vector{AbstractMatrix}
    pcof_by_gate:: Vector{Vector{Float64}} #will need to change to vector of vectors for differing size of gates
    predicted_infidelity:: Vector{Float64}
    controls::Vector{AbstractControl}
    pcof_l::Float64
    pcof_u::Float64
    
    #"real" system 
    measured_population_infidelity::Vector{Float64}
    measured_state_infidelity::Vector{Float64}
    readout_samples::Int64 
    Mspam_order::Float64
    Mspam_matrices::Vector{Matrix{Float64}}

    function QDT(N, Ne, Ng, gate_set, Mspam_order)
        number_of_samples = 10
        param_samples = zeros(number_of_samples)
        N_gates = length(gate_set)
        real_control_op = [0 1.0; 1.0 0]
        imag_control_op = [0 1.0; -1.0 0]
        degree = fill(2, N_gates)
        nsplines = fill(3, N_gates)
        T_gate = fill(50.0, N_gates)
        steps = fill(200, N_gates)
        N_coeff = 2 .*degree .*nsplines 
        
        controls = [FortranBSplineControl(degree[i], nsplines[i], T_gate[i]) for i in 1:N_gates]
        max_control_parameter = 0.1
        pcof_l = -max_control_parameter
        pcof_u = max_control_parameter
        pcof_by_gate = [(0.5 .- rand(controls[i].N_coeff)) .* max_control_parameter for i in 1:N_gates] 
        predicted_infidelity = zeros(N_gates) 

        measured_population_infidelity = zeros(N_gates)
        measured_state_infidelity = zeros(N_gates)
        readout_samples = 1000
        Mspam_matrices = [column_stochastic(Mspam_order*abs.(rand(Normal(0,1), Ne[i] + Ng[i]))) for i in 1:N]

        new(N, Ne, Ng, param_samples, number_of_samples, real_control_op, imag_control_op, degree, nsplines, T_gate, steps, N_gates, gate_set, pcof_by_gate, predicted_infidelity, controls, pcof_l, pcof_u, measured_population_infidelity,measured_state_infidelity, readout_samples, Mspam_order, Mspam_matrices)
    end
    #perform optimization with qdt 

    #check with PhysicalTwin 

    #update history 
end 



function optimize_controls(qdt::QDT)
    N_gates = qdt.N_gates 
    N = qdt.N
    Ne = qdt.Ne 
    Ng = qdt.Ng
    U0 = Matrix(1.0*I, Ne[1] + Ng[1], Ne[1] + Ng[1])
    real_control_ops = qdt.real_control_op 
    imag_control_ops = qdt.imag_control_op 
    steps = qdt.steps
    T_gates = qdt.T_gate
    
    gate_set = qdt.gate_set
    controls = qdt.controls
    pcof_by_gate = qdt.pcof_by_gate
    pcof_l = qdt.pcof_l 
    pcof_u = qdt.pcof_u
    for i in 1:N_gates 
        probs = [SchrodingerProb([0 0; 0 j], [real_control_ops], [imag_control_ops], U0, T_gates[i], steps[i]) for j in qdt.param_samples .- omega_rot]
        opt_ret_multiple = optimize_prob(probs, controls[i], pcof_by_gate[i], gate_set[i], pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100, "print_level" => 3])
        qdt.predicted_infidelity[i] = opt_ret_multiple.obj_val
        qdt.pcof_by_gate[i] = opt_ret_multiple.x
    end
end

# mutable struct PhysicalTwin


# end



function noisy_observation(epsilons, samples, p)
    N = length(epsilons)
    M = zeros(N, N)
    for i in 1:N
        col_vec = zeros(N)
        col_vec[1] = 1 - epsilons[i]
        remaining_elements = rand(N - 1)
        remaining_sum = sum(remaining_elements)
        remaining_elements *= epsilons[i]/remaining_sum 
        col_vec[2:N] = remaining_elements 
        col_vec[1], col_vec[i] = col_vec[i], col_vec[1]
        M[:,i] = col_vec 
    end
    p_hat = M*p
    p_obs = zeros(size(p_hat))
    for i in 1:size(p_hat, 1)
        p_hat[:,i] = p_hat[:,i]/sum(p_hat[:,i])
        p_distribution = Multinomial(samples, p_hat[:,i])
        N_shots = rand(p_distribution)
        p_obs[:,i] = N_shots/samples 
    end
    return p_obs 
end

function sample_quantum_pops(samples, p)
    p_obs = zeros(size(p))
    for i in 1:size(p, 1)
        p[:,i] = p[:,i]/sum(p[:,i])
        p_distribution = Multinomial(samples, p[:,i])
        N_shots = rand(p_distribution)
        p_obs[:,i] = N_shots/samples 
    end
    return p_obs 
end

function infidelity_population(p, q)
    A = (sum(sqrt.(p) .* sqrt.(q), dims = 1)).^2
    return 1 - mean(A)
end

function infidelity_state(p, q)
    A = abs2.(sum(conj(p).*q, dims = 1))
    return 1 - mean(A)
end

function measured_infidelity(qdt::QDT, true_parameter)
    real_control_ops = [0 1; 1 0]
    imag_control_ops = [0 1; -1 0]
    N_gates = qdt.N_gates 
    gate_set = qdt.gate_set 
    steps = qdt.steps
    controls = qdt.controls 
    pcof_by_gate = qdt.pcof_by_gate
    prod_states = prod(qdt.Ng .+ qdt.Ne)
    #Change this to be somewhere else

    # measured_population_infidelity = zeros(N_gates)
    # measured_state_infidelity = zeros(N_gates)
    for i in 1:N_gates
        true_problem = SchrodingerProb([0 0; 0 true_parameter - omega_rot], [real_control_ops], [imag_control_ops], U0, T_gates[i], steps[i])
        meas_state_history = eval_forward(true_problem, controls[i], pcof_by_gate[i])
        final_state = meas_state_history[:,end,:]
        final_state_abs = abs2.(final_state)
        final_gate_unitary = gate_set[i]*U0 
        
        final_gate_abs = abs2.(final_gate_unitary)
        # display(final_state_abs)
        # display(sum(final_state_abs[1,:]))
        # display(sum(final_state_abs[2,:]))
        # display(final_state_abs)
        M_spam_i = qdt.Mspam_matrices[1]
        p_hat = M_spam_i*final_state_abs
        observed_final_state = sample_quantum_pops(qdt.readout_samples, p_hat)
        # observed_final_state = noisy_observation(epsilon_vec, samples, final_state_abs)
        # println("Measured state")
        # display(observed_final_state)
        # println("Gate")
        # display(final_gate_abs)
        qdt.measured_population_infidelity[i] = infidelity_population(observed_final_state, final_gate_abs)
        qdt.measured_state_infidelity[i] = infidelity_state(final_state, final_gate_unitary)
        # println("infidelity gate")
        # println(infidelity_gate)
        #need to mess it up 
        #apply M_spam and then sample 
    end
end


# optimize_controls(qdt)

mutable struct qdt_history 
    true_parameter::Vector{Float64}
    param_samples::Vector{Vector{Float64}}
    pcof_by_gate::Vector{Vector{Vector{Float64}}}
    #Include gate set
    predicted_infidelity::Matrix{Float64}
    measured_population_infidelity::Matrix{Float64}
    measured_state_infidelity::Matrix{Float64}
    maximum_size::Int64 
    next_index::Int64 
    N_gates::Int64
    function qdt_history(maximum_size, N_gates)
        next_index = 1
        true_parameter = Vector{Float64}(undef, maximum_size)
        param_samples = Vector{Vector{Float64}}(undef, maximum_size)
        pcof_by_gate = Vector{Vector{Vector{Float64}}}(undef, maximum_size)
        # predicted_infidelity = Vector{Vector{Float64}}(undef, maximum_size)
        # measured_population_infidelity = Vector{Vector{Float64}}(undef, maximum_size)
        predicted_infidelity = zeros(N_gates, maximum_size)
        measured_population_infidelity = zeros(N_gates, maximum_size)
        measured_state_infidelity = zeros(N_gates, maximum_size)

        new(true_parameter, param_samples, pcof_by_gate, predicted_infidelity, measured_population_infidelity,measured_state_infidelity, maximum_size, next_index, N_gates)
    end
end


function save_state(qdt::QDT, history::qdt_history, true_parameter)
    index = history.next_index 
    history.true_parameter[index] = copy(true_parameter)
    history.param_samples[index] = copy(qdt.param_samples)
    history.pcof_by_gate[index] = copy(qdt.pcof_by_gate) 
    history.predicted_infidelity[:,index] .= qdt.predicted_infidelity
    history.measured_population_infidelity[:,index] .= qdt.measured_population_infidelity
    history.measured_state_infidelity[:,index] .= qdt.measured_state_infidelity
    history.next_index += 1
end




N = 1 
Ne = [2]
Ng = [0]
x_gate = [0 1.0; 1.0 0]
y_gate = [0 im; -im 0]
z_gate = [1.0 0; 0 -1.0]
gate_set = [x_gate, y_gate, z_gate]
N_gates = length(gate_set)
Mspam_order = 1E-3
Twin = QDT(N, Ne, Ng, gate_set, Mspam_order)
maximum_size = 100
history = qdt_history(maximum_size, N_gates)

# center = 4.5
# std_dev = 0.0
# param_samples = Normal(center, std_dev)
# omega_rot = center
# Twin.param_samples = rand(param_samples, Twin.number_of_samples) 

true_parameter = 4.5
param_samples = Normal(true_parameter, std_dev)
Twin.param_samples = rand(param_samples, Twin.number_of_samples)
optimize_controls(Twin)
# optimize_controls(Twin)
# measured_infidelity(Twin, true_parameter)
# save_state(Twin, history, true_parameter)

#Case where characterize and then optimize

for i in 1:maximum_size 
    global true_parameter += 0.01*abs(rand())
    std_dev = 0.0
    measured_infidelity(Twin, true_parameter)
    save_state(Twin, history, true_parameter)
end

measured_population_infidelity_plot = plot(history.measured_population_infidelity', yscale=:log10, labels = ["Gate 1" "Gate 2" "Gate 3"], ylabel = "Population Infidelity")
measured_state_infidelity_plot = plot(history.measured_state_infidelity', yscale=:log10, labels = ["Gate 1" "Gate 2" "Gate 3"], ylabel = "State Infidelity")



# mutable struct PhysicalDevice 
#     transition_freq::
#     self_kerr::
#     zz_coupling::
#     dipole_dipole::

    



# mutable struct qdt
#     #inputs:
#     N::Int64 #Number of subsystems 
#     Ne::Vector{Int64} #Number of essential energy levels in each subsystem
#     Ng::Vector{Int64} #Number of guard energy levels in each subsystem 
#     transition_samples::Vector{Float64} #Samples of Transition frequencies
#     degree::Int64 = 2
#     nsplines::Int64 = 2
#     T::Float64 #Duration of gate (same T for each gate for now)
#     steps::Int64 #steps for numerical integration 

#     #create schrodinger prob: 
#     probs = [SchrodingerProb([0 0; 0 i], real_control_ops, imag_control_ops, U0, tf, nsteps) for i in samples]
#     #now do Risk neutral optimization 
#     control = FortranBSplineControl(degree, nsplines, steps)
#     max_control_parameter = 0.1
#     pcof_l = -max_control_parameter
#     pcof_u = max_control_parameter
#     pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter 
#     opt_ret = optimize_prob(probs, control, pcof0, hadamard_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100])
#     pcof_optimal = opt_ret.x 
#     #get overall predicted and infidelity individual predicted infidelity




#     #nsplines will be coming from DM 
#     #specify initial control pcof (begin with random but think about)
#     #call optimize with transition samples, with specified control parameters
#     #Pass in samples of transition frequencies



#     #in some way need to feed in the model hamiltonian (whether that's just parameters or a matrix)
#     #need samples, can either pass in a vector of samples or a pre-specified distribution 
#     #probably want to determine structure of control pulse (degree, # of splines)
#     #for functions want to add functions like a simulate and optimize function (both of these will just call QuantumGateDesign.jl)
#     #set of gates to optimize(at most 2 qubit gates)
#     #pass in number of subsystems, energy levels for each qudit (essential and computational)
#     #

#     #returns: 
#     #want history of population of states
#     #want history of how parameters got updated
#     #

# mutable struct PhysicalTwin 
#     #in some way feed in the model hamiltonian (whether that's just parameters or a matrix)
#     #don't need samples (this is deterministic)
#     #