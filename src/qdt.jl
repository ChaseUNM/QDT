using LinearAlgebra, QuantumGateDesign, Random, Distributions

#####################################################################
#
# QDT STRUCT
#
#####################################################################

mutable struct QDT
    #inputs 

    #Model description
    N::Int64 
    Ne::Vector{Int64}
    Ng::Vector{Int64}
    omega_rot::Float64
    param_samples::Vector{Float64}
    n_param_samples::Int64
    
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
    
    #move outside 
    #"real" system 
    measured_population_infidelity::Vector{Float64}
    measured_state_infidelity::Vector{Float64}
    readout_samples::Int64 
    Mspam_order::Float64
    Mspam_matrices::Vector{Matrix{Float64}}

    function QDT(   N, Ne, Ng, 
                    omega_rot, 
                    gate_set, 
                    Mspam_order; 
                    n_param_samples=100,
                    control_degree=2,
                    control_nsplines=5,
                    T_gate = 50.0,
                    steps = 200,
                    readout_samples=1000
        )
        param_samples = zeros(n_param_samples)
        N_gates = length(gate_set)
        real_control_op = [0 1.0; 1.0 0]
        imag_control_op = [0 1.0; -1.0 0]
        degree = fill(control_degree, N_gates)
        nsplines = fill(control_nsplines, N_gates)
        T_gate = fill(T_gate, N_gates)
        steps = fill(steps, N_gates)
        N_coeff = 2 .*degree .*nsplines 
        
        controls = [FortranBSplineControl(degree[i], nsplines[i], T_gate[i]) for i in 1:N_gates]
        max_control_parameter = 0.1
        pcof_l = -max_control_parameter
        pcof_u = max_control_parameter
        pcof_by_gate = [(0.5 .- rand(controls[i].N_coeff)) .* max_control_parameter for i in 1:N_gates] 
        predicted_infidelity = zeros(N_gates) 

        measured_population_infidelity = zeros(N_gates)
        measured_state_infidelity = zeros(N_gates)
        Mspam_matrices = [column_stochastic(Mspam_order*abs.(rand(Normal(0,1), Ne[i] + Ng[i]))) for i in 1:N]

        new(N, Ne, Ng, omega_rot, param_samples, n_param_samples, real_control_op, imag_control_op, degree, nsplines, T_gate, steps, N_gates, gate_set, pcof_by_gate, predicted_infidelity, controls, pcof_l, pcof_u, measured_population_infidelity,measured_state_infidelity, readout_samples, Mspam_order, Mspam_matrices)
    end
    #perform optimization with qdt 

    #check with PhysicalTwin 

    #update history 
end 






function optimize_controls(
        qdt::QDT; 
        ipopt_options=["max_iter" => 100, "print_level" => 3]
    )
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
        #include while loop to accept predicted infidelity when it's small
        probs = [SchrodingerProb([0 0; 0 j], [real_control_ops], [imag_control_ops], U0, T_gates[i], steps[i]) for j in qdt.param_samples .- qdt.omega_rot]

        opt_ret_multiple = optimize_prob(probs, controls[i], pcof_by_gate[i], gate_set[i], pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=ipopt_options)

        qdt.predicted_infidelity[i] = opt_ret_multiple.obj_val
        qdt.pcof_by_gate[i] = opt_ret_multiple.x
    end
end




#####################################################################
#
# QDT_HISTORY STRUCT
#
#####################################################################

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
