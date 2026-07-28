using LinearAlgebra, QuantumGateDesign, ValueHistories
include("gates.jl")
include("controls.jl")
include("util.jl")
include("digital_qudit.jl")

################################################################
# Struct definition
################################################################

# mutable struct DigitalQuditPair

#     # FIELDS
#     Ne::Int64                          # Number of essential energy levels
#     Ng::Int64                          # Number of guard levels
#     omega_rot::Float64                     # Rotating frame frequency
#     omega::History{Int64, Vector{Float64}} # History of qubit frequency samples
#     xi::History{Int64, Vector{Float64}} # History of qubit self-kerr samples
#     controls::Dict{GateType, QuditControl}
#     infidelity::Dict{GateType, History{Int, Float64}}

#     # CONSTRUCTOR
#     function DigitalQuditPair(q1::DigitalQudit, q2::DigitalQudit)
#         xi = History(Vector{Float64})
#         J  = History(Vector{Float64})
#         controls = Dict{GateType, Vector{QuditControl}}()
#         infidelity = Dict{GateType, History{Int64, Float64}}()
#         new(q1, q2, xi, J, controls, infidelity)
#     end
# end

mutable struct DigitalQuditPair

    # FIELDS
    qudit1::DigitalQudit
    qudit2::DigitalQudit
    xi::History{Int64, Vector{Float64}}
    J::History{Int64, Vector{Float64}}
    controls::Dict{Union{GateType,ProductGate}, Vector{QuditControl}}
    infidelity::Dict{Union{GateType,ProductGate}, History{Int64, Float64}}

    # CONSTRUCTOR
    function DigitalQuditPair(q1::DigitalQudit, q2::DigitalQudit)
        xi = History(Vector{Float64})
        J  = History(Vector{Float64})
        controls = Dict{Union{GateType,ProductGate}, Vector{QuditControl}}()
        infidelity = Dict{Union{GateType,ProductGate}, History{Int64, Float64}}()

        new(q1, q2, xi, J, controls, infidelity)
    end
end



##################################################################
# SETTING/UPDATING PARAMETERS
##################################################################

# function add_param_samples(
#                     q::DigitalQudit, 
#                     omega::Vector{Float64}, 
#                     xi::Vector{Float64};
#                     iter::Int64=-1
#     )
#     # Adds a new sample of cross-Kerr and Jaynes-Cummings
#     # couplings for this pair of qudits.
#     #
#     # By default, 'iter' is set to the latest iteration + 1
#     # 
#     push!(q.omega, iter, omega)
#     push!(q.xi, iter, xi)
#     q.omega_rot = mean(omega)
# end

function add_param_samples(
    self::DigitalQuditPair,
    xi::Vector{Float64},
    J::Vector{Float64};
    iter::Int64 = -1
)
    if length(xi) != length(J)
        throw("DigitalQuditPair::add_param_samples(): Number of xi and J samples must match!")
    end

    push!(self.xi, iter, xi)
    push!(self.J, iter, J)
end


function update_param_samples(
                    self::DigitalQuditPair, 
                    xi::Vector{Float64},
                    J::Vector{Float64}
    )
    # Updates the sample cross-Kerr and Jaynes-Cummings coupling 
    # coefficients to this object's history at the latest 
    # timestamp in its history.
    
    # Throw error if the parameter histories are empty. 
    if length(self.xi) == 0 || length(self.J) == 0
        throw("Cannot update parameters of empty qudit history!")
    end

    # Verify the number of xi and J parameters match
    if length(xi) != length(J)
        throw("DigitalQuditPair::update_param_samples(): Number of xi and J samples must match!")
    end

    self.xi.values[end] = xi
    self.J.values[end] = J 
end



# function add_control(q::DigitalQudit, gate::GateType, 
#                      control_obj::AbstractControl; iter::Int64=-1)
#     # Adds an entry to this qudits control history for the provided gate.
#     # If timestamp 'iter' isn't provided, the control is added to the 
#     # history at 1 + {the latestest control timestamp}
#     if !haskey(q.controls, gate)
#         q.controls[gate] = QuditControl(control_obj)
#         randomize_coeffs(q.controls[gate])
#     else
#         push!(q.controls[gate], iter, c)
#     end
# end

function add_control(
    self::DigitalQuditPair,
    gate::Union{GateType, ProductGate},
    q1_control::QuditControl,
    q2_control::QuditControl
)
    self.controls[gate] = [q1_control, q2_control]

    if !haskey(self.infidelity, gate)
        self.infidelity[gate] = History(Float64)
    end
end

##################################################################
# SIMULATION ROUTINES
##################################################################


function get_drift_hamiltonians(self::DigitalQuditPair)
    # Returns a Vector of Matrices representing the drift
    # Hamiltonian (in the rotating frames) for each of this 
    # pair of qudits current parameter samples

    q1 = self.qudit1
    q2 = self.qudit2
    subsystem_sizes = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]

    # Lowering operator by subsystem
    a1 = promote_subsys_op(lower_op(subsystem_sizes[1]), subsystem_sizes, 1)
    a2 = promote_subsys_op(lower_op(subsystem_sizes[2]), subsystem_sizes, 2)

    # Drift Hamiltonians by subsystem
    H_drift_1 = get_drift_hamiltonians(q1)
    H_drift_2 = get_drift_hamiltonians(q2)

    # Coupling parameters
    _, xi = last(self.xi)
    _, J = last(self.J)
    n_samples = length(xi)    

    # Verify the number of parameters for the subsystems
    # matches with the number of coupling parameters
    # println(n_samples)
    # if n_samples != length(last(self.qudit1.xi)) || n_samples != length(last(self.qudit2))
    #     throw("DigitalQuditPair::get_drift_hamiltonians(): Number of coupling parameters doesnt match individual qudit parameters")
    # end

    # Full drift Hamiltonians
    N = prod(subsystem_sizes)
    H_drift = Array{Float64}(undef, n_samples, N, N)
    for j = 1:n_samples
        H_drift[j,:,:] = (
            promote_subsys_op(H_drift_1[j], subsystem_sizes, 1) 
            + promote_subsys_op(H_drift_2[j], subsystem_sizes, 2) 
            - xi[j] * (a1'*a1)*(a2'*a2)
            + J[j] * (a1 * a2' + a1' * a2)
        )
    end
    return H_drift
end


function get_control_hamiltonians(self::DigitalQuditPair)
    # Returns a Vector of Matrices representing the real and 
    # imaginary parts of the control Hamiltonians for each of 
    # this pair of qudits current parameter samples.
    # These control Hamiltonians are of the same size as the 
    # *full* system

    # Subsystem sizes
    q1 = self.qudit1
    q2 = self.qudit2
    n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]

    # Control Hamiltonians of the subsystems
    H_c_re_1, H_c_im_1 = get_control_hamiltonians(q1)
    H_c_re_2, H_c_im_2 = get_control_hamiltonians(q2)

    # Promote to the full system
    H_c_re = [
        promote_subsys_op(H_c_re_1, n, 1),
        promote_subsys_op(H_c_re_2, n, 2),  
    ]
    H_c_im = [
        promote_subsys_op(H_c_im_1, n, 1),
        promote_subsys_op(H_c_im_2, n, 2),  
    ]

    return H_c_re, H_c_im

end



function get_schrodinger_problems(self::DigitalQuditPair, T, dt)
    # Returns a Vector of SchrodingerProb objects, one for each 
    # of this pair of qudit's current parameter samples.
    # Argument 'T' specifies the time integration interval [0,T]
    # and dt is the stepsize
    #

    q1 = self.qudit1
    q2 = self.qudit2

    # Initial state
    n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]
    n_ess = [q1.Ne, q2.Ne]
    U0 = initial_states(n, n_ess)

    # Drift Hamiltonians for the full system
    H_drift = get_drift_hamiltonians(self)
    
    # Unscaled control Hamiltonian by subsystem
    H_c_re, H_c_im = get_control_hamiltonians(self)

    # Number of timesteps
    n_timesteps = ceil(Int, T/dt)

    # Generating the SchrodingerProb
    probs = [  SchrodingerProb(
                    H_drift[j,:,:], 
                    H_c_re, H_c_im, 
                    U0, T, n_timesteps
                ) 
                for j in 1:size(H_drift,1)
            ]
    
    return probs
end



# function run_control(
#         self::DigitalQuditPair, 
#         q1_control::QuditControl,
#         q2_control::QuditControl; 
#         dt=0.2
#     )
#     # Computes the terminal state Psi, for each of the qudit's current
#     # parameter settings, in response to the provided control signal

#     # Extract control variables
#     _, control_obj1 = last(q1_control.objs)
#     _, control_obj2 = last(q2_control.objs)
#     T_gate = control_obj1.tf
#     _, control_coeffs1 = last(q1_control.coeffs)
#     _, control_coeffs2 = last(q2_control.coeffs)
    
#     # Create a SchrodingerProb for each of the qudits param samples
#     probs = get_schrodinger_problems(self, T_gate, dt)  
#     n_probs = length(probs)  

#     # Run simulation for each parameter setting    
#     Psi = zeros(Complex, n_probs, N, q.Ne)
#     for j = 1:n_probs
#         state_history = eval_forward(
#                             probs[j], 
#                             [control_obj1, control_obj2], [control_coeffs1; control_coeffs2]
#                         )
#         Psi[j,:,:] = state_history[:,end,:]
#     end

#     return Psi
# end

function run_control(
    self::DigitalQuditPair,
    q1_control::QuditControl,
    q2_control::QuditControl;
    dt = 0.2
)
    _, control_obj1 = last(q1_control.objs)
    _, control_obj2 = last(q2_control.objs)

    T_gate = control_obj1.tf

    _, control_coeffs1 = last(q1_control.coeffs)
    _, control_coeffs2 = last(q2_control.coeffs)

    probs = get_schrodinger_problems(self, T_gate, dt)
    n_probs = length(probs)

    q1 = self.qudit1
    q2 = self.qudit2
    N = (q1.Ne + q1.Ng) * (q2.Ne + q2.Ng)
    Ne_total = q1.Ne * q2.Ne

    Psi = zeros(ComplexF64, n_probs, N, Ne_total)
    state_histories = zeros(ComplexF64, n_probs, N)
    for j in 1:n_probs
        state_history = eval_forward(
            probs[j],
            [control_obj1, control_obj2],
            [control_coeffs1; control_coeffs2], order = 4
        )

        Psi[j, :, :] = state_history[:, end, :]
    end

    return Psi
end


# function optimize_control(
#             self::DigitalQuditPair, 
#             gate::GateType; 
#             dt = 0.2,
#             options=["max_iter" => 100, "print_level" => 3], 
#             iter::Int = -1
#     )
#     # Optimizes the control signals for this qudit to implement 
#     # the provided 'gate'

#     q1 = self.qudit1
#     q2 = self.qudit2

#     # Target unitary: CNOT with qudit1 as the control qudit
#     n = [q1.Ne+q1.Ng, q2.Ne+q2.Ng]
#     n_ess = [q1.Ne, q2.Ne]
#     U_target = unitary(gate, [1,2], n, n_ess)

#     # Extract control variables
#     q_control = q.controls[gate]
#     max_amplitude = q_control.max_amplitude
#     iter, control_obj = last(q_control.objs)
#     T_gate = control_obj.tf
#     _, control_coeffs = last(q_control.coeffs)
    
#     # Create a SchrodingerProb for each of the qudits param samples
#     probs = get_schrodinger_problems(self, T_gate, dt)    

#     # Run the optimizer
#     opt_ret_multiple = optimize_prob(
#                             probs, [control_obj1, control_obj2], [control_coeffs1; control_coeffs2], U_target, 
#                             pcof_lbound=-max_amplitude, pcof_ubound=max_amplitude, cost_type=:Infidelity, ipopt_options=options
#                         )
#     # Save the new control coefficients
#     n_coeffs1 = length(control_coeffs1)
#     control_coeffs1 .= opt_ret_multiple.x[1:n_coeffs1]
#     control_coeffs2 .= opt_ret_multiple.x[n_coeffs1+1:end]

#     # Save Infidelity     
#     if iter == self.infidelity[gate].lastiter
#         self.infidelity[gate].values[end] = opt_ret_multiple.obj_val
#     else
#         push!(self.infidelity[gate], iter, opt_ret_multiple.obj_val)
#     end
# end

function optimize_control(
    self::DigitalQuditPair,
    gate::Union{GateType,ProductGate};
    dt = 0.2,
    options = ["max_iter" => 100, "print_level" => 3],
    max_amplitude::Union{Nothing, Float64} = nothing, 
    iter::Int64 = -1
)
    q1 = self.qudit1
    q2 = self.qudit2

    n = [q1.Ne + q1.Ng, q2.Ne + q2.Ng]
    n_ess = [q1.Ne, q2.Ne]

    U_target = unitary(gate, [1, 2], n, n_ess)

    q1_control, q2_control = self.controls[gate]

    if isnothing(max_amplitude) 
        max_amplitude = min(q1_control.max_amplitude, q2_control.max_amplitude)
    end

    iter1, control_obj1 = last(q1_control.objs)
    iter2, control_obj2 = last(q2_control.objs)

    _, control_coeffs1 = last(q1_control.coeffs)
    _, control_coeffs2 = last(q2_control.coeffs)

    T_gate = control_obj1.tf

    probs = get_schrodinger_problems(self, T_gate, dt)

    opt_ret_multiple = optimize_prob(
        probs,
        [control_obj1, control_obj2],
        [control_coeffs1; control_coeffs2],
        U_target;
        pcof_lbound = -max_amplitude,
        pcof_ubound = max_amplitude,
        cost_type = :Infidelity,
        ipopt_options = options, ridge_penalty_strength = 1e-4, objective_tol = 1E-4, order = 4
    )

    n_coeffs1 = length(control_coeffs1)

    control_coeffs1 .= opt_ret_multiple.x[1:n_coeffs1]
    control_coeffs2 .= opt_ret_multiple.x[n_coeffs1 + 1:end]

    if !haskey(self.infidelity, gate)
        self.infidelity[gate] = History(Float64)
    end

    push!(self.infidelity[gate], iter, opt_ret_multiple.obj_val)

    return opt_ret_multiple
end

function predicted_infidelity(
        q::DigitalQuditPair, 
        gate::Union{GateType,ProductGate},
        q_control1::QuditControl,
        q_control2::QuditControl; 
        dt = 0.2, 
        iter::Int = -1,
        ridge_penalty::Union{Nothing, <:Real} = nothing)
    
    q1 = q.qudit1
    q2 = q.qudit2

    n = [q1.Ne + q1.Ng, q2.Ne + q2.Ng]
    n_ess = [q1.Ne, q2.Ne]
    # N1 = q.qudit1.Ne
    # N2 = q.qudit2.Ne
    N = (q.qudit1.Ne + q.qudit1.Ng) * (q.qudit2.Ne + q.qudit2.Ng)
    Ne = q.qudit1.Ne * q.qudit2.Ne
    U_target = unitary(gate, [1, 2], n, n_ess)
    psi_final = run_control(q, q_control1, q_control2, dt = dt)
    state_infidelity = 0

    pcof_1 = get(q_control1.coeffs)
    pcof_2 = get(q_control2.coeffs)
    pcof_total = vcat(pcof_1, pcof_2)
    ridge_sum = 0
    n_probs = length(get(q.xi))
    state_infidelity_vec = zeros(n_probs)
    for i in 1:n_probs 
        # state_infidelity += infidelity(psi_final[i,:,:], U_target, size(U_target, 2))
        
        # psi_final_normal = psi_final[i,:,:]./norm.(eachcol(psi_final[i,:,:]))
        # println("psi_final 2")
        # display(psi_final_normal)
        # println("norm prob $i: ", norm.(eachcol(psi_final[i,:,:])))
        state_infidelity += 1 - (1/Ne^2)*abs(dot(psi_final[i,:,:], U_target))^2
        state_infidelity_vec[i] = 1 - (1/Ne^2)*abs(dot(psi_final[i,:,:], U_target))^2
        if !isnothing(ridge_penalty)
            ridge_sum += dot(pcof_total, pcof_total)*ridge_penalty/length(pcof_total)
        end
    end
    # println(state_infidelity/n_probs)
    # println("n probs: ", n_probs)
    # println("state infidelities: ", state_infidelity_vec)
    return (state_infidelity + ridge_sum)/n_probs
end