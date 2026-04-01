
using LinearAlgebra, QuantumGateDesign, ValueHistories
include("gates.jl")
include("controls.jl")
include("util.jl")

########################################################################
# DigitalQudit struct
########################################################################

mutable struct DigitalQudit

    # FIELDS
    Ne::Int64                          # Number of essential energy levels
    Ng::Int64                          # Number of guard levels
    omega_rot::Float64                     # Rotating frame frequency
    omega::History{Int64, Vector{Float64}} # History of qubit frequency samples
    xi::History{Int64, Vector{Float64}} # History of qubit self-kerr samples
    controls::Dict{GateType, QuditControl}
    infidelity::Dict{GateType, History{Int, Float64}}

    # CONSTRUCTOR
    function DigitalQudit(Ne, Ng)
        omega = History(Vector{Float64})
        xi = History(Vector{Float64})
        omega_rot = 0
        controls = Dict{GateType, QuditControl}()
        infidelity = Dict{GateType, History{Int, Float64}}()
        new(Ne, Ng, omega_rot, omega, xi, controls, infidelity)
    end

end


########################################################################
# SETTING/UPDATING PARAMETERS
########################################################################


function num_samples(q::DigitalQudit)
    # Returns the number of parameter samples at the latest 
    # timestamp/iteration in this qudit's history
    n_samples_omega = length(q.omega.values[end])
    n_samples_xi    = length(q.xi.values[end])
    # Verify the number of samples are identical
    if n_samples_omega != n_samples_xi
        throw("DigitalQudit::Number of omega and xi samples do no match!")
    end
    return n_samples_omega
end


function add_param_samples(
                    q::DigitalQudit, 
                    omega::Vector{Float64}, 
                    xi::Vector{Float64};
                    iter::Int64=-1
    )
    # Appends a sample of frequencies and self-kerr coefficients 
    # to this qudit's history with timestamp 'iter'
    #
    # By default, 'iter' is set to the latest iteration + 1
    # 
    push!(q.omega, iter, omega)
    push!(q.xi, iter, xi)
    q.omega_rot = mean(omega)
end

function add_param_samples(q::DigitalQudit, 
                           omega::Float64, xi::Float64; 
                           kwargs...
    )
    # Overloaded version of the above function to work with 
    # omega and xi as floats rather than vectors of floats
    add_param_samples(q, [omega], [xi], kwargs...)
end



function update_param_samples(
                    q::DigitalQudit, 
                    omega::Vector{Float64}, 
                    xi::Vector{Float64}
    )
    # Updates the sample frequencies and self-kerr coefficients 
    # of this qudit's history at the latest timestamp in its history.
    #
    # Throws an error if the parameter histories are empty. 
    #

    if length(q.omega) == 0 || length(q.xi) == 0
        throw("Cannot update parameters of empty qudit history!")
    end

    iter = q.omega.lastiter
    update!(q.omega, iter, omega)
    update!(q.xi, iter, xi)
    q.omega_rot = mean(omega)
end



function add_control(q::DigitalQudit, gate::GateType, 
                     control_obj::AbstractControl; iter::Int64=-1)
    # Adds an entry to this qudits control history for the provided gate.
    # If timestamp 'iter' isn't provided, the control is added to the 
    # history at 1 + {the latestest control timestamp}
    if !haskey(q.controls, gate)
        q.controls[gate] = QuditControl(control_obj)
        randomize_coeffs(q.controls[gate])
    else
        push!(q.controls[gate], iter, c)
    end
end



########################################################################
# SIMULATION ROUTINES
########################################################################

function get_drift_hamiltonians(q::DigitalQudit)
    # Returns a Vector of Matrices representing the drift
    # Hamiltonian (in the rotating frame) for each of this 
    # qudit's current parameter samples

    N = q.Ne + q.Ng

    # Set unscaled drift Hamiltonian
    a = lower_op(N) 
    H_omega = a' * a 
    H_xi = a' * a' * a * a;

    # Qudit parameters
    omega_rot = q.omega_rot;
    _, omega = last(q.omega)
    _, xi = last(q.xi)
    n_samples = length(omega)

    # Scaled Hamiltonians
    H_drift = [
        ((omega[j]-omega_rot)*H_omega) .- (0.5*xi[j]*H_xi)
        for j = 1:n_samples
    ]
    
    return H_drift
end


function get_control_hamiltonians(q::DigitalQudit)
    # Returns the (unscaled) control Hamiltonians a + a'
    # and a-a' for this qudit
    N = q.Ne + q.Ng
    a = lower_op(N) 
    H_c_re = a + a';
    H_c_im = a - a';
    return H_c_re, H_c_im
end


function get_schrodinger_problems(q::DigitalQudit, T, dt)
    # Returns a Vector of SchrodingerProb objects, one for each 
    # of this qudit's current parameter samples.
    # Argument 'T' specifies the time integration interval [0,T]
    # and dt is the stepsize
    #

    # Initial state
    U0 = gate_initial_states(N, q.Ne)

    # Drift Hamiltonian by parameter sample
    H_drift = get_drift_hamiltonians(q)

    # Unscaled control Hamiltonian
    H_c_re, H_c_im = get_control_hamiltonians(q)

    # Number of timesteps
    n_timesteps = ceil(Int, T/dt)

    # Generating the SchrodingerProb
    probs = [  SchrodingerProb(
                    H_drift[j], [H_c_re], [H_c_im], 
                    U0, T, n_timesteps
                ) 
                for j in 1:length(H_drift)
            ]
    
    return probs
end



function run_control(q::DigitalQudit, q_control::QuditControl; dt=0.2)
    # Computes the terminal state Psi, for each of the qudit's current
    # parameter settings, in response to the provided control signal

    iter, control_obj = last(q_control.objs)
    T_gate = control_obj.tf
    _, control_coeffs = last(q_control.coeffs)
    
    # Create a SchrodingerProb for each of the qudits param samples
    probs = get_schrodinger_problems(q, T_gate, dt)  
    n_probs = length(probs)  

    # Run simulation for each parameter setting    
    Psi = zeros(Complex, n_probs, N, q.Ne)
    for j = 1:n_probs
        state_history = eval_forward(probs[j], control_obj, control_coeffs)
        Psi[j,:,:] = state_history[:,end,:]
    end

    return Psi
end



function optimize_control(
            q::DigitalQudit, 
            gate::GateType; 
            dt = 0.2,
            options=["max_iter" => 100, "print_level" => 3] 
    )
    # Optimizes the control signals for this qudit to implement 
    # the provided 'gate'

    N = q.Ne + q.Ng
    U_target = unitary(gate, N)

    # Extract control variables
    q_control = q.controls[gate]
    max_amplitude = q_control.max_amplitude
    iter, control_obj = last(q_control.objs)
    T_gate = control_obj.tf
    _, control_coeffs = last(q_control.coeffs)
    
    # Create a SchrodingerProb for each of the qudits param samples
    probs = get_schrodinger_problems(q, T_gate, dt)    

    # Run the optimizer
    opt_ret_multiple = optimize_prob(
                            probs, control_obj, control_coeffs, U_target, pcof_lbound=-max_amplitude, pcof_ubound=max_amplitude, cost_type=:Infidelity, ipopt_options=options
                        )

    # Save results     
    push!(q.infidelity[gate], iter, opt_ret_multiple.obj_val)
    control_coeffs .= opt_ret_multiple.x

end

