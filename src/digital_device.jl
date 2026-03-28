
using LinearAlgebra, QuantumGateDesign, ValueHistories


########################################################################
# GATE TYPES
########################################################################

@enum GateType PauliX PauliY PauliZ Hadamard CNOT

function unitary(gate::GateType)
    # Returns the unitary associated with the gate
    # as a 2x2 or 4x4 matrix
    if gate == PauliX
        return PauliX_gate()
    elseif gate == PauliY
        return PauliY_gate()
    elseif gate == PauliZ
        return PauliZ_gate()
    elseif gate == Hadamard
        return Hadamard_gate()
    elseif gate == CNOT
        return CNOT_gate()
    else
        throw("GateType::unitary() You shouldnt be here?")
    end
end


function unitary(gate::GateType, N::Int64)
    # Returns the unitary associated with the gate as an NxN matrix
    Ug = unitary(gate)
    N_gate = size(Ug,1)
    if N_gate < N
        U = zeros(eltype(Ug), N,N_gate)
        U[1:N_gate,1:N_gate] = Ug
    else
        U = Ug
    end
    return U
end


########################################################################
# QUDIT CONTROL
########################################################################

mutable struct QuditControl

    ### FIELDS
    max_amplitude::Float64
    objs::History{Int64, AbstractControl}
    coeffs::History{Int64, Vector{Float64}}
    
    ### CONSTRUCTORS

    function QuditControl(;max_amplitude=0.1)
        objs = History(AbstractControl)
        coeffs = History(Vector{Float64})
        new(max_amplitude, objs, coeffs)
    end

    function QuditControl(c::AbstractControl; iter=0, max_amplitude=0.1)
        objs = History(AbstractControl)
        push!(objs, c)
        coeffs = History(Vector{Float64})
        new(max_amplitude, objs, coeffs)
    end

end   


function lastiter(c::QuditControl)
    # Returns the timestamp of the last control logged in 
    # this object's history
    return max(c.objs.lastiter, 0) 
end



function randomize_coeffs(c::QuditControl)
    # Sets the spline coefficients of this control the values
    # drawn from U[-0.5,0.5] * max_amplitude

    # Verify the control already has an entry that we can
    # build off of
    if length(c.coeffs) == 0 && length(c.objs) == 0
        throw("Empty QubitControl cannot be randomized")
    end

    iter, control_obj = last(c.objs)
    N_coeff = control_obj.N_coeff

    # Generate the first set of random coefficients
    if c.coeffs.lastiter < iter
        coefs = (0.5 .- rand(N_coeff)) .* c.max_amplitude
        push!(c.coeffs, iter, coefs)

    # Randomize existing coefficients
    else 
        coefs = c.coeffs.values[end]
        coefs .= (0.5 .- rand(N_coeff)) .* c.max_amplitude
    end
    
end



function randomize(iter::Int64, c::QuditControl)
    # Sets the spline coefficients of this control the values
    # drawn from U[-0.5,0.5] * max_amplitude

    # Verify the control already has an entry that we can
    # build off of
    if length(c.coeffs) == 0
        throw("Empty QubitControl cannot be initialized")
    end

    # Pointer to the previous control object
    last_obj = last(c.objs)
    push!(c, iter, last_obj)

    # Randomize the coefficients
    N_coeff = size(last(c.spline_coeffs))
    new_coeffs = (0.5 .- rand(N_coeff)) .* c.max_amplitude
    push!(c.coeffs, iter, new_coeffs) 

end



########################################################################
# QUDITs themselves
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



function update_param_samples(
                    q::DigitalQudit, 
                    omega::Vector{Float64}, 
                    xi::Vector{Float64}
    )
    # Updates the sample frequencies and self-kerr coefficients 
    # to this qudit's history at the latest timestamp in its history.
    #
    # Throws an error if the parameter histories are empty. 
    #

    if length(q.omega) == 0 || length(q.xi) == 0
        throw("Cannot update parameters of empty qudit history!")
    end

    iter = q.omega.lastiter
    push!(q.omega, iter, omega)
    push!(q.xi, iter, xi)
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


function get_schrodinger_problems(q::DigitalQudit, T, dt)
    # Returns a Vector of SchrodingerProb objects, one for each 
    # of this qudit's current parameter samples.
    # Argument 'T' specifies the time integration interval [0,T]
    # and dt is the stepsize
    #

    # Initial state
    N = q.Ne + q.Ng
    U0 = Matrix(1.0*I, N, q.Ne)

    # Set unscaled drift and control Hamiltonian
    a = lower_op(N) 
    H_omega = a' * a 
    H_xi = a' * a' * a * a;
    H_c_re = a + a';
    H_c_im = a - a';
    
    # Qudit parameters
    omega_rot = q.omega_rot;
    _, omega = last(q.omega)
    _, xi = last(q.xi)
    n_samples = length(omega)

    # Number of timesteps
    n_timesteps = ceil(Int, T_gate/dt)

    # Generating the SchrodingerProb
    probs = [  SchrodingerProb(
                    ((omega[j]-omega_rot)*H_omega) .- (0.5*xi[j]*H_xi), 
                    [H_c_re], [H_c_im], 
                    U0, T, n_timesteps
                ) 
                for j in 1:n_samples
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

    U_target = unitary(gate, q.Ne)

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





########################################################################
# PAIRS OF DIGITAL QUDITS
########################################################################

mutable struct DigitalQuditPair

    # FIELDS
    qudit1::DigitalQudit
    qudit2::DigitalQudit
    xi::History{Int64,Float64}
    J::History{Int64,Float64}
    controls::Dict{GateType, QuditControl}
    infidelity::Dict{GateType, History{Int, Float64}}
    
    # CONSTRUCTOR

end