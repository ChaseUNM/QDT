
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
    ω_rot::Float64                     # Rotating frame frequency
    ω::History{Int64, Vector{Float64}} # History of qubit frequency samples
    ξ::History{Int64, Vector{Float64}} # History of qubit self-kerr samples
    controls::Dict{GateType, QuditControl}
    infidelity::Dict{GateType, History{Int, Float64}}

    # CONSTRUCTOR
    function DigitalQudit(Ne, Ng)
        ω = History(Vector{Float64})
        ξ = History(Vector{Float64})
        ω_rot = 0
        controls = Dict{GateType, QuditControl}()
        infidelity = Dict{GateType, History{Int, Float64}}()
        new(Ne, Ng, ω_rot, ω, ξ, controls, infidelity)
    end

end


function add_param_samples(
                    q::DigitalQudit, 
                    ω::Vector{Float64}, 
                    ξ::Vector{Float64};
                    iter::Int64=-1
    )
    # Appends a sample of frequencies and self-kerr coefficients 
    # to this qudit's history with timestamp 'iter'
    #
    # By default, 'iter' is set to the latest iteration + 1
    # 
    push!(q.ω, iter, ω)
    push!(q.ξ, iter, ξ)
    q.ω_rot = mean(ω)
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



function run_control(q::DigitalQudit, q_control::QuditControl; Δt=0.2)
        
    # Initial states
    N = q.Ne + q.Ng
    U0 = Matrix(1.0*I, N, q.Ne)

    # Set unscaled drift and control Hamiltonian
    a = lower_op(N) 
    H_ω = a' * a 
    H_ξ = a' * a' * a * a;
    H_c_re = a + a';
    H_c_im = a - a';

    # Extract control param
    iter, control_obj = last(q_control.objs)
    T_gate = control_obj.tf
    _, control_coeffs = last(q_control.coeffs)
    
    # Extract qudit parameter
    ω_rot = q.ω_rot;
    _, ω = last(q.ω)
    _, ξ = last(q.ξ)
    n_samples = length(ω)
    n_timesteps = ceil(Int, T_gate/Δt)

    # Run simulation for each parameter setting    
    Psi = zeros(Complex, n_samples, N, q.Ne)
    for j = 1:n_samples
        prob = SchrodingerProb(
                    ((ω[j]-ω_rot)*H_ω) - (0.5*ξ[j]*H_ξ), 
                    [H_c_re], [H_c_im], 
                    U0, T_gate, n_timesteps
                ) 
        state_history = eval_forward(prob, control_obj, control_coeffs)
        Psi[j,:,:] = state_history[:,end,:]
    end

    return Psi
end



function optimize_control(
            q::DigitalQudit, 
            gate::GateType; 
            Δt = 0.2,
            options=["max_iter" => 100, "print_level" => 3] 
    )
    # Optimizes the control signals for this qudit to implement 
    # the provided 'gate'

    # Initial states
    N = q.Ne+q.Ng
    U0 = Matrix(1.0*I, N, q.Ne)

    # Select the gate  
    U_target = unitary(gate, N);  

    # Set unscaled drift and control Hamiltonian
    a = lower_op(N) 
    H_ω = a' * a 
    H_ξ = a' * a' * a * a;
    H_c_re = a + a';
    H_c_im = a - a';

    # Extract control param
    q_control = q.controls[gate]
    max_amplitude = q_control.max_amplitude
    iter, control_obj = last(q_control.objs)
    T_gate = control_obj.tf
    _, control_coeffs = last(q_control.coeffs)
    
    # SchrodingerProb for each qudit param sample
    ω_rot = q.ω_rot;
    _, ω = last(q.ω)
    _, ξ = last(q.ξ)
    n_samples = length(ω)
    n_timesteps = ceil(Int, T_gate/Δt)
    probs = [  SchrodingerProb(
                    ((ω[j]-ω_rot)*H_ω) .- (0.5*ξ[j]*H_ξ), 
                    [H_c_re], [H_c_im], 
                    U0, T_gate, n_timesteps
                ) 
                for j in 1:n_samples
            ]

    # Run the optimizer
    opt_ret_multiple = optimize_prob(
                            probs, control_obj, control_coeffs, U_target, pcof_lbound=-max_amplitude, pcof_ubound=max_amplitude, cost_type=:Infidelity, ipopt_options=options
                        )

    # Save results     
    push!(q.infidelity[gate], iter, opt_ret_multiple.obj_val)
    control_coeffs .= opt_ret_multiple.x

end





########################################################################
# QUANTUM DIGITAL DEVICE
########################################################################

mutable struct DigitalDevice

    # FIELDS

    
    # CONSTRUCTOR

end