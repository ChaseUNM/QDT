"""
DigitalQubitPair struct

Represents a pair of digital qubits. Intended for evaluation and optimization 
of control pulses. 

Fields

    omega1::Union{Float64, Vector{Float64}} 
    omega2::Union{Float64, Vector{Float64}}         
                            
        Frequency value or samples for qubits 1 and 2
        
    xi::Union{Float64, Vector{Float64}}           
                            
        Cross-kerr value or samples

    omega_rot::Float64      
    
        Rotating frame frequency

    control::AbstractControl

        Object to evaluate control amplitudes

"""
mutable struct DigitalQubitPair <: DigitalDevice
    N::Int64
    Ne::Int64
    omega1::Union{Float64, Vector{Float64}}
    omega2::Union{Float64, Vector{Float64}}
    xi::Union{Float64, Vector{Float64}}
    omega_rot::Float64
end



function DigitalQubitPair(
        omega1::Union{Float64, Vector{Float64}}, 
        omega2::Union{Float64, Vector{Float64}},
        xi::Union{Float64, Vector{Float64}},
        omega_rot::Float64
    )
    N = 4
    Ne = 4
    DigitalQubitPair(N, Ne, omega1, omega2, xi, omega_rot)
end


"""
Creates a copy of this DigitalQubitPair
"""
function Base.copy(q::DigitalQubitPair)
    return DigitalQubitPair(
        copy(q.omega1), copy(q.omega2), copy(q.xi), 
        q.omega_rot
    )
end



"""
Sets the parameters (ω₁, ω₂, ξ) of the DigitalQubitPair so they 
can be used for control evaluation and/or optimization.
"""
function set_parameters(q::DigitalQubitPair, ω₁::Float64, ω₂::Float64, ξ::Float64)
    q.omega1 = ω₁
    q.omega2 = ω₂
    q.xi     = ξ
end

function set_parameters(q::DigitalQubitPair, θ::Vector{Float64})
    q.omega1 = θ[1]
    q.omega2 = θ[2]
    q.xi     = θ[3]
end

function set_parameters(q::DigitalQubitPair, θ::Matrix{Float64})
    q.omega1 = θ[1,:]
    q.omega2 = θ[2,:]
    q.xi     = θ[3,:]
end


"""
Returns a copy of the current parameters of the DigitalQubitPair
as a single Matrix θ = [ω₁; ω₂; ξ]
"""
function get_parameters(q::DigitalQubitPair)
    if isa(q.omega1, Float64)
        return [q.omega2; q.omega2; q.xi]
    else
        return [q.omega1 q.omega2 q.xi]
    end
end



##################################################################
# SIMULATION ROUTINES
##################################################################


"""
Returns the following collections of matrices:

    H_drift         Vector of Matrices representing the drift
                    Hamiltonian (in the rotating frame) for each 
                    of q's current parameter samples

    H_c_re          Vector of Matrices representing the real part 
    H_c_im          (aₖ+aₖ') and imaginary parts (aₖ-aₖ') of the 
                    control Hamiltonians for each qubit.
"""
function get_hamiltonians(q::DigitalQubitPair)

    n = 2
    N = 4
    subsystem_sizes = [n,n]

    # Qudit parameters
    omega_rot = q.omega_rot;
    omega1    = q.omega1
    omega2    = q.omega2
    xi        = q.xi
    n_samples = length(omega1)

    # Lowering operator by subsystem
    a  = lower_op(n) 
    a1 = promote_subsys_op(a, subsystem_sizes, 1)
    a2 = promote_subsys_op(a, subsystem_sizes, 2)

    # (Unscaled) single-qubit drift Hamiltonians by subsystem
    H1 = a1' * a1
    H2 = a2' * a2

    # Unscaled coupling Hamiltonian
    H_12 = (a1'*a1)*(a2'*a2)
    
    # Full drift Hamiltonians
    H_drift = Array{Float64}(undef, n_samples, N, N)
    for j = 1:n_samples
        H_drift[j,:,:] = (
            (omega1[j] - omega_rot)   * H1 
            + (omega2[j] - omega_rot) * H2
            - xi[j] * H_12
        )
    end

    # Control Hamiltonians
    H_c_re = [a1 + a1', a2 + a2']
    H_c_im = [a1 - a1', a2 - a2']

    return H_drift, H_c_re, H_c_im
end


"""
Returns a Vector of Matrices representing the real part (aₖ+aₖ')
and imaginary parts (aₖ-aₖ') of the control Hamiltonians for each qubit.
"""
function get_control_hamiltonians(self::DigitalQubitPair)

    n = 2
    N = 4
    subsystem_sizes = [n, n]
    a = lower_op(n) 
    a1 = promote_subsys_op(a, subsystem_sizes, 1)
    a2 = promote_subsys_op(a, subsystem_sizes, 2)

    # Promote to the full system
    H_c_re = [a1 + a1', a2 + a2']
    H_c_im = [a1 - a1', a2 - a2']

    return H_c_re, H_c_im
end




"""
Returns a Vector of SchrodingerProb objects, one for each 
of this DigitalQubitPair's current parameter samples.

Argument 'T' specifies the time integration interval [0,T]
and dt is the stepsize.
"""
function get_schrodinger_problems(q::DigitalQubitPair, T, dt)
    
    N = q.N

    # Initial state
    U0 = gate_initial_states(N, N)

    # Drift Hamiltonian by parameter sample
    # and control Hamiltonians by qubit
    H_drift, H_c_re, H_c_im = get_hamiltonians(q)

    # Number of timesteps
    n_timesteps = ceil(Int, T/dt)

    # Generating the SchrodingerProb
    probs = [  SchrodingerProb(
                    H_drift[j,:,:], H_c_re, H_c_im, 
                    U0, T, n_timesteps
                ) 
                for j in axes(H_drift,1)
            ]
    return probs
end



"""
Returns the unitary for the provided `gate`, appropriately 
sized to this DigitalQubitPair

If `gate` specifies a single-qubit gate, use kwarg `which_qubit` to 
specify which qubit to apply the gate to.
"""
function unitary(q::DigitalQubitPair, gate::GateType; which_qubit::Int=-1, kwargs...)
    
    if gate in SINGLE_QUDIT_GATES
        Id = Matrix(I,2,2)
        U  = unitary(gate)
        if which_qubit==1
            return kron(U,Id)
        elseif which_qubit==2
            return kron(Id,U)
        else
            error("Tried to set which_qubit>2 for DigitalQubitPair")
        end
    
    else
        return unitary(gate, q.N, q.Ne)
    end
end



"""
Parses the control-related variables provided to run_control()
and optimize_control() for a DigitalQubitPair, adding an extra
controller with output 0 if needed for single qubit gates. 

Arguments

    controller

        AbstractControl (for single qubit gates) or 
        Vector{AbstractControl} (for two qubit gates)

    control_coeffs          
    
        Parameters (βs) of the control to execute

        When running a two-qubit gate, `control_coeffs` should be a single 
        vector of control parameters, e.g. it is the concatenation the 
        controls parameters for qubit 1 and qubit 2 into a single vector.

    which_qubit::Int        
    
        Flag indicating the qubit(s) to which the control pulse(s) will 
        be applied.

        `which_qubit=-1` to indicate the `control_coeffs` 
        represent control pulses for both qubits.

        To apply a control to only one qubit (and get input to 0 for 
        the other qubit), set `which_qubit` to 1 or 2, and have 
        `control_coeffs` contain the control coefficients for that 
        one qubit's control pulse.
"""
function parse_paired_controls(
        controller::Union{AbstractControl,Vector{AbstractControl}},
        control_coeffs::Vector{Float64},
        which_qubit::Int64
    )

    # Single-Qubit Gates
    if which_qubit > 0
        @assert which_qubit < 3
        @assert isa(controller, AbstractControl)
        @assert length(control_coeffs) == controller.N_coeff
        zero_controller = GRAPEControl(1, controller.tf);
        if which_qubit == 1
            controller_ = Vector{AbstractControl}([controller, zero_controller])
            control_coeffs_ = [0.0; 0.0; control_coeffs];
        else
            controller_ = Vector{AbstractControl}([zero_controller, controller])
            control_coeffs_ = [control_coeffs; 0.0; 0.0];
        end
        return controller_, control_coeffs_

    # Two-Qubit Gates
    else
        @assert isa(controller, Vector{AbstractControl})
        @assert length(controller) == 2
        @assert length(control_coeffs) == controller[1].N_coeff + controller[2].N_coeff
        return controller, control_coeffs
    end

end


"""
Computes the state evolution Psi(t), for each of the device's current
parameter settings, in response to the control signals generated by 
the provided `control_coeffs.` 

Arguments

    q::DigitalQubitPair         
    
        Qudit on which to run the control 

    controller

        AbstractControl (for single qubit gates) or 
        Vector{AbstractControl} (for two qubit gates)

    control_coeffs          
    
        Parameters (βs) of the control to execute

        When running a two-qubit gate, `control_coeffs` should be a single 
        vector of control parameters, e.g. it is the concatenation the 
        controls parameters for qubit 1 and qubit 2 into a single vector.

    which_qubit::Int        
    
        Flag indicating the qubit(s) to which the control pulse(s) will 
        be applied.

        By default, `which_qubit=-1` to indicate the `control_coeffs` 
        represent control pulses for both qubits.

        To apply a control to only one qubit (and get input to 0 for 
        the other qubit), set `which_qubit` to 1 or 2, and have 
        `control_coeffs` contain the control coefficients for that 
        one qubit's control pulse.

    dt::Float64             Integration step size    

"""
function run_control(
        device::DigitalQubitPair,
        controller::Union{AbstractControl, Vector{AbstractControl}},
        control_coeffs::Vector{Float64};
        which_qubit::Int=-1,
        kwargs...
    )
    controller_, control_coeffs_ = parse_paired_controls(
                                        controller, 
                                        control_coeffs, 
                                        which_qubit
                                   )
    Psi = run_control_base(device, controller_, control_coeffs_; kwargs...)
    return Psi
end


"""
Optimizes the control signals for this device to implement the 
provided gate

Arguments

    q::DigitalDevice                DigitalDevice on which to simulate
                                    the response of the device to the
                                    control signals

    controller                      AbstractControl or Vector{AbstractControl}             

    control_β0::Vector{Float64}     Initial/pre-optimized control parameters

    gate::GateType                  Which gate to optimize the controls for
    
See DigitalDevice.optimize_control() for additional keyword arguments.             
"""
function optimize_control(
        device::DigitalQubitPair,
        controller::Union{AbstractControl, Vector{AbstractControl}},
        control_coeffs::Vector{Float64},
        gate::GateType;
        which_qubit::Int=-1,
        kwargs...
    )
    controller_, control_coeffs_ = parse_paired_controls(
                                    controller, 
                                    control_coeffs, 
                                    which_qubit
                                   )
    Psi = optimize_control_base(device, controller, control_coeffs_, gate, kwargs...)
    return Psi
end




#======================================================================

    Conversion to PhysicalDevice

======================================================================#

function PhysicalDevice(digital_qs::DigitalQubitPair; M_spam_order=1e-3)

    # Verify the DigitalQubitPair has only a single parameter sample
    @assert(length(digital_qs.omega1) == 1)
    @assert(length(digital_qs.omega2) == 1)
    @assert(length(digital_qs.xi) == 1)

    # Generate the Mspam matrix
    N = Ne + Ng;
    ϵ = M_spam_order * rand(N)
    M_spam_1 = column_stochastic(ϵ)
    M_spam_2 = column_stochastic(ϵ)
    M_spam = kron(M_spam_1, M_spam_2)

    return PhysicalDevice(digital_qs, M_spam)
end

