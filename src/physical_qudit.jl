using LinearAlgebra, QuantumGateDesign, Random
include("QDT.jl")
include("events.jl")
include("digital_qudit.jl")
include("measurement.jl")


########################################################################
########################################################################

mutable struct PhysicalQudit <: PhysicalDevice
    qudit::DigitalQudit
    M_spam::AbstractMatrix
    observations::Vector{ObservationEvent}
end


# CONSTRUCTOR V1
function PhysicalQudit(
        Ne::Int64, 
        Ng::Int64, 
        omega::Float64, 
        xi::Float64,
        omega_rot::Float64,
        control::AbstractControl;
        M_spam_order=1e-3
    )

    # Generate the underlying DigitQudit
    qudit = DigitalQudit(Ne, Ng, omega, xi, omega_rot, control)
    
    # Generate the Mspam matrix
    N = Ne + Ng;
    ϵ = M_spam_order * rand(N)
    M_spam = column_stochastic(ϵ)

    observations = Vector{ObservationEvent}(undef, 0)

    return PhysicalQudit(qudit, M_spam, observations)
end


# CONSTRUCTOR V2
function PhysicalQudit(digital_q::DigitalQudit; M_spam_order=1e-3)

    # Verify the DigitalQudit has only a single parameter sample
    @assert(length(digital_q.omega) == 1)
    @assert(length(digital_q.xi) == 1)
    
    # Clone the digital qubit
    qudit = copy(digital_q)

    # Generate the Mspam matrix
    N = Ne + Ng;
    ϵ = M_spam_order * rand(N)
    M_spam = column_stochastic(ϵ)

    observations = Vector{ObservationEvent}(undef, 0)

    return PhysicalQudit(qudit, M_spam, observations)
end



"""
Evaluates a control signal on a PhysicalQudit, returning an 
ObservationEvent storing the noisy population data read out 
from the device, e.g. after sampling error and (optionally)
applying the M_SPAM matrix.

Arguments

    q_physical::PhysicalQudit
    
        Qudit on which to run the control signals


    control_coeffs::Vector{Float64}

        Control coefficients, e.g. the βs
    

    n_readout_samples::Int64

        Number of "shots" / readout samples when estimating
        the population data <0|ψ(t)> from the simulated state
        evolution of the PhysicalQudit
    

    add_SPAM::Bool

        (Optional) Flag. Set to true to add SPAM errors when
        performing the population readout
        Default: true


    target_gate::Union{Nothing,GateType}
    
        (Optional) If set to a GateType, the populations of
        the final state Ψ(T) will be compared to the unitary 
        associated with the gate, resulting in state and measured 
        infidelity scores being stored in the ObservationEvent 
        returned by this function.


    dt::Float64

        (Optional) Integrator step size when running the control.
        Default: 0.2

        
"""
@views function run_control(
        self::PhysicalQudit, 
        control_coeffs::Vector{Float64},
        n_readout_samples::Int64;
        add_SPAM::Bool=true, 
        target_gate::Union{Nothing,GateType}=nothing,
        dt::Float64=0.2
    )
    
    # Run the control signals
    Psi = run_control(self.qudit, control_coeffs, dt=dt)
    Psi = Psi[1,:,:,:]

    # Normalize states
    N = self.qudit.N
    mat_Psi = reshape(Psi, N, :)
    mat_Psi ./= transpose(norm.(eachcol(mat_Psi)))

    # Apply SPAM error?
    meas_populations = abs2.(Psi)
    if add_SPAM
        tmp = reshape(meas_populations, N, :)
        tmp .= self.M_spam * tmp
    end

    # Sampling the quantum state
    meas_populations = sample_quantum_state(n_readout_samples, meas_populations)

    # Calculate gate infidelities?
    if !isnothing(target_gate)
        N = self.qudit.N
        U_target = unitary(target_gate, N)
        state_infidelity = infidelity(Psi[:,end,:], U_target, size(U_target,2))
        meas_infidelity  = infidelity_population(meas_populations[:,end,:], abs2.(U_target))
    else
        state_infidelity = nothing
        meas_infidelity  = nothing
    end
    
    # Create an ObservationEvent to store the data generated 
    # by this function.
    obs = ObservationEvent( self, control_coeffs, meas_populations,
                            target_gate, state_infidelity, meas_infidelity, dt)
    push!(self.observations, obs)
    return obs
end