#======================================================================

    Generic PhysicalDevice

======================================================================#


function PhysicalDevice(device::DigitalDevice, M_spam::AbstractMatrix)
    device_ = copy(device)
    observations = Vector{ObservationEvent}(undef, 0)
    return PhysicalDevice(device_, M_spam, observations)
end



"""
Evaluates a control signal on a PhysicalDevice, returning an 
ObservationEvent storing the noisy population data read out 
from the device, e.g. after sampling error and (optionally)
applying the M_SPAM matrix.

Arguments

    q_physical::PhysicalDevice
    
        Device on which to run the control signals

    
    controller
    
        AbstractControl or Vector{AbstractControl}


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
        self::PhysicalDevice,
        controller::Union{AbstractControl,Vector{AbstractControl}}, 
        control_coeffs::Vector{Float64},
        n_readout_samples::Int64;
        add_SPAM::Bool=true, 
        target_gate::Union{Nothing,GateType}=nothing,
        kwargs...
    )
    
    # Run the control signals
    Psi = run_control(self.device, controller, control_coeffs; kwargs...)
    Psi = Psi[1,:,:,:]

    # Normalize states
    N = self.device.N
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
        U_target = unitary(self.device, target_gate; kwargs...)
        state_infidelity = infidelity(Psi[:,end,:], U_target, size(U_target,2))
        meas_infidelity  = infidelity_population(meas_populations[:,end,:], abs2.(U_target))
    else
        state_infidelity = nothing
        meas_infidelity  = nothing
    end
    
    # Create an ObservationEvent to store the data generated 
    # by this function.
    obs = ObservationEvent( self, controller, control_coeffs, meas_populations,
                            target_gate, state_infidelity, meas_infidelity, dt)
    push!(self.observations, obs)
    return obs
end


