include("QDT.jl")
include("prior.jl")
include("gates.jl")
using Dates

"""
ObservationEvent struct

This struct logs the parameters and data generated when evaluating a control
signal on a PhysicalQudit or PhysicalQuditPair

Fields

    timestamp::DateTime             
    
        Identifier for when this control optimization was performed


    device::PhysicalDevice          
    
        e.g. a PhysicalQudit or PhysicalQuditPair


    control_coeffs::Union{Vector{Float64},Vector{Vector{Float64}}}    
                                    
        Copy of the control coefficients that were used to generate 
        this measurement.
    

    measured_populations::Array{Float64,3}

        Noisy population data read out from the device, e.g. after 
        the M_SPAM matrix and sampling error has been applied. 


    measured_populations_postprocessed::Array{Float64,3}

        Postprocessed `measured_populations` array, where each subarray
        measured_populations[i,:,j] has been shifted and rescaled s.t.
        it encodes a probability distribution, e.g. 

                    ∑ₜ measured_populations[i,t,j] = 1

        This variable defaults to undef array, with element [1,1,1] set to 
        `-1` to indicate the array has not been set. The true values will 
        set within the characterization loop.

    
    gate::Union{Nothing,GateType}

        (Optional) Gate associated with the control signal

    
    state_infidelity::Union{Nothing,Float64}

        (Optional) State infidelity associated with the 
        control signal 

    
    measured_infidelity::Union{Nothing,Float64}

        (Optional) Measured infidelity associated with the 
        control signal 


    dt::Union{Nothing,Float64}

        (Optional) Step size between data points in the 
        `measured_populations` array

"""
struct ObservationEvent
    
    timestamp::DateTime
    device::PhysicalDevice
    control_coeffs::Union{Vector{Float64},Vector{Vector{Float64}}}    
    measured_populations::Array{Float64,3}
    measured_populations_postprocessed::Array{Float64,3}
    gate::Union{Nothing,GateType}
    state_infidelity::Union{Nothing,Float64}
    measured_infidelity::Union{Nothing,Float64}
    dt::Union{Nothing,Float64}

    function ObservationEvent(
            device::PhysicalDevice,
            control_coeffs::Union{Vector{Float64},Vector{Vector{Float64}}},
            measured_populations::Array{Float64,3},
            gate::Union{Nothing,GateType}=nothing,
            state_infidelity::Union{Nothing,Float64}=nothing,
            meas_infidelity::Union{Nothing,Float64}=nothing,
            dt::Union{Nothing,Float64}=nothing
        )
        timestamp = now()
        measured_populations_postprocessed = Array{Float64,3}(undef, size(measured_populations))
        measured_populations_postprocessed[1,1,1] = -1
        new(
            timestamp, device, copy(control_coeffs), measured_populations, 
            measured_populations_postprocessed, gate, state_infidelity, 
            meas_infidelity, dt
        ) 
    end

end


"""
Determines where the postprocessed measured population data has 
been computed already.
"""
function postprocessed(event::ObservationEvent)
    return event.measured_populations_postprocessed[1,1,1] != -1
end



"""
    eval_forward(event_obs, digital_device, θ)

Performs a simulation of the experiment whose data was recorded in 
the ObservationEvent `event_obs`, but using device parameters `θ` 
rather than the true device parameters.
"""
function eval_forward(
        event_obs::ObservationEvent, 
        digital_device::DigitalDevice, 
        θ::Vector{Float64}
    )

    # Set the digit device's parameters to θ
    set_parameters(digital_device, θ)

    # Evaluate the control signals used during 
    # the ObservationEvent on the digital device
    Ψ = run_control(digital_device, event_obs.control_coeffs, dt=event_obs.dt)

    # Timestep at which Ψ was be computed
    dt = event_obs.dt
    T = get_control_time(digital_device)
    t_grid = collect(0:dt:T)

    # Convert to state populations
    return t_grid, abs2.(Ψ)
end



"""
CharacterizationEvent Struct

Logs parameters and data generated during each instance of characterization.

Fields

    timestamp::DateTime             
    
        Identifier for when this characterization was performed

    
    obs_events::Union{ObservationEvent,Vector{ObservationEvent}}

        Device data used to perform this characterization

    
    prior::Prior
    
        Prior distribution, either a UniformPrior or a KDEPrior


    n_samples::Int64

        Number of samples generated during this characterization

    
    samples::Matrix{Float}

        Parameter samples generated during this characterization.
        samples[:,i] denotes the i-th parameter sample.

    
    risk::Matrix{Float64}

        Empirical risk associated with each parameter samples.
        risk[j,i] denotes the risk associated with samples[:,i]
        on the j-th `ObservationEvent` defining the `posterior`.

    
    accept_ratio::Float64

        Proportion of ALL proposals (e.g. pre burnin and thinning)
        accepted during the MCMC loop

"""
struct CharacterizationEvent

    timestamp::DateTime
    posterior::Posterior
    n_samples::Int64
    samples::Matrix{Float64}
    risk::Matrix{Float64}
    accept_ratio::Float64

    function CharacterizationEvent(
            posterior::Posterior,    
            samples::Matrix{Float64},
            risk::Matrix{Float64},
            accept_ratio::Float64
        )
        timestamp = now()
        n_samples = size(samples,1)
        new(timestamp, posterior, n_samples, samples, risk, accept_ratio)
    end
end




"""
OptimizationEvent struct

This struct logs the parameters and data generated during each instance of 
control optimization.

Fields

    timestamp::DateTime             Identifier for when this control 
                                    optimization was performed

    gate::GateType                  Which gate was the target for the optimization

    params::Matrix{Float64}         params[:,i] denotes the i-th parameter sample.

    coeffs::Union{Vector{Float64},Vector{Vector{Float64}}}    
                                    Copy of the control coefficients after optimization.
    
    predicted_infidelities::Vector{Float64}
                                    Predicted infidelity for each parameter
                                    sample
    
    dt_opt::Int64                   Timestep used by the integrators when 
                                    optimizing the control signals

    dt_eval::Int64                  Timestep used by the integrators when
                                    evaluating predicted infidelities 
"""
struct OptimizationEvent
    
    timestamp::DateTime
    gate::GateType
    params::Matrix{Float64}
    control_coeffs::Union{Vector{Float64},Vector{Vector{Float64}}}    
    predicted_infidelities::Vector{Float64}
    dt_opt::Float64
    dt_eval::Float64

    function OptimizationEvent(
                gate::GateType,
                params::Matrix{Float64},
                control_coeffs::Union{Vector{Float64},Vector{Vector{Float64}}},
                predicted_infidelities::Vector{Float64},
                dt_opt::Float64=-1,
                dt_eval::Float64=-1
        )
        timestamp = now()
        new(timestamp, gate, params, control_coeffs, predicted_infidelities, dt_opt, dt_eval) 
    end

end






