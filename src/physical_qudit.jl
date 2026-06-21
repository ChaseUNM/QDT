using LinearAlgebra, QuantumGateDesign, Random
include("QDT.jl")
include("digital_qudit.jl")
include("measurement.jl")

########################################################################
########################################################################

mutable struct PhysicalQudit <: PhysicalDevice
    device::DigitalQudit
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
    device = DigitalQudit(Ne, Ng, omega, xi, omega_rot, control)
    
    # Generate the Mspam matrix
    N = Ne + Ng;
    ϵ = M_spam_order * rand(N)
    M_spam = column_stochastic(ϵ)

    observations = Vector{ObservationEvent}(undef, 0)

    return PhysicalQudit(device, M_spam, observations)
end


# CONSTRUCTOR V2
function PhysicalQudit(digital_q::DigitalQudit; M_spam_order=1e-3)

    # Verify the DigitalQudit has only a single parameter sample
    @assert(length(digital_q.omega) == 1)
    @assert(length(digital_q.xi) == 1)
    
    # Clone the digital qubit
    device = copy(digital_q)

    # Generate the Mspam matrix
    N = Ne + Ng;
    ϵ = M_spam_order * rand(N)
    M_spam = column_stochastic(ϵ)

    observations = Vector{ObservationEvent}(undef, 0)

    return PhysicalQudit(device, M_spam, observations)
end
