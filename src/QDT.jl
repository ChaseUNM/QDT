using LinearAlgebra, QuantumGateDesign

include("gates.jl")
abstract type DigitalDevice end
abstract type AbstractEvent end
abstract type Domain end
abstract type Prior end
abstract type Posterior end

struct PhysicalDevice
    device::DigitalDevice
    M_spam::AbstractMatrix
    observations::Vector{AbstractEvent}
end

include("events.jl")
include("measurement.jl")
include("physical_device.jl")

include("digital_device.jl")
include("digital_qudit.jl")
include("digital_qubit_pair.jl")

include("prior.jl")
include("posterior.jl")
include("characterization.jl")