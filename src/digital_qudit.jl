using LinearAlgebra, QuantumGateDesign
include("QDT.jl")
include("digital_device.jl")

"""
DigitalQudit struct

Represents a single digital qudit. Intended for evaluation and optimization 
of control pulses. 

Fields

    Ne:Int64                Number of essential energy levels

    Ng::Int64               Number of guard levels

    N::Int64                Total number of energy levels

    omega::Union{Float64, Vector{Float64}}        
                            
        Qudit frequency value or samples
        
    xi::Union{Float64, Vector{Float64}}           
                            
        Qudit self-kerr value or samples

    omega_rot::Float64      
    
        Rotating frame frequency

"""
mutable struct DigitalQudit <: DigitalDevice

    Ne::Int64
    Ng::Int64
    N::Int64

    omega::Union{Float64, Vector{Float64}}
    xi::Union{Float64, Vector{Float64}}
    
    omega_rot::Float64

    function DigitalQudit(
            Ne::Int64, 
            Ng::Int64, 
            omega::Union{Float64, Vector{Float64}}, 
            xi::Union{Float64, Vector{Float64}},
            omega_rot::Float64
        )
        N = Ne + Ng
        new(Ne, Ng, N, omega, xi, omega_rot)
    end
end


"""
Creates a copy of this DigitalQudit
"""
function Base.copy(q::DigitalQudit)
    return DigitalQudit(
        q.Ne, q.Ng, 
        copy(q.omega), copy(q.xi), 
        q.omega_rot
    )
end


"""
Sets the parameters (ω, ξ) of the DigitalQudit so they 
can be used for control evaluation and/or optimization.
"""
function set_parameters(q::DigitalQudit, ω::Float64, ξ::Float64)
    q.omega = ω
    q.xi    = ξ
end

function set_parameters(q::DigitalQudit, θ::Vector{Float64})
    q.omega = θ[1]
    q.xi    = θ[2]
end

function set_parameters(q::DigitalQudit, ω::Vector{Float64}, ξ::Vector{Float64})
    q.omega = copy(ω)
    q.xi    = copy(ξ)
end

function set_parameters(q::DigitalQudit, θ::Matrix{Float64})
    q.omega = θ[1,:]
    q.xi    = θ[2,:]
end


"""
Returns a copy of the current parameters of the DigitalQudit
as a single Matrix θ = [ω; ξ]
"""
function get_parameters(q::DigitalQudit)
    if isa(q.omega, Float64)
        return [q.omega; q.xi]
    else
        return [q.omega q.xi]
    end
end



########################################################################
# SIMULATION ROUTINES
########################################################################

"""
Returns a Vector of Matrices representing the drift
Hamiltonian (in the rotating frame) for each of this 
qudit's current parameter samples
"""
function get_drift_hamiltonians(q::DigitalQudit)

    # Set unscaled drift Hamiltonian
    a = lower_op(q.N) 
    H_omega = a' * a 
    H_xi = a' * a' * a * a;

    # Qudit parameters
    omega_rot = q.omega_rot;
    omega = q.omega
    xi = q.xi
    n_samples = length(omega)

    # Scaled Hamiltonians
    H_drift = [
        ((omega[j]-omega_rot)*H_omega) .- (0.5*xi[j]*H_xi)
        for j = 1:n_samples
    ]
    
    return H_drift
end


"""
Returns the (unscaled) control Hamiltonians a + a'
and a-a' for this qudit
"""
function get_control_hamiltonians(q::DigitalQudit)
    a = lower_op(q.N) 
    H_c_re = a + a';
    H_c_im = a - a';
    return H_c_re, H_c_im
end


"""
Returns a Vector of SchrodingerProb objects, one for each 
of this qudit's current parameter samples.

Argument 'T' specifies the time integration interval [0,T]
and dt is the stepsize.
"""
function get_schrodinger_problems(q::DigitalQudit, T, dt)
    
    # Initial state
    U0 = gate_initial_states(q.N, q.Ne)

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
                for j in eachindex(H_drift)
            ]
    
    return probs
end
