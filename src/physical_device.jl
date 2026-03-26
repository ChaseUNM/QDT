using LinearAlgebra, QuantumGateDesign, ValueHistories
include("digital_device.jl")

########################################################################
########################################################################

function column_stochastic(ϵ::Array{Float64})
    # Creates an column stochastic matrix M[i,j] with diagonal elements
    # M[i,i] = 1 - ϵ[i], i = 1, 2, ..., length(ϵ), and off diagonal 
    # elements M[i,j] ~ [0,1] and then normalized such that Σ_j M[j,i] = 1
    
    N = length(ϵ)
    M = rand(N, N)   # Start with all random numbers in [0,1]

    for i in 1:N
        # Put the diagonal element at position 1 initially 
        M[1,i] = 1 - ϵ[i]
        # Normalize elements 2:N
        M[2:N,i] *= ϵ[i]/sum(M[2:N,i])
        # Swap the diagonal element into its correct position now
        M[1,i], M[i,i] = M[i,i], M[1,i]
    end
    return M 
end


function infidelity_population(p, q)
    A = (sum(sqrt.(p) .* sqrt.(q), dims = 1)).^2
    return 1 - mean(A)
end


########################################################################
########################################################################

mutable struct PhysicalQudit <: DigitalDevice 

    qubit::DigitalQudit
    M_spam::AbstractMatrix
    n_readout_samples::Int64
    measured_state_infidelity::Dict{GateType, History{Int, Float64}}
    measured_population_infidelity::Dict{GateType, History{Int, Float64}}
    
    function PhysicalQudit(qubit::DigitalQudit; M_spam_order=1e-3, n_readout_samples=1000)

        # Generate the Mspam matrix
        N = Ne + Ng;
        ϵ = M_spam_order * rand(N)
        M_spam = column_stochastic(ϵ)

        # Fidelity histories
        measured_state_infidelity = Dict{GateType, History{Int64, Float64}}()
        measured_population_infidelity = Dict{GateType, History{Int64, Float64}}()

        new(qubit, M_spam, n_readout_samples, measured_state_infidelity, measured_population_infidelity)

    end

end


function measured_infidelity(
        q::PhysicalQudit, 
        gate::GateType, 
        q_control::QubitControl, 
        add_SPAM=true
    )
    # Computes the measured infidelities (state and population) of this qubit
    # for the given gate using the provided control signals

    # Target unitary
    N = q.qubit.Ne + q.qubit.Ng
    U_target = unitary(gate, N)

    # Run the controls to get the final state
    psi_final = run_control(q.qubit, q_control)
    psi_final = psi_final[1,:,:]

    # State infidelity
    state_infidelity = infidelity(psi_final, U_target, size(U_target,2))

    # Population infidelity
    if add_SPAM
        p_hat = q.M_spam * psi_final
        observed_final_state = sample_quantum_pops(q.n_readout_samples, p_hat)
    else
        observed_final_state = abs2.(psi_final)
    end
    population_infidelity = infidelity_population(observed_final_state, abs.(U_target))

    # Return or state to q?
    return (state_infidelity, population_infidelity)
end
