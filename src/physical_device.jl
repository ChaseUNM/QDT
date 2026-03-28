using LinearAlgebra, QuantumGateDesign, ValueHistories, Random, Distributions
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



function sample_quantum_state(samples, p)
    p_obs = zeros(size(p))
    for i in 1:size(p, 1)
        p[:,i] = p[:,i]/sum(p[:,i])
        p_distribution = Multinomial(samples, p[:,i])
        N_shots = rand(p_distribution)
        p_obs[:,i] = N_shots/samples 
    end
    return p_obs 
end


########################################################################
########################################################################

mutable struct PhysicalQudit

    # FIELDS
    qudit::DigitalQudit
    M_spam::AbstractMatrix
    measured_state_infidelity::Dict{GateType, History{Int, Float64}}
    measured_population_infidelity::Dict{GateType, History{Int, Float64}}
    
    # CONSTRUCTOR
    function PhysicalQudit(
                Ne::Int64, Ng::Int64, 
                omega::Float64, xi::Float64;
                M_spam_order=1e-3
        )

        # Generate the underlying Digitqudit
        qudit = DigitalQudit(Ne, Ng)
        add_param_samples(qudit, [omega], [xi])

        # Generate the Mspam matrix
        N = Ne + Ng;
        ϵ = M_spam_order * rand(N)
        M_spam = column_stochastic(ϵ)

        # Fidelity histories
        measured_state_infidelity = Dict{GateType, History{Int64, Float64}}()
        measured_population_infidelity = Dict{GateType, History{Int64, Float64}}()

        new(
            qudit, 
            M_spam, 
            measured_state_infidelity, 
            measured_population_infidelity
        )

    end

end



function measure_infidelity(
        q::PhysicalQudit, 
        gate::GateType, 
        q_control::QuditControl, 
        n_readout_samples::Int64;
        add_SPAM=true,
        dt=0.2
    )
    # Computes the measured infidelities (state and population) of this qudit
    # for the given gate using the provided control signals

    # Target unitary
    N = q.qudit.Ne + q.qudit.Ng
    U_target = unitary(gate, N)

    # Run the controls to get the final state
    psi_final = run_control(q.qudit, q_control, dt=dt)
    psi_final = psi_final[1,:,:]
    # Normalize 
    psi_final = psi_final ./ norm.(eachcol(psi_final)) 

    # State infidelity
    state_infidelity = infidelity(psi_final, U_target, size(U_target,2))

    # Population infidelity
    observed_populations = abs2.(psi_final)
    if add_SPAM
        observed_populations = sample_quantum_state(
                                    n_readout_samples, 
                                    q.M_spam * observed_populations
                                )
    end
    population_infidelity = infidelity_population(
                                observed_populations, 
                                abs2.(U_target)
                            )

    # Return or state to q?
    return (state_infidelity, population_infidelity)
end
