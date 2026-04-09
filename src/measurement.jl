using LinearAlgebra, Random, Distributions
include("qdt.jl")

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



function infidelity_state(p, q)
    A = abs2.(sum(conj(p).*q, dims = 1))
    return 1 - mean(A)
end




function sample_quantum_pops(samples, p)
    p_obs = zeros(size(p))
    for i in 1:size(p, 1)
        p[:,i] = p[:,i]/sum(p[:,i])
        p_distribution = Multinomial(samples, p[:,i])
        N_shots = rand(p_distribution)
        p_obs[:,i] = N_shots/samples 
    end
    return p_obs 
end




function measured_infidelity(qdt::QDT, true_parameter; add_SPAM=true)
    #
    #
    #
    #
    real_control_ops = [0 1; 1 0]
    imag_control_ops = [0 1; -1 0]
    N_gates = qdt.N_gates 
    gate_set = qdt.gate_set 
    steps = qdt.steps
    controls = qdt.controls 
    T_gate = qdt.T_gate
    pcof_by_gate = qdt.pcof_by_gate
    prod_states = prod(qdt.Ng .+ qdt.Ne)
    #Change this to be somewhere else
    U0 = Matrix(1.0*I, Ne[1] + Ng[1], Ne[1] + Ng[1])
    # measured_population_infidelity = zeros(N_gates)
    # measured_state_infidelity = zeros(N_gates)
    for i in 1:N_gates
        true_problem = SchrodingerProb([0 0; 0 true_parameter - qdt.omega_rot], [real_control_ops], [imag_control_ops], U0, T_gate[i], steps[i])
        meas_state_history = eval_forward(true_problem, controls[i], pcof_by_gate[i])
        final_state = meas_state_history[:,end,:]
        # Normalize the columns
        final_state = final_state ./ norm.(eachcol(final_state))'

        final_state_abs = abs2.(final_state)
        final_gate_unitary = gate_set[i]*U0 
        
        final_gate_abs = abs2.(final_gate_unitary)
        
        # Add readout noise?
        if add_SPAM
            M_spam_i = qdt.Mspam_matrices[1]
            p_hat = M_spam_i*final_state_abs
            observed_final_state = sample_quantum_pops(qdt.readout_samples, p_hat)
        else
            observed_final_state = final_state_abs
        end
        
        qdt.measured_population_infidelity[i] = infidelity_population(observed_final_state, final_gate_abs)
        qdt.measured_state_infidelity[i] = infidelity(
                                                    observed_final_state, 
                                                    final_gate_abs, 
                                                    size(U0,2)
                                                )

    end
end