using LinearAlgebra, Random, Distributions

"""
Creates an column stochastic matrix M[i,j] with diagonal elements
M[i,i] = 1 - ϵ[i], i = 1, 2, ..., length(ϵ), and off diagonal 
elements M[i,j] ~ [0,1] and then normalized such that Σ_j M[j,i] = 1
"""
function column_stochastic(ϵ::Array{Float64})
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



function sample_quantum_state(n_samples, p::Array{T,3}) where T
    p_obs = zeros(size(p))
    for j in axes(p,2), k in axes(p,3)
        p_distribution = Multinomial(n_samples, p[:,j,k])
        p_obs[:,j,k] = rand(p_distribution)/n_samples 
    end
    return p_obs 
end
