##########################
# src/postprocessing.jl
##########################

# Numerically stable softplus with sharpness parameter δ:
# softplus_δ(z) = log(1 + exp(δ z))
function softplus(z, δ)
    y = δ * z
    return y > 0 ? y + log1p(exp(-y)) : log1p(exp(y))
end

function mean_center_matrix(g::AbstractMatrix)
    return g .- mean(g)
end

function mean_center_trace(v::AbstractVector)
    return v .- mean(v)
end

function simplex_map_vector(v::AbstractVector; δ::Real)
    w = softplus.(v, δ)
    s = sum(w)
    return s > 0 ? w ./ s : fill(1 / length(v), length(v))
end

# -------------------------------------------------------------------
# Sinkhorn vectorization convention
#
# Wavefields are stored physically as matrices of size (Nx, Nt),
# i.e. rows = receivers x, columns = time t.
#
# The Sinkhorn MVM operator MVM_Q assumes vectors correspond to a
# matrix of size (Nt, Nx), flattened columnwise.
#
# Therefore, before vectorization we transpose/permutedims the matrix.
# -------------------------------------------------------------------
function sinkhorn_vectorize(g::AbstractMatrix)
    # Input:  Nx x Nt
    # Output: vec of (Nt x Nx)
    return vec(permutedims(g))
end

function sinkhorn_unvectorize(v::AbstractVector, Nx::Int, Nt::Int)
    # Input:  vec of (Nt x Nx)
    # Output: Nx x Nt
    return permutedims(reshape(v, Nt, Nx))
end

function postprocess_event_sinkhorn(event; δ::Real)
    Nx, Nt = size(event.g)

    g_centered = mean_center_matrix(event.g)

    g_vec     = sinkhorn_vectorize(g_centered)
    g_neg_vec = sinkhorn_vectorize(-g_centered)

    g_prob     = simplex_map_vector(g_vec; δ=δ)
    g_prob_neg = simplex_map_vector(g_neg_vec; δ=δ)

    g_prob_matrix     = sinkhorn_unvectorize(g_prob, Nx, Nt)
    g_prob_neg_matrix = sinkhorn_unvectorize(g_prob_neg, Nx, Nt)

    return (
        g = event.g,
        g_centered = g_centered,
        g_prob = g_prob,
        g_prob_neg = g_prob_neg,
        g_prob_matrix = g_prob_matrix,
        g_prob_neg_matrix = g_prob_neg_matrix
    )
end

function postprocess_event_wasserstein(event; δ::Real)
    Nx, Nt = size(event.g)

    g_centered_traces = zeros(Nx, Nt)
    g_prob_traces = zeros(Nx, Nt)

    for i in 1:Nx
        tr = vec(event.g[i, :])
        tr_centered = mean_center_trace(tr)
        tr_prob = simplex_map_vector(tr_centered; δ=δ)

        g_centered_traces[i, :] .= tr_centered
        g_prob_traces[i, :] .= tr_prob
    end

    return (
        g = event.g,
        g_centered_traces = g_centered_traces,
        g_prob_traces = g_prob_traces
    )
end

function postprocess_event_wasserstein_quantum(data; δ::Real)
    # Inputting an order 3 tensor with dimension N_ess + N_guard, N_time, N_initial
    N_states, Nt, N_init = size(data)
    g_centered_traces = zeros(N_states, Nt, N_init)
    g_prob_traces = zeros(N_states, Nt, N_init)
    for i in 1:N_states
        for j in 1:N_init 
            tr = vec(data[i,:,j]) 
            tr_centered = mean_center_trace(tr)
            tr_prob = simplex_map_vector(tr_centered; δ=δ)

            g_centered_traces[i,:,j] .= tr_centered
            g_prob_traces[i,:,j] .= tr_prob 
        end
    end
    return ( 
        g = data, 
        g_centered_traces = g_centered_traces,
        g_prob_traces = g_prob_traces
    )
end

