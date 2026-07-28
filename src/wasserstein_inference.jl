##################################
# src/wasserstein_inference.jl
##################################

using Random
using Statistics
using Distributions
using MCMCDiagnosticTools

# ------------------------------------------------------------
# Prior on θ only
# ------------------------------------------------------------
function log_prior_theta_w2(
    θ::Real;
    θmin::Real,
    θmax::Real
)   # do this element-wise for the input vector and then take the product
    if !(θmin <= θ <= θmax)
        return -Inf
    end
    return log(pdf(Uniform(θmin, θmax), θ))
end


function log_prior_theta_w2_quantum(
    ω::Real;
    ωmin::Real,
    ωmax::Real,
    prior::Union{Nothing, Distribution} = nothing
)
    if !(ωmin <= ω <= ωmax)
        return -Inf
    end
    if isnothing(prior)
        return log(pdf(Uniform(ωmin, ωmax), ω))
    else
        return log(pdf(truncated(prior, lower = ωmin, upper = ωmax), ω))
    end
end

function log_prior_theta_w2_quantum_adaptive(
    ω::Real,
    λ_log::Real; 
    ωmin::Real, 
    ωmax::Real,
    λ_log_min::Real = log(1.0001),
    λ_log_max::Real = log(50.0),
    ω_prior::Union{Nothing, Distribution} = nothing, 
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3
)   
    if !(ωmin <= ω <= ωmax) || !(λ_log_min <= λ_log <= λ_log_max)
        return -Inf
    end
    # lp = 0.0
    # if isnothing(ω_prior)
    #     lp += log(pdf(Uniform(ωmin, ωmax), ω))
    # else
    #     lp += log(pdf(truncated(ω_prior, lower = ωmin, upper = ωmax), ω))
    # end 
    # if isnothing(λ_log_prior)
    #     lp += log(pdf(Uniform(λ_log_min, λ_log_max), λ_log))
    # else
    #     lp += log(pdf(truncated(λ_log_prior, λ_log_min, λ_log_max), λ_log))
    # end
    if isnothing(ω_prior)
        # println("Prior is nothing")
        lp_ω = logpdf(Uniform(ωmin, ωmax), ω)
        lp_λ = log(pdf(Normal(λ_log_prior_mean, λ_log_prior_sd), λ_log))
        return lp_ω + lp_λ
    else 
        lp_ω = log(pdf(ω_prior, ω))
        lp_λ = log(pdf(Normal(λ_log_prior_mean, λ_log_prior_sd), λ_log))
        return lp_ω + lp_λ
    end


    return lp 
end

function log_prior_theta_w2_quantum_multi(
    ω::Vector{<:Real};
    ωmin::AbstractVector{<:Real}, ωmax::AbstractVector{<:Real}, 
    priors::AbstractVector{<:Union{Nothing, Distribution}} = fill(nothing, length(ω))
)   

    if !(any(ωmin .<= ω .<= ωmax))
        return -Inf
    end
    lp_total = 0.0
    for i in 1:length(ω)
        if isnothing(priors[i])
            lp_total += log(pdf(Uniform(ωmin[i], ωmax[i]), ω[i]))
        else
            lp_total += log(pdf(truncated(priors[i], lower = ωmin[i], upper = ωmax[i]), ω[i]))
        end
    end
    return lp_total
end

# this function will use a multi-variate Gaussian as the log prior
function log_prior_theta_w2_quantum_multi(
    ω::Vector{<:Real};
    ωmin::AbstractVector{<:Real}, ωmax::AbstractVector{<:Real}, 
    prior::Union{Nothing, Distribution} = nothing
)   

    if !(any(ωmin .<= ω .<= ωmax))
        return -Inf
    end
    if isnothing(prior)
        return sum([logpdf(Uniform(ωmin[i], ωmax[i]), 1.5) for i in 1:length(ωmin)])
    else 
        return log(pdf(prior))
    end
end

#adaptive learning parameter version of above 
function log_prior_theta_w2_quantum_multi_adaptive(
    ω::Vector{<:Real},
    λ_log::Real;
    ωmin::AbstractVector{<:Real}, ωmax::AbstractVector{<:Real},
    λ_log_min::Real = log(1.0001),
    λ_log_max::Real = log(50.0),
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3,
    prior::Union{Nothing, Distribution} = nothing
)   

    if !(all(ωmin .<= ω .<= ωmax)) || !(λ_log_min <= λ_log <= λ_log_max)
        # println("Returning infinity")
        return -Inf
    end
    if isnothing(prior)
        # println("Prior is nothing")
        lp_ω = sum([logpdf(Uniform(ωmin[i], ωmax[i]), ω[i]) for i in eachindex(ω)])
        lp_λ = log(pdf(Normal(λ_log_prior_mean, λ_log_prior_sd), λ_log))
        return lp_ω + lp_λ
    else 
        lp_ω = log(pdf(prior, ω))
        lp_λ = log(pdf(Normal(λ_log_prior_mean, λ_log_prior_sd), λ_log))
        return lp_ω + lp_λ
    end
end

function log_prior_theta_w2_quantum(
    ω::Real, 
    prior_density::NamedTuple
)   
    xs = prior_density.x_grid 
    ωmin = xs[1]
    ωmax = xs[end]
    # println("ω: $ω, ωmin: $ωmin, ωmax: $ωmax")
    if !(ωmin <= ω <= ωmax)
        return -Inf 
    end
    return log(prior_density.f_pdf(ω))
end

# function log_prior_theta_w2_quantum(
#     ω::Real; 
#     ωmin::Real, 
#     ωmax::Real, 
#     prior::Distribution
# )
#     if !(ωmin <= ω <= ωmax)
#         return -Inf 
#     end
#     return log(pdf(truncated(prior, lower = ωmin, upper = ωmax), ω))
# end

# ------------------------------------------------------------
# 1D W2^2 on a single trace
# ------------------------------------------------------------
function Wasserstein_trace_squared(f::AbstractVector, g::AbstractVector, t::AbstractVector)
    Nt = length(t)
    # println(Nt)
    # println(length(f))
    # println(length(g))
    @assert length(f) == Nt == length(g)

    Fint = cumsum(f)
    Gint = cumsum(g)

    tG = similar(t, Float64, Nt)
    tG[1] = t[1]

    @inbounds for k in 2:Nt-1
        val = Fint[k]
        kL = findlast(x -> x <= val, Gint)
        kR = findfirst(x -> x > val, Gint)

        if kL === nothing
            tG[k] = t[1]
        elseif kR === nothing
            tG[k] = t[end]
        else
            tL, tR = t[kL], t[kR]
            GL, GR = Gint[kL], Gint[kR]
            denom = GR - GL
            tG[k] = denom <= 0 ? tL :
                clamp(tL + (tR - tL) * (val - GL) / denom, t[1], t[end])
        end
    end
    tG[Nt] = t[end]

    return sum(((t .- tG).^2) .* f)
end

# ------------------------------------------------------------
# Trace-by-trace W2^2 empirical risk
# ------------------------------------------------------------
function trace_wasserstein_squared_loss(θ::Real, event_obs; δ::Real, kwargs...)
    event_sim = forward_event(
        θ;
        xmin=first(event_obs.xgrid),
        xmax=last(event_obs.xgrid),
        T=last(event_obs.tgrid),
        Nx=length(event_obs.xgrid),
        Nt=length(event_obs.tgrid),
        kwargs...
    )

    post_obs = postprocess_event_wasserstein(event_obs; δ=δ)
    post_sim = postprocess_event_wasserstein(event_sim; δ=δ)

    Nx = length(event_obs.xgrid)
    tgrid = event_obs.tgrid
    dvals = zeros(Nx)

    for i in 1:Nx
        f = vec(post_sim.g_prob_traces[i, :])
        g = vec(post_obs.g_prob_traces[i, :])
        dvals[i] = Wasserstein_trace_squared(f, g, tgrid)
    end

    return mean(dvals)
end

# ------------------------------------------------------------
# Log posterior with fixed λ
# ------------------------------------------------------------
function log_w2_posterior(
    θ::Vector{Real},
    event_obs;
    λ::Real,
    scale_factor::Real,
    risk_scale::Real=1.0,
    θmin::Vector{Real},
    θmax::Vector{Real},
    δ::Real,
    kwargs...
)
    lp = log_prior_theta_w2(θ; θmin=θmin, θmax=θmax)
    if !isfinite(lp)
        return -Inf, Inf
    end

    Φ = trace_wasserstein_squared_loss(θ, event_obs; δ=δ, kwargs...)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ
end

function run_w2_chain(
    event_obs;
    θ0::Real=0.0,
    λ::Real=10.0,
    iterations::Int=5000,
    burnin::Int=2500,
    thin::Int=2,
    θmin::Real=0.0,
    θmax::Real=1.0,
    δ::Real=2.0,
    scale_factor::Real=length(event_obs.g),
    risk_scale::Real=1.0,
    t0_adapt::Int=100,
    target_accept::Real=0.44,
    rng=Random.default_rng(),
    kwargs...
)
    @assert λ > 0

    chain = zeros(iterations + 1, 4)

    logpost0, Φ0 = log_w2_posterior(
        θ0, event_obs;
        λ=λ,
        scale_factor=scale_factor,
        risk_scale=risk_scale,
        θmin=θmin,
        θmax=θmax,
        δ=δ,
        kwargs...
    )
    chain[1, :] .= [θ0, logpost0, Φ0]

    θ_curr = θ0
    logpost_curr, Φ_curr = logpost0, Φ0

    accept_theta = 0

    μθ = θ0
    Σθ = 0.01 * Matrix(1.0*I, length(θ), length(θ)) # multiply by identity 
    ηθ = 0.0
    γ = k -> (k + 1)^(-2/3)

    for iter in 1:iterations
        θ_prop = θ_curr + exp(ηθ) * sqrt(Σθ) * randn(rng) # change to multivariate normal with 0 mean and identity covariance

        logpost_prop, Φ_prop = log_w2_posterior(
            θ_prop, event_obs;
            λ=λ,
            scale_factor=scale_factor,
            risk_scale=risk_scale,
            θmin=θmin,
            θmax=θmax,
            δ=δ,
            kwargs...
        )

        αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
        
        if rand(rng) < αθ
            θ_curr, logpost_curr, Φ_curr = θ_prop, logpost_prop, Φ_prop
            accept_theta += 1
        end

        chain[iter + 1, 1] = θ_curr
        chain[iter + 1, 2] = logpost_curr
        chain[iter + 1, 3] = Φ_curr

        if iter >= t0_adapt
            dθ = θ_curr - μθ # vector
            μθ += γ(iter) * dθ # vector
            Σθ += γ(iter) * (dθ*transpose(dθ) - Σθ) # matrix
            Σθ = max(Σθ, 1e-10) # matrix  # floor diagonal entries with 1e-10
            ηθ += γ(iter) * (αθ - target_accept) # scalar
        end
    end

    kept = collect(burnin:thin:(iterations + 1))
    chain_post = chain[kept, :]

    return (
        chain = chain,
        chain_post = chain_post,
        λ = λ,
        accept_theta = accept_theta / iterations
    )
end

function trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs; δ::Real, kwargs...)

    # simulate discrete event here 
    event_sim = forward_event_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal)

    data_dims = size(event_sim)
    N_init = (data_dims[1])
    N_states = (data_dims[3])
    # event_obs = event_obs[1:N_states - 1,:,1:data_dims]
    # event_sim = event_sim[1:N_states - 1,:,1:data_dims]

    post_obs = postprocess_event_wasserstein_quantum(event_obs; δ = δ)
    post_sim = postprocess_event_wasserstein_quantum(event_sim; δ = δ)
    t_grid = LinRange(0, T, nsteps + 1)
    Nx = N_init*N_states
    dvals = zeros(Nx)
    count = 1
    for i in 1:N_init 
        for j in 1:N_states
            f = vec(post_sim.g_prob_traces[i,:,j])
            g = vec(post_obs.g_prob_traces[i,:,j])
            dvals[count] = Wasserstein_trace_squared(f, g, t_grid)
            count += 1
        end
    end
    # for i in 1:Nx 
    #     f = vec(post_sim.g_prob_traces)
    #     g = vec(post_obs.g_prob_traces)
    #     dvals[i] = Wasserstein_trace_squared(f, g, t_grid)
    # end 
    return mean(dvals)
end

function trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal_total, event_obs_total, total_data_count; δ::Real, verbose::Bool, kwargs...)

    data_dims = size(event_obs_total[1])
    N_states = data_dims[1]-1
    N_init = data_dims[3]
    Nx = N_init*N_states*total_data_count
    dvals = zeros(Nx)
    # println("Nx: $Nx")
    # println("ω: $ω")
    count = 1
    for i in 1:total_data_count
        # println(degree)
        # println(n_splines)
        # println(T)
        # println(nsteps)
        # println(pcof_optimal_total)
        # println("ω: ", ω)
        # println(degree[i])
        # println(n_splines[i])
        # println(T[i])
        # println(nsteps[i])
        # println(pcof_optimal_total[i])
        event_sim = forward_event_quantum(ω, ωr, degree[i], n_splines[i], U0, T[i], nsteps[i], pcof_optimal_total[i])
        event_obs = event_obs_total[i]
        # println("event_sim: ")
        # display(event_sim)
        # println("event_obs: ")
        # display(event_obs)
        # println("diff: ")
        # display(event_sim .- event_obs)
        post_obs = postprocess_event_wasserstein_quantum(event_obs; δ = δ)
        post_sim = postprocess_event_wasserstein_quantum(event_sim; δ = δ)
        t_grid = LinRange(0, T[i], nsteps[i] + 1)
        for j in 1:N_init
            for k in 1:N_states
                f = vec(post_sim.g_prob_traces[j,:,k])
                g = vec(post_obs.g_prob_traces[j,:,k])
                dvals[count] = Wasserstein_trace_squared(f, g, t_grid)
                # println("W2 distance")
                # println(Wasserstein_trace_squared(f, g, t_grid))
                count += 1
            end
        end
    end
    if verbose 
        println("ω: $ω")
        # println("dvals: $dvals")
        println("mean dvals: $(mean(dvals))")
    end
    
    return mean(dvals), sum(dvals)
end

function trace_wasserstein_squared_loss_quantum_multi(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal_total, event_obs_total, total_data_count, carrier_freqs; δ::Real, verbose::Bool = false, kwargs...)

    ω_vec = ω[1:2]
    cross_kerr = ω[3]
    # dipole = ω[3]
    data_dims = size(event_obs_total[1])
    N_states = data_dims[1]-1
    N_init = data_dims[3]
    Nx = N_init*N_states*total_data_count
    dvals = zeros(Nx)
    # println("Nx: $Nx")
    # println("ω: $ω")
    count = 1
    for i in 1:total_data_count
        # println(degree)
        # println(n_splines)
        # println(T)
        # println(nsteps)
        # println(carrier_freqs)
        # println(pcof_optimal_total)
        # println(pcof_optimal_total[i][1])
        # println(pcof_optimal_total[i][2])
        # println(carrier_freqs)
        event_sim,_, _ = forward_event_quantum_multi([2,2], [0,0], ω_vec, ωr, [0.0,0.0], 0.0, cross_kerr, degree[i], n_splines[i], U0, T[i], nsteps[i], [pcof_optimal_total[i][1], pcof_optimal_total[i][2]], carrier_freqs)
        event_obs = event_obs_total[i]
        # println("ωr: $ωr")
        
        # display(event_obs)
        # println("event_sim: ")
        # display(event_sim)
        # println("event_obs: ")
        # display(event_obs)
        # println("diff: ")
        # display(event_sim .- event_obs)
        post_obs = postprocess_event_wasserstein_quantum(event_obs; δ = δ)
        post_sim = postprocess_event_wasserstein_quantum(event_sim; δ = δ)
        t_grid = LinRange(0, T[i], nsteps[i] + 1)
        for j in 1:N_init
            for k in 1:N_states
                f = vec(post_sim.g_prob_traces[j,:,k])
                g = vec(post_obs.g_prob_traces[j,:,k])
                dvals[count] = Wasserstein_trace_squared(f, g, t_grid)
                # println("W2 distance")
                # println(Wasserstein_trace_squared(f, g, t_grid))
                count += 1
            end
        end
    end
    # println("ω: $ω")
    # println("dvals: $dvals")
    if verbose
        println("ω: $ω_vec, ξ: $cross_kerr")
        println("mean dvals: $(mean(dvals))")
    end
    return mean(dvals), sum(dvals)

end



function log_w2_posterior_quantum(
    ω, 
    ωr, 
    degree, 
    n_splines, 
    U0, 
    T, 
    nsteps, 
    pcof_optimal,
    event_obs, 
    total_data_count;
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    λ::Real,
    scale_factor::Real,
    risk_scale::Real=1.0,
    ωmin::Real,
    ωmax::Real,
    δ::Real,
    kwargs...
)   
    if typeof(prior) == NamedTuple
        lp = log_prior_theta_w2_quantum(ω, prior)
    else
        lp = log_prior_theta_w2_quantum(ω; ωmin = ωmin, ωmax = ωmax, prior = prior)
    end
    if !isfinite(lp)
        return -Inf, Inf, -Inf, Inf
    end

    # need to change pcof_optimal and event_obs to some sort of dictionary
    # println("lp: ", lp)
    Φ, _ = trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count; δ=δ, kwargs...)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ, -risk_scale * λ * scale_factor * Φ, lp
end

function log_w2_posterior_quantum_adaptive(
    ω, 
    λ_log,
    ωr, 
    degree, 
    n_splines, 
    U0, 
    T, 
    nsteps, 
    pcof_optimal,
    event_obs, 
    total_data_count;
    ω_prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3,
    scale_factor::Real,
    risk_scale::Real=1.0,
    ωmin::Real,
    ωmax::Real,
    λ_log_min::Real = log(1.0001), 
    λ_log_max::Real = log(50.0),
    verbose::Bool = false, 
    δ::Real,
    kwargs...
)   

    lp = log_prior_theta_w2_quantum_adaptive(ω, λ_log; ωmin = ωmin, ωmax = ωmax, λ_log_min = λ_log_min, λ_log_max = λ_log_max, ω_prior = ω_prior, λ_log_prior_mean= λ_log_prior_mean,
    λ_log_prior_sd = λ_log_prior_sd)
    if !isfinite(lp)
        return -Inf, Inf, -Inf, Inf
    end

    # need to change pcof_optimal and event_obs to some sort of dictionary
    # println("lp: ", lp)
    Φ, _ = trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count; δ=δ, verbose = verbose, kwargs...)
    λ = exp(λ_log)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ, -risk_scale * λ * scale_factor * Φ, lp
end

function log_w2_posterior_quantum_multi(
    ω::Vector{Float64}, 
    ωr::Vector{Float64}, 
    degree::Vector{<:Real},
    n_splines::Vector{<:Real}, 
    U0::AbstractMatrix, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal::AbstractVector,
    carrier_freqs::AbstractVector,
    event_obs::AbstractArray, 
    total_data_count::Real; 
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    λ::Real, 
    scale_factor::Real, 
    risk_scale::Real=1.0, 
    ωmin::AbstractVector{<:Real}, 
    ωmax::AbstractVector{<:Real}, 
    δ::Real, 
    kwargs...)
    lp = log_prior_theta_w2_quantum_multi(ω; ωmin = ωmin, ωmax = ωmax, prior = prior)
    if !isfinite(lp)
        return -Inf, Inf, -Inf, Inf
    end
    Φ, _ = trace_wasserstein_squared_loss_quantum_multi(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count, carrier_freqs; δ=δ, kwargs...)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ, -risk_scale * λ * scale_factor * Φ, lp
end

function log_w2_posterior_quantum_multi_adaptive(
    ω::Vector{Float64}, 
    ωr::Vector{Float64}, 
    degree::Vector{<:Real},
    n_splines::Vector{<:Real}, 
    U0::AbstractMatrix, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal::AbstractVector,
    carrier_freqs::Union{AbstractVector, Nothing},
    event_obs::AbstractArray, 
    total_data_count::Real; 
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    λ_log::Real, 
    scale_factor::Real, 
    risk_scale::Real=1.0, 
    ωmin::AbstractVector{<:Real}, 
    ωmax::AbstractVector{<:Real},
    λ_log_min::Real = log(1.0001), 
    λ_log_max::Real = log(50.0), 
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3,
    δ::Real, 
    verbose::Bool = false,
    kwargs...)
    lp = log_prior_theta_w2_quantum_multi_adaptive(ω, λ_log; ωmin = ωmin, ωmax = ωmax, λ_log_min = λ_log_min, λ_log_max = λ_log_max, λ_log_prior_mean = λ_log_prior_mean, λ_log_prior_sd = λ_log_prior_sd, prior = prior)
    # println("lp: $lp")
    if !isfinite(lp)
        return -Inf, Inf, -Inf, Inf
    end
    Φ, _ = trace_wasserstein_squared_loss_quantum_multi(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count, carrier_freqs; δ=δ, verbose = verbose, kwargs...)
    λ = exp(λ_log)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ, -risk_scale * λ * scale_factor * Φ, lp
end

function log_w2_posterior_quantum_sum(
    ω, 
    ωr, 
    degree, 
    n_splines, 
    U0, 
    T, 
    nsteps, 
    pcof_optimal,
    event_obs, 
    total_data_count;
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    λ::Real,
    scale_factor::Real,
    risk_scale::Real=1.0,
    ωmin::Real,
    ωmax::Real,
    δ::Real,
    kwargs...
)   
    if typeof(prior) == NamedTuple
        lp = log_prior_theta_w2_quantum(ω, prior)
    else
        lp = log_prior_theta_w2_quantum(ω; ωmin = ωmin, ωmax = ωmax, prior = prior)
    end
    if !isfinite(lp)
        return -Inf, Inf, -Inf, Inf
    end

    # need to change pcof_optimal and event_obs to some sort of dictionary
    # println("lp: ", lp)
    _, Φ_sum = trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count; δ=δ, kwargs...)
    return -risk_scale * λ * scale_factor * Φ_sum + lp, Φ_sum, -risk_scale * λ * scale_factor * Φ_sum, lp
end


# ------------------------------------------------------------
# Adaptive scaling Metropolis for θ only
# chain columns:
#   1: θ
#   2: log posterior
#   3: empirical risk Φ
# ------------------------------------------------------------


function run_w2_chain_quantum(
    event_obs_total;
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    ω0_vec::Vector{<:Real} = zeros(length(ω0_vec)), 
    ωr::Real = 0.0, 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    total_data_count,
    λ::Real=10.0,
    iterations::Int=5000,
    burnin::Int=2500,
    thin::Int=2,
    ωmin::Real=0.0,
    ωmax::Real=1.0,
    δ::Real=2.0,
    scale_factor::Real=length(event_obs_total[1,:,1]),
    risk_scale::Real=1.0,
    t0_adapt::Int=100,
    target_accept::Real=0.44,
    rng=Random.default_rng(),
    kwargs...
)
    @assert λ > 0
    # run the following code for each chain, starting from initial conditions
    # create empty vectors to concatenate each chain
    num_chains = length(ω0_vec)
    accept_theta_vec = zeros(num_chains)
    chain_samples = zeros((iterations + 1)*num_chains, 5)
    chain_kept_samples = length(collect(burnin:thin:iterations + 1))
    total_chain_kept_samples = zeros(num_chains*chain_kept_samples, 5)
    diff_samples = zeros((iterations + 1)*num_chains, 4)
    diagnostic_chain = zeros(chain_kept_samples, num_chains)
    hyperparam_history = zeros(num_chains, iterations + 1, 3)
    proposed_ω = zeros(num_chains, iterations)
    count = 1
    for ω0 in ω0_vec
        chain = zeros(iterations + 1, 5)
        diff_chain = zeros(iterations, 4)
        logpost0, Φ0, Φ0_scaled, logprior0 = log_w2_posterior_quantum(
            ω0, 
            ωr, 
            degree, 
            n_splines, 
            U0, 
            T, 
            nsteps, 
            pcof_optimal_total,
            event_obs_total,
            total_data_count;
            prior = prior, 
            λ=λ,
            scale_factor=scale_factor,
            risk_scale=risk_scale,
            ωmin=ωmin,
            ωmax=ωmax,
            δ=δ,
            kwargs...
        )
        chain[1, :] .= [ω0, logpost0, Φ0, Φ0_scaled, logprior0]
        ω_curr = ω0
        logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = logpost0, Φ0, Φ0_scaled, logprior0
        
        accept_theta = 0

        μθ = ω0
        Σθ = 0.1
        ηθ = 0.0
        γ = k -> (k + 1)^(-2/3)
        hyperparam_history[count,1,:] .= [μθ, Σθ, ηθ]
        for iter in 1:iterations
            rand_number = randn(rng)
            ω_prop = ω_curr + exp(ηθ) * sqrt(Σθ) * rand_number
            proposed_ω[count, iter] = ω_prop
            logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop = log_w2_posterior_quantum(
                ω_prop, 
                ωr, 
                degree, 
                n_splines, 
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                event_obs_total,
                total_data_count;
                prior = prior, 
                λ=λ,
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                δ=δ,
                kwargs...
            )
            # println("ω_prop: $(ω_prop)")
            diff_prop = logpost_prop - logpost_curr
            diff_lp = logprior_prop - logprior_curr
            diff_chain[iter, 1] = ω_prop
            diff_chain[iter, 2] = diff_prop
            diff_chain[iter, 3] = exp(diff_prop)
            diff_chain[iter, 4] = diff_lp
            αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
            # println("αθ: $(αθ)")  
            if rand(rng) < αθ
                ω_curr, logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = ω_prop, logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop
                accept_theta += 1
            end
            accept_theta_vec[count] = accept_theta
            chain[iter + 1, 1] = ω_curr
            chain[iter + 1, 2] = logpost_curr
            chain[iter + 1, 3] = Φ_curr
            chain[iter + 1, 4] = Φ_curr_scaled
            chain[iter + 1, 5] = logprior_curr

            # include λ to be tunable parameter
            # update λ in log-space 
            # λ_n = log10(λ) + ηθ*randn(rng)
            # αλ = isfinite(logpost_prop)

            if iter >= t0_adapt
                dθ = ω_curr - μθ
                μθ += γ(iter) * dθ
                Σθ += γ(iter) * (dθ^2 - Σθ)
                Σθ = max(Σθ, 1e-10)
                ηθ += γ(iter) * (αθ - target_accept)
            end
            hyperparam_history[count,iter + 1,:] .= [μθ, Σθ, ηθ]
        end
        kept = collect(burnin:thin:(iterations + 1))
        chain_post = chain[kept, :]
        chain_samples[(count - 1)*iterations + 1: count*iterations + 1,:] .= chain
        total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post
        diagnostic_chain[:,count] .= chain_post[:,1]
        count += 1
        
    end
    return (
        chain = chain_samples,
        diagnostic_chain = diagnostic_chain,
        chain_post = total_chain_kept_samples,
        hyperparam_history = hyperparam_history,
        proposed_ω = proposed_ω, 
        diff_chain = diff_samples,
        λ = λ,
        accept_theta = accept_theta_vec ./ iterations
    )
end

function run_w2_chain_quantum_adaptive(
    event_obs_total;
    ω_prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    ω0_vec::Vector{<:Real} = zeros(length(ω0_vec)), 
    ωr::Real = 0.0, 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    total_data_count,
    λ_log0::Real=log(10.0),
    iterations::Int=5000,
    burnin::Int=2500,
    thin::Int=2,
    ωmin::Real=0.0,
    ωmax::Real=1.0,
    λ_log_min::Real = log(1.0001),
    λ_log_max::Real = log(50.0),
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3,
    δ::Real=2.0,
    scale_factor::Real=length(event_obs_total[1,:,1]),
    risk_scale::Real=1.0,
    t0_adapt::Int=100,
    target_accept::Real=0.44,
    rng=Random.default_rng(),
    verbose::Bool = false,
    ESS_stopping::Union{Nothing, <:Real} = nothing, 
    variance_stopping::Union{Nothing, <:Real} = nothing,
    check_interval::Union{Nothing, <:Real} = nothing,
    kwargs...
)
    @assert exp(λ_log0) > 0
    # run the following code for each chain, starting from initial conditions
    # create empty vectors to concatenate each chain
    num_chains = length(ω0_vec)
    accept_theta_vec = zeros(num_chains)
    accept_lambda_vec = zeros(num_chains)
    chain_samples = zeros((iterations + 1)*num_chains, 6)
    chain_kept_samples = length(collect(burnin:thin:iterations + 1))
    total_chain_kept_samples = zeros(num_chains*chain_kept_samples, 6)
    diff_samples = zeros((iterations + 1)*num_chains, 4)
    diagnostic_chain = zeros(chain_kept_samples, num_chains)
    diagnostic_chain_λ = zeros(chain_kept_samples, num_chains)
    hyperparam_history_ω = zeros(num_chains, iterations + 1, 4)
    hyperparam_history_λ_log = zeros(num_chains, iterations + 1, 4)
    proposed_ω = zeros(num_chains, iterations)
    proposed_λ_log = zeros(num_chains, iterations)
    count = 1
    ω_best = zeros(num_chains)
    Φ_best = fill(1E3, num_chains)
    for ω0 in ω0_vec
        chain = zeros(iterations + 1, 6)
        diff_chain = zeros(iterations, 4)
        logpost0, Φ0, Φ0_scaled, logprior0 = log_w2_posterior_quantum_adaptive(
            ω0, 
            λ_log0,
            ωr, 
            degree, 
            n_splines, 
            U0, 
            T, 
            nsteps, 
            pcof_optimal_total,
            event_obs_total,
            total_data_count;
            ω_prior = ω_prior,
            λ_log_prior_mean = λ_log_prior_mean,
            λ_log_prior_sd = λ_log_prior_sd,
            scale_factor=scale_factor,
            risk_scale=risk_scale,
            ωmin=ωmin,
            ωmax=ωmax,
            δ=δ,
            verbose = verbose, 
            kwargs...
        )
        chain[1, :] .= [ω0, exp(λ_log0), logpost0, Φ0, Φ0_scaled, logprior0]
        ω_curr, λ_log_curr = ω0, λ_log0
        logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = logpost0, Φ0, Φ0_scaled, logprior0
        
        accept_theta = 0
        accept_λ_log = 0

        μθ, μ_λ_log = ω0, λ_log0
        Σθ, Σ_λ_log  = 0.05, 0.1
        ηθ, η_λ_log = 0.0, 0.0
        γ = k -> (k + 1)^(-2/3)
        adaptive_iter_count = iterations - t0_adapt + 1
        hyperparam_history_ω[count,1,:] .= [0.0, μθ, Σθ, ηθ]
        hyperparam_history_λ_log[count,1,:] .= [0.0, μ_λ_log, Σ_λ_log, η_λ_log]
        stopping_iter = iterations
        for iter in 1:iterations
            
            if iter % 1000 == 0
                println("Iteration $iter")
            end

            ω_prop = ω_curr + exp(ηθ) * sqrt(Σθ) * randn(rng)
            proposed_ω[count, iter] = ω_prop
            logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop = log_w2_posterior_quantum_adaptive(
                ω_prop, 
                λ_log_curr,
                ωr, 
                degree, 
                n_splines, 
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                event_obs_total,
                total_data_count;
                ω_prior = ω_prior,
                λ_log_prior_mean = λ_log_prior_mean,
                λ_log_prior_sd = λ_log_prior_sd, 
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                λ_log_min=λ_log_min,
                λ_log_max=λ_log_max,
                δ=δ,
                verbose = verbose,
                kwargs...
            )
            # println("ω_prop: $(ω_prop)")
            diff_prop = logpost_prop - logpost_curr
            diff_lp = logprior_prop - logprior_curr
            diff_chain[iter, 1] = ω_prop
            diff_chain[iter, 2] = diff_prop
            diff_chain[iter, 3] = exp(diff_prop)
            diff_chain[iter, 4] = diff_lp
            αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
            # println("αθ: $(αθ)")  
            
            if rand(rng) < αθ
                ω_curr, logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = ω_prop, logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop
                accept_theta += 1
            end

            if Φ_curr < Φ_best[count] 
                ω_best[count] = ω_curr 
                Φ_best[count] = Φ_curr 
            end

            accept_theta_vec[count] = accept_theta
            chain[iter + 1, 1] = ω_curr
            chain[iter + 1, 3] = logpost_curr
            chain[iter + 1, 4] = Φ_curr
            chain[iter + 1, 5] = Φ_curr_scaled
            chain[iter + 1, 6] = logprior_curr
            
            dθ = ω_curr - μθ
            if iter >= t0_adapt
                dθ = ω_curr - μθ
                μθ += γ(adaptive_iter_count) * dθ
                Σθ += γ(adaptive_iter_count) * (dθ^2 - Σθ)
                Σθ = max(Σθ, 1e-10)
                ηθ += γ(adaptive_iter_count) * (αθ - target_accept)
            end
            hyperparam_history_ω[count,iter + 1,:] .= [dθ, μθ, Σθ, ηθ]
            # println("λ current: ", exp(λ_log_curr))
            λ_log_prop = λ_log_curr + exp(η_λ_log) * sqrt(Σ_λ_log) * randn(rng)
            # println("λ prop: ", exp(λ_log_prop))
            proposed_λ_log[count, iter] = λ_log_prop
            logpost_prop, _, _, _ = log_w2_posterior_quantum_adaptive(
                ω_curr, 
                λ_log_prop,
                ωr, 
                degree, 
                n_splines, 
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                event_obs_total,
                total_data_count;
                ω_prior = ω_prior,
                λ_log_prior_mean = λ_log_prior_mean,
                λ_log_prior_sd = λ_log_prior_sd, 
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                λ_log_min=λ_log_min,
                λ_log_max=λ_log_max,
                δ=δ,
                kwargs...
            )
            # println("logpost curr: ", logpost_curr)
            # println("logpost prop: ", logpost_prop)
            αλ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
            if rand(rng) < αλ
                λ_log_curr, logpost_curr = λ_log_prop, logpost_prop
                accept_λ_log += 1
            end
            accept_lambda_vec[count] = accept_λ_log
            chain[iter + 1, 2] = exp(λ_log_curr)
            chain[iter + 1, 3] = logpost_curr
            d_λ_log = λ_log_curr - μ_λ_log
            if iter >= t0_adapt
                d_λ_log = λ_log_curr - μ_λ_log
                μ_λ_log += γ(adaptive_iter_count) * d_λ_log
                Σ_λ_log += γ(adaptive_iter_count) * (d_λ_log^2 - Σ_λ_log)
                Σ_λ_log = max(Σ_λ_log, 1e-10)
                η_λ_log += γ(adaptive_iter_count) * (αλ - target_accept)
            end

            if !isnothing(check_interval) && iter >= burnin && iter % stopping_check == 0 
                ess_curr = ess(@view chain[burnin+1:iter, 1])
                diff_var = abs(hyperparam_history_ω[count,iter,3] - hyperparam_history_ω[count, iter - 1, 3])
                ess_done = isnothing(ESS_stopping) || (ess_curr >= ESS_stopping)
                var_done = isnothing(variance_stopping) || (diff_var <= variance_stopping)
                
                if ess_done && var_done 
                    stopping_iter = iter
                    break 
                end
            end

            hyperparam_history_λ_log[count,iter + 1,:] .= [d_λ_log, μ_λ_log, Σ_λ_log, η_λ_log]
            adaptive_iter_count += 1
        end

        kept = collect(burnin:thin:(stopping_iter + 1))
        chain_post = chain[kept, :]
        chain_samples[(count - 1)*stopping_iter + 1: count*stopping_iter + 1,:] .= chain
        total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post
        diagnostic_chain[:,count] .= chain_post[:,1]
        diagnostic_chain_λ[:,count] .= chain_post[:,2]
        count += 1

        
        # kept = collect(burnin:thin:(iterations + 1))
        # chain_post = chain[kept, :]
        # chain_samples[(count - 1)*iterations + 1: count*iterations + 1,:] .= chain
        # total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post
        # diagnostic_chain[:,count] .= chain_post[:,1]
        # diagnostic_chain_λ[:,count] .= chain_post[:,2]
        # count += 1
        
        
    end
    return (
        chain = chain_samples,
        diagnostic_chain = diagnostic_chain,
        diagnostic_chain_λ = diagnostic_chain_λ,
        chain_post = total_chain_kept_samples,
        hyperparam_history_ω = hyperparam_history_ω,
        hyperparam_history_λ_log = hyperparam_history_λ_log,
        proposed_ω = proposed_ω, 
        proposed_λ_log = proposed_λ_log,
        diff_chain = diff_samples,
        accept_theta = accept_theta_vec ./ iterations,
        accept_lambda = accept_lambda_vec ./ iterations,
        ω_best, Φ_best
    )
end

function run_w2_chain_quantum_multi(
    event_obs_total;
    
    θ::AbstractMatrix{<:Real} = zeros(size(θ)), 
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    ωr::AbstractVector{<:Real} = zeros(size(θ, 2)-1), 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    carrier_freqs::AbstractVector,
    total_data_count,
    λ::Real=10.0,
    iterations::Int=5000,
    burnin::Int=2500,
    thin::Int=2,
    ωmin::AbstractVector{<:Real}=zeros(size(θ, 2)),
    ωmax::AbstractVector{<:Real}=ones(size(θ, 2)),
    δ::Real=2.0,
    scale_factor::Real=length(event_obs_total[1,:,1]),
    risk_scale::Real=1.0,
    t0_adapt::Int=100,
    target_accept::Real=0.23,
    rng=Random.default_rng(),
    kwargs...
)
    num_chains, params = size(θ)
    @assert λ > 0
    # println("event obs")
    # display(event_obs_total[1])
    # display(event_obs_total[2])
    # display(event_obs_total[3])
    # display(event_obs_total[4])
    chain_samples = Matrix{Any}(undef, (iterations + 1)*num_chains, 5)
    accept_theta_vec = zeros(num_chains)
    chain_kept_samples = length(collect(burnin:thin:iterations + 1))
    total_chain_kept_samples = Matrix{Any}(undef, num_chains*chain_kept_samples, 5)
    diagnostic_chain = zeros(chain_kept_samples, num_chains, params)
    hyperparam_history = Array{Any}(undef, num_chains, iterations + 1, 3)
    count = 1
    for ω0 in eachrow(θ)
        ω0 = collect(ω0)
        chain = Matrix{Any}(undef, iterations + 1, 5)
        logpost0, Φ0, Φ0_scaled, logprior0 = log_w2_posterior_quantum_multi(
            ω0, 
            ωr, 
            degree, 
            n_splines, 
            U0, 
            T, 
            nsteps, 
            pcof_optimal_total,
            carrier_freqs,
            event_obs_total,
            total_data_count;
            priors = priors, 
            λ=λ,
            scale_factor=scale_factor,
            risk_scale=risk_scale,
            ωmin=ωmin,
            ωmax=ωmax,
            δ=δ,
            kwargs...
        )
        chain[1, :] .= [ω0, logpost0, Φ0, Φ0_scaled, logprior0]

        ω_curr = deepcopy(ω0)
        logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = logpost0, Φ0, Φ0_scaled, logprior0

        accept_theta = 0

        μθ = deepcopy(ω0)
        Σθ = 0.01 * Matrix(1.0*I, params, params)
        ηθ = 0.0
        γ = k -> (k + 1)^(-2/3)
        hyperparam_history[count, 1,:] = [μθ, Σθ, ηθ]
        for iter in 1:iterations
            if iter % 1000 == 0
                println("Iteration $iter")
            end
            rand_number = randn(rng, params)
            ω_prop = ω_curr + exp(ηθ) * sqrt(Σθ) * rand_number
            logpost_prop, Φ_prop,Φ_prop_scaled, logprior_prop = log_w2_posterior_quantum_multi(
                ω_prop, 
                ωr, 
                degree, 
                n_splines, 
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                carrier_freqs,
                event_obs_total,
                total_data_count;
                prior = prior, 
                λ=λ,
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                δ=δ,
                kwargs...
            )
            αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0

            if rand(rng) < αθ
                ω_curr, logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = ω_prop, logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop
                accept_theta += 1
            end

            accept_theta_vec[count] = accept_theta
            chain[iter + 1, 1] = ω_curr
            chain[iter + 1, 2] = logpost_curr
            chain[iter + 1, 3] = Φ_curr
            chain[iter + 1, 4] = Φ_curr_scaled
            chain[iter + 1, 5] = logprior_curr

            if iter >= t0_adapt
                dθ = ω_curr - μθ
                μθ += γ(iter) * dθ
                Σθ += γ(iter) * (dθ*dθ' - Σθ)
                Σθ = clamp_diag(Σθ, cutoff = 1E-10)
                ηθ += γ(iter) * (αθ - target_accept)
            end
            hyperparam_history[count, iter + 1, :] = [μθ, Σθ, ηθ]
        end

        kept = collect(burnin:thin:(iterations + 1))
        chain_post = chain[kept, :]
        chain_samples[(count - 1)*iterations + 1: count*iterations + 1,:] .= chain
        total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post
        diagnostic_chain[:,count,:] .= stack(chain_post[:,1],dims = 1)
        count += 1
    end
    
    return (
        chain = chain_samples,
        diagnostic_chain = diagnostic_chain,
        chain_post = total_chain_kept_samples,
        hyperparam_history = hyperparam_history,
        λ = λ,
        accept_theta = accept_theta_vec ./ iterations
    )
end


function run_w2_chain_quantum_multi_adaptive(
    event_obs_total;
    
    θ::AbstractMatrix{<:Real} = zeros(size(θ)), 
    prior::Union{Nothing, NamedTuple, Distribution} = nothing,
    ωr::AbstractVector{<:Real} = zeros(size(θ, 2)-1), 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    carrier_freqs::Union{AbstractVector, Nothing} = nothing,
    total_data_count,
    λ_log0::Real=log(10.0),
    iterations::Int=5000,
    burnin::Int=2500,
    thin::Int=2,
    ωmin::AbstractVector{<:Real}=zeros(size(θ, 2)),
    ωmax::AbstractVector{<:Real}=ones(size(θ, 2)),
    λ_log_min::Real = log(1.001),
    λ_log_max::Real = log(50.0),
    λ_log_prior_mean::Real=log(5.0),
    λ_log_prior_sd::Real = 0.3,
    δ::Real=2.0,
    scale_factor::Real=length(event_obs_total[1,:,1]),
    ridge_param::Real=1e-10,
    risk_scale::Real=1.0,
    t0_adapt::Int=100,
    target_accept::Real=0.23,
    target_accept_λ::Real = 0.44,
    initial_cov_p = nothing,
    rng=Random.default_rng(),
    verbose::Bool = false,
    ESS_stopping::Union{Nothing, <:Real} = nothing, 
    variance_stopping::Union{Nothing, <:Real} = nothing,
    check_interval::Union{Nothing, <:Real} = nothing,
    kwargs...
)
  
    num_chains, params = size(θ)
    @assert exp(λ_log0) > 0
    # println("event obs")
    # display(event_obs_total[1])
    # display(event_obs_total[2])
    # display(event_obs_total[3])
    # display(event_obs_total[4])
    chain_samples = Matrix{Any}(undef, (iterations + 1)*num_chains, 6)
    accept_theta_vec = zeros(num_chains)
    accept_lambda_vec = zeros(num_chains)
    chain_kept_samples = length(collect(burnin:thin:iterations + 1))
    total_chain_kept_samples = Matrix{Any}(undef, num_chains*chain_kept_samples, 6)
    diagnostic_chain = zeros(chain_kept_samples, num_chains, params)
    hyperparam_history = Array{Any}(undef, num_chains, iterations + 1, 3)
    hyperparam_history_λ_log = zeros(num_chains, iterations + 1, 4)
    diagnostic_chain_λ = zeros(chain_kept_samples, num_chains)
    acceptance_flag_theta = zeros(num_chains, iterations)
    count = 1
    ω_best = zeros(num_chains, params)
    Φ_best = fill(1E3, num_chains)
    for ω0 in eachrow(θ)
        ω0 = collect(ω0)
        chain = Matrix{Any}(undef, iterations + 1, 6)
        logpost0, Φ0, Φ0_scaled, logprior0 = log_w2_posterior_quantum_multi_adaptive(
            ω0, 
            ωr, 
            degree, 
            n_splines, 
            U0, 
            T, 
            nsteps, 
            pcof_optimal_total,
            carrier_freqs,
            event_obs_total,
            total_data_count;
            prior = prior, 
            λ_log=λ_log0,
            scale_factor=scale_factor,
            risk_scale=risk_scale,
            ωmin=ωmin,
            ωmax=ωmax,
            λ_log_min=λ_log_min,
            λ_log_max=λ_log_max,
            λ_log_prior_mean,
            λ_log_prior_sd,
            δ=δ,
            verbose,
            kwargs...
        )
        chain[1, :] .= [ω0, exp(λ_log0), logpost0, Φ0, Φ0_scaled, logprior0]

        ω_curr, λ_log_curr = deepcopy(ω0), λ_log0
        logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = logpost0, Φ0, Φ0_scaled, logprior0

        accept_theta = 0
        accept_λ_log = 0

        μθ, μ_λ_log = deepcopy(ω0), λ_log0
        Σθ = initial_cov_p === nothing ? Diagonal(((ωmax .- ωmin) ./ 10).^2) |> Matrix : Matrix(initial_cov_p)
        Σ_λ_log = 0.1
        ηθ, η_λ_log = 0.0, 0.0
        γ = k -> (k + 1)^(-2/3)
        adaptive_iter_count = iterations - t0_adapt + 1
        hyperparam_history[count, 1,:] = [μθ, Σθ, ηθ]
        hyperparam_history_λ_log[count,1,:] .= [0.0, μ_λ_log, Σ_λ_log, η_λ_log]
        
        stopping_iter = iterations

        for iter in 1:iterations
            if iter % 1000 == 0
                println("Iteration $iter")
            end
            rand_number = randn(rng, params)
            Lp = cholesky(Symmetric(Σθ + ridge_param * I), check = false).L
            ω_prop = ω_curr + exp(ηθ) * Lp * rand_number
            logpost_prop, Φ_prop,Φ_prop_scaled, logprior_prop = log_w2_posterior_quantum_multi_adaptive(
                ω_prop, 
                ωr, 
                degree, 
                n_splines,
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                carrier_freqs,
                event_obs_total,
                total_data_count;
                prior = prior, 
                λ_log=λ_log_curr,
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                λ_log_min=λ_log_min,
                λ_log_max=λ_log_max,
                λ_log_prior_mean,
                λ_log_prior_sd,
                δ=δ,
                verbose,
                kwargs...
            )

            # println(logpost_prop)
            αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
            accepted_theta = false
            if rand(rng) < αθ
                accepted_theta = true
                ω_curr, logpost_curr, Φ_curr, Φ_curr_scaled, logprior_curr = ω_prop, logpost_prop, Φ_prop, Φ_prop_scaled, logprior_prop
                accept_theta += 1
                acceptance_flag_theta[count, iter] = 1
            end

            if Φ_curr < Φ_best[count] 
                ω_best[count,:] = ω_curr 
                Φ_best[count] = Φ_curr 
            end

            accept_theta_vec[count] = accept_theta
            chain[iter + 1, 1] = ω_curr
            chain[iter + 1, 3] = logpost_curr
            chain[iter + 1, 4] = Φ_curr
            chain[iter + 1, 5] = Φ_curr_scaled
            chain[iter + 1, 6] = logprior_curr



            if iter >= t0_adapt
                dθ = ω_curr - μθ
                μθ += γ(adaptive_iter_count) * dθ
                Σθ += γ(adaptive_iter_count) * (dθ*dθ' - Σθ)
                Σθ = clamp_eigs(Σθ, min_eig = 1E-10, max_eig = 1.0)
                ηθ += γ(adaptive_iter_count) * (αθ - target_accept)
            end
            hyperparam_history[count, iter + 1, :] = [μθ, Σθ, ηθ]

            λ_log_prop = λ_log_curr + exp(η_λ_log) * sqrt(Σ_λ_log) * randn(rng)

            logpost_prop, _ ,_ ,_  = log_w2_posterior_quantum_multi_adaptive(
                ω_curr, 
                ωr, 
                degree, 
                n_splines, 
                U0, 
                T, 
                nsteps, 
                pcof_optimal_total,
                carrier_freqs,
                event_obs_total,
                total_data_count;
                prior = prior, 
                λ_log=λ_log_prop,
                scale_factor=scale_factor,
                risk_scale=risk_scale,
                ωmin=ωmin,
                ωmax=ωmax,
                λ_log_min=λ_log_min,
                λ_log_max=λ_log_max,
                δ=δ,
                kwargs...
            )
            αλ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0
            accepted_lambda = false
            if rand(rng) < αλ
                    accepted_lambda = true
                    λ_log_curr, logpost_curr = λ_log_prop, logpost_prop
                    accept_λ_log += 1
            end
            accept_lambda_vec[count] = accept_λ_log
            chain[iter + 1, 2] = exp(λ_log_curr)
            chain[iter + 1, 3] = logpost_curr
            d_λ_log = λ_log_curr - μ_λ_log
            if iter >= t0_adapt
                d_λ_log = λ_log_curr - μ_λ_log
                μ_λ_log += γ(adaptive_iter_count) * d_λ_log
                Σ_λ_log += γ(adaptive_iter_count) * (d_λ_log^2 - Σ_λ_log)
                Σ_λ_log = max(Σ_λ_log, 1e-10)
                η_λ_log += γ(adaptive_iter_count) * (αλ - target_accept_λ)
            end
            hyperparam_history_λ_log[count,iter + 1,:] .= [d_λ_log, μ_λ_log, Σ_λ_log, η_λ_log]

            adaptive_iter_count += 1

            if !isnothing(check_interval) && iter > burnin + ESS_stopping && iter % check_interval == 0 
                chain_mat = (hcat(chain[burnin+1:iter, 1]...))'
                ess_curr = minimum(ess.(eachcol(chain_mat)))
                diff_var = abs(hyperparam_history[count,iter,3] - hyperparam_history[count, iter - 1, 3])
                ess_done = isnothing(ESS_stopping) || (ess_curr >= ESS_stopping)
                var_done = isnothing(variance_stopping) || (diff_var <= variance_stopping)
                
                if ess_done && var_done 
                    stopping_iter = iter
                    break 
                end
            end

        end
        


        kept = collect(burnin:thin:(stopping_iter + 1))
        chain_post = chain[kept,:]
        chain_samples[(count - 1)*stopping_iter + 1:count*stopping_iter + 1,:] .= chain 
        total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post 
        diagnostic_chain[:,count,:] .= stack(chain_post[:,1], dims = 1)
        diagnostic_chain_λ[:,count] .= chain_post[:,2]

        # kept = collect(burnin:thin:(iterations + 1))
        # chain_post = chain[kept, :]
        # chain_samples[(count - 1)*iterations + 1: count*iterations + 1,:] .= chain
        # total_chain_kept_samples[(count - 1)*chain_kept_samples + 1: count*chain_kept_samples,:] .= chain_post
        # diagnostic_chain[:,count,:] .= stack(chain_post[:,1],dims = 1)
        # diagnostic_chain_λ[:,count] .= chain_post[:,2]
        
        count += 1
    end
    
    return (
        chain = chain_samples,
        diagnostic_chain = diagnostic_chain,
        diagnostic_chain_λ = diagnostic_chain_λ,
        chain_post = total_chain_kept_samples,
        hyperparam_history = hyperparam_history,
        accept_theta = accept_theta_vec ./ iterations,
        accept_lambda = accept_lambda_vec ./ iterations, 
        acceptance_flag_theta, 
        ω_best, Φ_best
    )
end

posterior_mean_theta_w2(res) = mean(res.chain_post[:, 1])
posterior_var_theta_w2(res) = var(res.chain_post[:, 1])
