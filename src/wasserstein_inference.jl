##################################
# src/wasserstein_inference.jl
##################################

using Random
using Statistics
using Distributions

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
    ωmax::Real
)
    if !(ωmin <= ω <= ωmax)
        return -Inf
    end
    return log(pdf(Uniform(ωmin, ωmax), ω))
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

function trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal_total, event_obs_total, total_data_count; δ::Real, kwargs...)

    data_dims = size(event_obs_total[1])
    N_states = data_dims[1]-1
    N_init = data_dims[3]
    Nx = N_init*N_states*total_data_count
    dvals = zeros(Nx)

    count = 1
    for i in 1:total_data_count

        event_sim = forward_event_quantum(ω, ωr, degree[i], n_splines[i], U0, T[i], nsteps[i], pcof_optimal_total[i])
        event_obs = event_obs_total[i]

        post_obs = postprocess_event_wasserstein_quantum(event_obs; δ = δ)
        post_sim = postprocess_event_wasserstein_quantum(event_sim; δ = δ)
        t_grid = LinRange(0, T[i], nsteps[i] + 1)

        for j in 1:N_init 
            for k in 1:N_states
                f = vec(post_sim.g_prob_traces[j,:,k])
                g = vec(post_obs.g_prob_traces[j,:,k])
                dvals[count] = Wasserstein_trace_squared(f, g, t_grid)
                count += 1
            end
        end
    end
    return mean(dvals)
end

    # simulate discrete event here 
    # event_sim = forward_event_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal)

    # data_dims = size(event_sim)
    # N_init = (data_dims[1])
    # N_states = (data_dims[3])
    # # event_obs = event_obs[1:N_states - 1,:,1:data_dims]
    # # event_sim = event_sim[1:N_states - 1,:,1:data_dims]

    # post_obs = postprocess_event_wasserstein_quantum(event_obs; δ = δ)
    # post_sim = postprocess_event_wasserstein_quantum(event_sim; δ = δ)
    
    # Nx = N_init*N_states
    # dvals = zeros(Nx)
    # count = 1
    # for i in 1:N_init 
    #     for j in 1:N_states
    #         f = vec(post_sim.g_prob_traces[i,:,j])
    #         g = vec(post_obs.g_prob_traces[i,:,j])
    #         dvals[count] = Wasserstein_trace_squared(f, g, t_grid)
    #         count += 1
    #     end
    # end
    # # for i in 1:Nx 
    # #     f = vec(post_sim.g_prob_traces)
    # #     g = vec(post_obs.g_prob_traces)
    # #     dvals[i] = Wasserstein_trace_squared(f, g, t_grid)
    # # end 
    # return mean(dvals)
# end


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
    prior::Union{Nothing, NamedTuple} = nothing,
    λ::Real,
    scale_factor::Real,
    risk_scale::Real=1.0,
    ωmin::Real,
    ωmax::Real,
    δ::Real,
    kwargs...
)   
    if isnothing(prior)
        lp = log_prior_theta_w2_quantum(ω; ωmin=ωmin, ωmax=ωmax)
    else
        lp = log_prior_theta_w2_quantum(ω, prior)
    end

    if !isfinite(lp)
        return -Inf, Inf
    end

    # need to change pcof_optimal and event_obs to some sort of dictionary

    Φ = trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal, event_obs, total_data_count; δ=δ, kwargs...)
    return -risk_scale * λ * scale_factor * Φ + lp, Φ
end


# ------------------------------------------------------------
# Adaptive scaling Metropolis for θ only
# chain columns:
#   1: θ
#   2: log posterior
#   3: empirical risk Φ
# ------------------------------------------------------------
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

    chain = zeros(iterations + 1, 3)

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

function run_w2_chain_quantum(
    event_obs_total;
    prior::Union{Nothing, NamedTuple} = nothing,
    ω0::Real = 0.0, 
    ωr::Real = 0.0, 
    degree::Vector{Real}, 
    n_splines::Vector{Real}, 
    U0, 
    T::Vector{Real}, 
    nsteps::Vector{Real}, 
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
    # println("event obs")
    # display(event_obs_total[1])
    # display(event_obs_total[2])
    # display(event_obs_total[3])
    # display(event_obs_total[4])
    chain = zeros(iterations + 1, 3)
    
    logpost0, Φ0 = log_w2_posterior_quantum(
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
    chain[1, :] .= [ω0, logpost0, Φ0]

    ω_curr = ω0
    logpost_curr, Φ_curr = logpost0, Φ0

    accept_theta = 0

    μθ = ω0
    Σθ = 0.01
    ηθ = 0.0
    γ = k -> (k + 1)^(-2/3)

    for iter in 1:iterations
        rand_number = randn(rng)
        ω_prop = ω_curr + exp(ηθ) * sqrt(Σθ) * rand_number
        logpost_prop, Φ_prop = log_w2_posterior_quantum(
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
        αθ = isfinite(logpost_prop) ? min(1.0, exp(logpost_prop - logpost_curr)) : 0.0

        if rand(rng) < αθ
            ω_curr, logpost_curr, Φ_curr = ω_prop, logpost_prop, Φ_prop
            accept_theta += 1
        end

        chain[iter + 1, 1] = ω_curr
        chain[iter + 1, 2] = logpost_curr
        chain[iter + 1, 3] = Φ_curr

        if iter >= t0_adapt
            dθ = ω_curr - μθ
            μθ += γ(iter) * dθ
            Σθ += γ(iter) * (dθ^2 - Σθ)
            Σθ = max(Σθ, 1e-10)
            ηθ += γ(iter) * (αθ - target_accept)
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


posterior_mean_theta_w2(res) = mean(res.chain_post[:, 1])
posterior_var_theta_w2(res) = var(res.chain_post[:, 1])
