include("QDT.jl")
include("prior.jl")
include("events.jl")

#============================================================================
=============================================================================

    W2 DISTANCE CALCULATIONS

=============================================================================
============================================================================#

# ------------------------------------------------------------
# 1D W2^2 on a single trace
# ------------------------------------------------------------
function Wasserstein_trace_squared(f::AbstractVector, g::AbstractVector, t::AbstractVector)
    Nt = length(t)
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


"""
Numerically stable softplus with sharpness parameter δ:
softplus_δ(z) = log(1 + exp(δ z))
"""
function softplus(z, δ)
    y = δ * z
    return y > 0 ? y + log1p(exp(-y)) : log1p(exp(y))
end


"""
Mapping from a vector v = [v₁,...,vₙ] to a probabability
vector p = [p₁, ..., pₙ] with elements

                pᵢ = σ(δ vᵢ) / ∑ⱼ σ(δ vⱼ) 
    
where σ(z) = log(1 + exp(z))
"""
function simplex_map_vector(v::AbstractVector; δ::Real)
    w = softplus.(v, δ)
    s = sum(w)
    return s > 0 ? w ./ s : fill(1 / length(v), length(v))
end


"""
Given an array X of size (Nx,Nt,Ny), returns a new array
Y s.t. Y[i,:,j] stores X[i,:,j] shifted and squeezed 
into a probability distribution on Nt elements. 
"""
function postprocess_event_wasserstein(X::Array{T,3}; δ::Real) where T
    Nx, Nt, Ny = size(X)

    tmp = zeros(Nt)
    Y = zeros(Nx, Nt, Ny)

    for i in 1:Nx
        for j = 1:Ny
            tmp .= vec(X[i, :, j])
            tmp .= tmp .- mean(tmp)
            Y[i,:,j] .= simplex_map_vector(tmp; δ=δ)
        end
    end

    return Y
end



#============================================================================
=============================================================================

    W2POSTERIOR DEFINITION

=============================================================================
============================================================================#


"""
    W2Posterior struct

Stucture to contain all variables needed to evaluate 
posterior distributions based on the Wasserstein 2-distance (W2)

Fields

    digital_device::DigitalDevice

        Object on which to perform the forward simulations needed 
        to evaluate the posterior.

    
    n_obs::Int

        Number of `ObservationEvent`s defining this posterior.
        

    event_obs::Vector{ObservationEvent}

        Device data used to define the posterior distribution

    
    prior::Prior

        Prior distribution, either a UniformPrior or a KDEPrior

    λ::Real                 Various hyperparameters associated 
    δ::Real                 with the posterior distribution.
    risk_scale::Real        
"""
struct W2Posterior <: Posterior

    digital_device::DigitalDevice
    n_obs::Int
    event_obs::Vector{ObservationEvent}
    prior::Prior
    λ::Real
    δ::Real
    risk_scale::Real

    function W2Posterior(
            device::DigitalDevice,
            event_obs::Union{ObservationEvent, Vector{ObservationEvent}},
            prior::Prior; 
            λ::Real=10.0, δ::Real=2.0, risk_scale::Real=1.0
        )
        @assert λ > 0
        if isa(event_obs, ObservationEvent)
            _event_obs = [event_obs]
        else
            _event_obs = event_obs
        end
        n_obs = length(_event_obs)
        new(device, n_obs, _event_obs, prior, λ, δ, risk_scale)
    end

end



"""
    log_post(post, θ)

Evaluates the log of the W2-based posterior `post` at the point `θ`
"""
function Base.log(post::W2Posterior, θ::Vector{Float64})

    prior = post.prior
    event_obs = post.event_obs

    # Evaluate the (log) prior at θ
    logprior = log(prior, θ)

    # Wasserstein distance by observation event 
    n_obs = post.n_obs
    Φ = zeros(n_obs)
    Nt = Vector{Int}(undef,n_obs)
    for i = 1:n_obs
        
        # Post-processed observed data
        obs_data = event_obs[i].measured_populations_postprocessed
        if isnothing(obs_data)
            # Post-processing the first time we use this data for characterization
            obs_data = postprocess_event_wasserstein(
                            event_obs[i].measured_populations, 
                            δ=post.δ
                        )
            event_obs[i].measured_populations_postprocessed = obs_data
        end
        (Nx,Nt[i],Ny) = size(obs_data)

        # Forward evaluation for the given parameter setting
        t_grid, sim_data = eval_forward(event_obs[i], post.digital_device, θ)
        sim_data = reshape(sim_data, Nx, Nt[i], Ny)
        sim_data = postprocess_event_wasserstein(sim_data, δ=post.δ)

        # Wasserstein distance calc
        for j = 1:Nx
            for k = 1:Ny
                dⱼₖ = Wasserstein_trace_squared(
                            view(sim_data, j, :, k), 
                            view(obs_data, j, :, k), 
                            t_grid
                         )
                Φ[i] = Φ[i] + dⱼₖ
            end
        end 
        Φ[i] = Φ[i] / (Nx * Ny)
    end
    
    # Finalize posterior value
    return (
        -post.risk_scale * post.λ * mean(Φ .* Nt) + logprior, 
        Φ
    )

end