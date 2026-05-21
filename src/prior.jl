include("QDT.jl")

using Statistics
using SpecialFunctions

"""
    RectangularDomain struct

Represents the rectangular domain [a₁,b₁] × ⋯ × [aₙ,bₙ]

Fields

    n_params::Int               Dimension / number of parameters
    
    bounds::Matrix{Float64}     Parameter bounds by parameter
                                    aᵢ := bounds[i,1]  
                                    bᵢ := bounds[i,2] 
                                These specify the min and max for parameter 
                                i, respectively.

    vol::Float64                (hyper)volume of the domain,
                                i.e. vol = ∏ⱼ |bⱼ-aⱼ|
"""
struct RectangularDomain <: Domain
    n_params::Int
    bounds::Matrix{Float64}
    vol::Float64

    function RectangularDomain(bounds::Matrix{Float64})
        n = size(bounds,1)
        @assert(size(bounds,2) == 2)
        @assert(all(bounds[:,2] .> bounds[:,1]))
        vol = prod(bounds[:,2] - bounds[:,1])
        new(n, bounds, vol)
    end
end


"""
    contains(domain, x)

Returns true if the point `θ` belongs to the RectangularDomain `domain`
"""
function contains(domain::RectangularDomain, θ::AbstractVector{Float64})
    return all(θ .<= domain.bounds[:,2] .&& θ .>= domain.bounds[:,1])
end


"""
    standardize(domain, x)

Maps the point `x` in the given `domain` to the corresponding
point `y` in [0,1] × ⋯ × [0,1].
"""
function standardize(domain::RectangularDomain, x::AbstractVector{Float64})
    return (x - domain.bounds[:,1]) ./ (domain.bounds[:,2] - domain.bounds[:,1])
end


"""
    standardize(domain, X)

Maps the points `X[:,i]` in the given `domain` to the corresponding
point `Y[:,i]` in [0,1] × ⋯ × [0,1].
"""
function standardize(domain::RectangularDomain, X::AbstractMatrix{Float64})
    return (X .- domain.bounds[:,1]') ./ (domain.bounds[:,2] - domain.bounds[:,1])'
end



"""
    UniformPrior struct

Represents a uniform prior over the rectangular domain
[a₁,b₁] × ⋯ × [aₙ,bₙ]

Fields

    support::RectangularDomain      Defines the region of nonzero
                                    support for the prior
"""
struct UniformPrior <: Prior

    support::RectangularDomain

    function UniformPrior(support::RectangularDomain)
        new(support)
    end

    function UniformPrior(bounds::Matrix{Float64})
        support = RectangularDomain(bounds)
        new(support)
    end
end


"""
    Prior evaluation log p(θ)
    
Evaluates the prior logpdf `p` at the parameter `θ`
"""
function Base.log(p::UniformPrior, θ::AbstractVector{Float64})
    if !contains(p.support, θ)
        return -Inf
    else
        return -log(p.support.vol)
    end
end



"""
    TructGaussianPrior struct

Represents a truncated Gaussian distribution prior over the rectangular domain
Ω = [a₁,b₁] × ⋯ × [aₙ,bₙ].

    ρ(θ) ∝  0                               if θ ∉ Ω
    ρ(θ) ∝  exp[-α/2 (θ-μ)' Σ⁻¹ (θ-μ)]      if θ ∈ Ω

Fields

    μ::Vector{Float64}              Mean / center of the distribution

    Σinv::Matrix{Float64}           Inverse covariance matrix

    α::Float64                      Downweighting power

    support::RectangularDomain      Defines the region of nonzero
                                    support for the prior
"""
struct TructGaussianPrior <: Prior
    μ::Vector{Float64}
    Σinv::Matrix{Float64}
    α::Float64
    support::RectangularDomain
end

"""
Returns a `TructGaussianPrior` from a collection of samples
from another multivariate distribution, assumed to be (nearly) Gaussian.

    samples::Matrix{Float64}        samples[:,i] denotes the i-th sample

    support::RectangularDomain      Domain to which this distribution 
                                    will be truncated

    downweight_power::Float64       Exponent α. Smooths the distribution 
                                    if α ∈ (0,1)
"""
function TructGaussianPrior(
        samples::Matrix{Float64}, 
        support::RectangularDomain, 
        downweight_power::Float64=0.2
    )
    μ = mean(samples, dims=2)
    Σ = cov(samples)
    Σinv = inv(Σ) 
    new(μ, Σinv, downweight_power, support)
end


"""
    Prior evaluation log p(θ)
    
Evaluates the log pdf of the prior `p` at the parameter `θ`.

WARNING: The pdf computed here is *unnormalized* because we do not
         explicitly compute the effect of truncation to a finite 
         domain.
"""
function Base.log(p::TructGaussianPrior, θ::AbstractVector{Float64})
    if !contains(p.support, θ)
        return -Inf
    else
        return -0.5 * p.α * (θ-p.μ)' * p.Σinv * (θ-p.μ)
    end
end




"""
    KDEPrior struct

Represents the downweighted distribution

        p(θ) ∝ [ 1/N ∑ᵢ Kₕ(Θ; θⁱ) ] ^ α 

where θⁱ, i = 1, ..., N, denote i.i.d. samples from
some distribution π(θ), and Kₕ denotes a *kernel function*
with bandwidth h. 

Current options for the kernel function:

    "beta"          Kₕ(θ;θⁱ) ∝ ∏ⱼ fᵦ(θⁱⱼ; 1+θⱼ/h,1+(1-θⱼ)/h)

                    where   fᵦ(x;a,b) = xᵃ⁻¹(1-x)ᵇ⁻¹ / Beta(a,b)
                            Beta(a,b) = Γ(a)Γ(b)/Γ(a+b)
                            Γ(t) <--- Gamma function 

    "gaussian"      Kₕ(θ;θⁱ) ∝ exp(-‖θ-θⁱ‖² / h²)



Fields

    support::RectangularDomain      Defines the region of nonzero
                                    support for the prior

    kernel::String                  Specifies the kernel function. 
                                    Options: "beta" or "gaussian"

    bandwidth::Float64              Bandwidth of the kernel functions.

    downweight_power::Float64       Exponent α
    
    samples::Matrix{Float64}        Parameter samples used to define the 
                                    distribution via KDE.
                                    samples[:,i] denotes the i-th sample

    A::Matrix{Float64}              A[i,j] = log(S[i,j]) where S[:,j] denotes
                                    sample[:,j] `standardized` to the domain
                                    [0,1] × ⋯ × [0,1].

    B::Matrix{Float64}              B[i,j] = log(1-S[i,j]) where S[i,j] is as 
                                    defined above for A[i,j]

"""
struct KDEPrior

    support::RectangularDomain
    kernel::String
    bandwidth::Float64
    downweight_power::Float64
    samples::Matrix{Float64}
    A::Matrix{Float64}
    B::Matrix{Float64}

    function KDEPrior(
                bounds::Matrix{Float64}, 
                samples::Matrix{Float64};
                kernel::Symbol=:beta,
                bandwidth::Float64=0.1,
                downweighting::Float64=0.2
        )
        kernel = lowercase(kernel)
        @assert(kernel == :beta || kernel == :gaussian)
        support = RectangularDomain(n, bounds)
        S = standardize(domain,samples)
        new(support, 
            kernel, 
            bandwidth, 
            downweight_power, 
            samples,
            log.(S),
            log.(1 .- S)
        )
    end
end


"""
    KDE Prior evaluation log p(θ)
    
Evaluates the KDEPrior's logpdf at the parameter `θ`, up
to a constant shift. Returns, in particular, 

                α log( ∑ᵢ Kₕ(Θ; θⁱ) )     
"""
function Base.log(p::KDEPrior, θ::AbstractVector{Float64})

    n = length(θ)
    h = p.bandwidth
    α = p.downweight_power
    S = p.samples
    
    # Case 1: Outside the support
    if !contains(p.domain, θ)
        return -Inf

    # Case 2: beta kernel
    elseif p.kernel == :beta
        # Transform all variables to domain [0,1]
        θtilde = standardize(p.domain, θ)
        # Compute kernel parameters 
        a = θtilde/h            # vectors of size `n_params`
        b = (1 .- θtilde)/h
        #
        # Evaluate the kernels
        #  
        #   γᵢ := ∑ⱼ[aⱼlogθᵢⱼ + bⱼlog(1-θᵢⱼ)]
        #              ------      ---------
        #             p.A[i,j]      p.B[i,j]  
        #
        γ = sum( (p.A .* a') + (p.B .* b'), dims=2)
        #
        #  log Kₕ(θ;θⁱ) = ∑ⱼ[aⱼlogθᵢⱼ + bⱼlog(1-θᵢⱼ) - logbeta(aⱼ,bⱼ)] 
        #               = γᵢ - ∑ⱼ logbeta(aⱼ,bⱼ)
        #
        #  ∑ᵢ Kₕ(θ;θⁱ) = ∑ᵢ exp(log Kₕ(θ;θⁱ)) = (∑ᵢ exp γᵢ) / ∏ⱼ beta(aⱼ,bⱼ)
        #
        #  α log(∑ᵢ Kₕ(θ;θⁱ)) = α log[∑ᵢ exp(γᵢ)] - α ∑ⱼ logbeta(aⱼ,bⱼ)
        #
        return α * log(sum(exp.(γ))) - α * sum(logbeta.(a,b))

    # Case 3: gaussian kernel
    else
        return -α * norm(p.samples-θ')^2 / (h.^2)
    end
end


