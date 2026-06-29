using ValueHistories

function get(hist::History{I,V}, iter::I=-1) where {I <: Real, V} 
    # Returns the value at iteration 'iter' of the history 
    # If iter = -1, returns the last set specifically
    if iter == -1
        return hist.values[end]
    else
        idx = findfirst(hist.iterations .== iter)
        if idx === nothing
            throw("Iteration not found!")
        end
        return hist.values[idx]
    end
end

function make_unique_folder(base_name::String, overwrite::Bool = false)
    tag = 1
    folder = "$(base_name)_$(tag)"
    
    while isdir(folder)
        tag += 1
        folder = "$(base_name)_$(tag)"
    end

    mkpath(folder)

    return folder, tag
end

function update!(hist::History{I,V}, iter::I, value::V) where {I <: Real, V} 
    # Updates the value in the history associated with 
    # timestamp/iteration 'iter'. Set 'iter' to -1 to 
    # update to the latest timestamp.
    if iter == -1
        return hist.values[end] = value
    else
        idx = findfirst(hist.iterations .== iter)
        if idx === nothing
            throw("Iteration not found!")
        end
        hist.values[idx] = value
    end
end

# Used for plotting subscripts on labels
_subdigits = Dict('0'=>'₀','1'=>'₁','2'=>'₂','3'=>'₃','4'=>'₄','5'=>'₅','6'=>'₆','7'=>'₇','8'=>'₈','9'=>'₉')
function int_to_subscript(n::Integer)
    s = string(n)
    out = IOBuffer()
    for c in s
        print(out, Base.get(_subdigits, c, c))
    end
    return String(take!(out))
end

function clamp_diag(A::Matrix{<:Real}; cutoff::Real = 1E-10)
    A[diagind(A)] .= max.(diag(A), cutoff)
    return A 
end


function clamp_eigs(A::Matrix{<:Real}; cutoff::Real = 1E-10)
    F = eigen(A)
    λ = F.values 
    V = F.vectors 
    λ .= max.(λ, cutoff)
    return V * diagm(λ) * V'
end

function clamp_eigs(A::Matrix{<:Real}; min_eig::Real = 1E-10, max_eig::Real=1.0)
    F = eigen(A)
    λ = F.values 
    V = F.vectors 
    λ .= clamp.(λ, min_eig, max_eig)
    return V * diagm(λ) * V'
end

function true_posterior(event_obs_total,
    prior::Union{Nothing, NamedTuple, Distribution},
    ω::Real, 
    ωr::Real, 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    total_data_count,
    λ::Real=10.0)

    Φ, Φ_sum = trace_wasserstein_squared_loss_quantum(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal_total, event_obs_total, total_data_count, δ = 2.0)
    f = exp(-length(event_obs_total[1][1,:,1])*λ*Φ)*pdf(prior, ω)
    f_sum = exp(-length(event_obs_total[1][1,:,1])*λ*Φ_sum)*pdf(prior, ω)
    return f, f_sum, Φ, Φ_sum
end

function true_posterior_multi(event_obs_total,
    prior::AbstractVector{<:Union{Nothing, NamedTuple, Distribution}},
    ω::Vector{<:Real}, 
    ωr::Vector{<:Real}, 
    degree::Vector{<:Real}, 
    n_splines::Vector{<:Real}, 
    U0, 
    T::Vector{<:Real}, 
    nsteps::Vector{<:Real}, 
    pcof_optimal_total,
    total_data_count,
    λ::Real=10.0)
    # println("Parameters: ", ω)
    # calculate the prior as a product of joint distributions 
    prior_val = 1.0
    for i in 1:length(ω)
        prior_val *= pdf(prior[i], ω[i])
    end
    Φ, Φ_sum = trace_wasserstein_squared_loss_quantum_multi(ω, ωr, degree, n_splines, U0, T, nsteps, pcof_optimal_total, event_obs_total, total_data_count, carrier_freqs, δ = 2.0)
    f = exp(-length(event_obs_total[1][1,:,1])*λ*Φ)*prior_val
    f_sum = exp(-length(event_obs_total[1][1,:,1])*λ*Φ_sum)*prior_val
    return f, f_sum, Φ, Φ_sum
end

function interpolate_function(xs, ys)
    itp = interpolate(ys,BSpline(Quadratic(Line(OnGrid()))))
    itp_scaled = Interpolations.scale(itp, xs)
    f = extrapolate(itp_scaled, 0.0)
    return f 
end

function normalize_function(xs, ys)
    itp = interpolate(ys,BSpline(Quadratic(Line(OnGrid()))))
    itp_scaled = Interpolations.scale(itp, xs)
    f = extrapolate(itp_scaled, 0.0)
    area = gauss_integral(f, xs)
    f_normalized = x -> f(x) / area
    return f_normalized, area
end

function max_tag(dir::String)
    max_val = -Inf

    for f in readdir(dir)
        m = match(r"_(\d+)\.jld2$", f)  # captures digits before .jld2
        if m !== nothing
            tag = parse(Int, m.captures[1])
            max_val = max(max_val, tag)
        end
    end

    return max_val == -Inf ? nothing : max_val
end