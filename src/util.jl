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
    λ .= clamp.(real.(λ), min_eig, max_eig)
    return real.(V * diagm(λ) * V')
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
    carrier_freqs::Union{AbstractVector, Nothing}, 
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

function eigen_and_reorder(H0; verbose=false)

    Ntot = size(H0, 1)

    # Eigenvalue decomposition
    F = eigen(H0)

    # Sort eigenvalues in ascending order
    perm = sortperm(real(F.values))
    evals = F.values[perm]
    evects = F.vectors[:, perm]

    # Find the column containing the largest element in each row
    max_col = zeros(Int, Ntot)

    for row in 1:Ntot
        max_col[row] = argmax(abs.(evects[row, :]))
    end

    # Check for duplicate column assignments
    Ndup_col = 0
    for row in 1:Ntot-1
        for k in row+1:Ntot
            if max_col[row] == max_col[k]
                Ndup_col += 1
                println("Error: detected identical max_col = $(max_col[row]) for rows $row and $k")
            end
        end
    end

    if Ndup_col > 0
        error("Permutation of eigenvector matrix failed.")
    end

    # Reorder eigenvectors/eigenvalues
    evects = evects[:, max_col]
    evals = evals[max_col]

    # Make diagonal entries positive
    for j in 1:Ntot
        if real(evects[j,j]) < 0
            evects[:,j] .*= -1
        end
    end

    return evals, evects

end

function map_to_oscillators(id::Integer, Ne::AbstractVector{<:Integer},
                            Ng::AbstractVector{<:Integer})

    # Number of levels in each subsystem
    nlevels = Ne .+ Ng

    localIDs = Int[]

    # Convert to zero-based indexing internally
    index = id - 1

    for iosc in eachindex(Ne)

        postdim = isempty(nlevels[iosc+1:end]) ? 1 : prod(nlevels[iosc+1:end])

        push!(localIDs, div(index, postdim))

        index = mod(index, postdim)
    end

    return localIDs
end

gate_to_str(g) = g == PauliX ? "X" :
                g == PauliY ? "Y" :
                g == PauliZ ? "Z" :
                g == Hadamard ? "Hadamard" :
                g == Tgate ? "T" :
                g == IdentityGate ? "I" :
                string(g)

function gate_to_str(gate::ProductGate)
    left_gate = gate.left 
    right_gate = gate.right
    gate_str = gate_to_str(gate.left) * "⊗" * gate_to_str(gate.right)
    return gate_str 
end

function get_resonances(;
    Ne,
    Ng,
    Hsys,
    Hc_re = [],
    Hc_im = [],
    rotfreq = [],
    cw_amp_thres = 1e-7,
    cw_prox_thres = 1e-2,
    verbose = true,
    stdmodel = true
)

    if verbose
        println("\nComputing carrier frequencies, ignoring growth rate slower than ",
            cw_amp_thres,
            " and frequencies closer than ",
            cw_prox_thres,
            " [GHz]")
    end

    nqubits = length(Ne)
    n = size(Hsys,1)

    # Eigenvalues and reordered eigenvectors
    Hsys_evals, Utrans = eigen_and_reorder(Hsys, verbose = verbose)

    Hsys_evals = real.(Hsys_evals) ./ (2π)

    resonances = [Float64[] for _ in 1:nqubits]
    speed       = [Float64[] for _ in 1:nqubits]

    for q in 1:nqubits

        Hsym_trans  = Utrans' * Hc_re[q] * Utrans
        Hanti_trans = Utrans' * Hc_im[q] * Utrans

        resonances_a = Float64[]
        speed_a = Float64[]

        if verbose
            println("  Resonances in oscillator #", q)
        end

        for Hc_trans in (Hsym_trans, Hanti_trans)

            for i in 1:n
                for j in 1:i-1

                    abs(Hc_trans[i,j]) < 1e-14 && continue

                    delta_f = Hsys_evals[i] - Hsys_evals[j]

                    if abs(delta_f) < 1e-10
                        delta_f = 0.0
                    end

                    ids_i = map_to_oscillators(i, Ne, Ng)
                    ids_j = map_to_oscillators(j, Ne, Ng)

                    is_ess_i = all(ids_i[k] < Ne[k] for k in eachindex(Ne))
                    is_ess_j = all(ids_j[k] < Ne[k] for k in eachindex(Ne))

                    if is_ess_i && is_ess_j

                        if any(abs(delta_f - f) < cw_prox_thres for f in resonances_a)

                            if verbose
                                println("    Ignoring resonance from ",
                                    ids_j,
                                    " to ",
                                    ids_i,
                                    ", freq ",
                                    delta_f,
                                    ", growth rate = ",
                                    abs(Hc_trans[i,j]),
                                    " being too close to one that already exists.")
                            end

                        elseif abs(Hc_trans[i,j]) < cw_amp_thres

                            if verbose
                                println("    Ignoring resonance from ",
                                    ids_j,
                                    " to ",
                                    ids_i,
                                    ", freq ",
                                    delta_f,
                                    ", growth rate = ",
                                    abs(Hc_trans[i,j]),
                                    " growth rate is too slow.")
                            end

                        else

                            push!(resonances_a, delta_f)
                            push!(speed_a, abs(Hc_trans[i,j]))

                            if verbose
                                println("    Resonance from ",
                                    ids_j,
                                    " to ",
                                    ids_i,
                                    ", freq ",
                                    delta_f,
                                    ", growth rate = ",
                                    abs(Hc_trans[i,j]))
                            end
                        end
                    end
                end
            end
        end

        resonances[q] = resonances_a
        speed[q] = speed_a
    end

    # Prepare output
    Nfreq = zeros(Int, nqubits)

    om = Vector{Vector{Float64}}(undef, nqubits)
    growth_rate = Vector{Vector{Float64}}(undef, nqubits)

    for q in 1:nqubits

        Nfreq[q] = max(1, length(resonances[q]))

        if isempty(resonances[q])
            om[q] = zeros(Nfreq[q])
        else
            om[q] = copy(resonances[q])
        end

        if isempty(speed[q])
            growth_rate[q] = ones(Nfreq[q])
        else
            growth_rate[q] = copy(speed[q])
        end
    end

    return om, growth_rate
end