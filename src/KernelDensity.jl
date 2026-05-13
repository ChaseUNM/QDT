using Distributions 
using KernelDensity
using LinearAlgebra 
using Plots
using Interpolations
using FastGaussQuadrature

# Kernel density utilities and custom KDE with downweighting
# Note: comments added for clarity and to flag potential issues; no code lines were changed.

using Distributions 
using KernelDensity
using LinearAlgebra 
using Plots
using Interpolations
using FastGaussQuadrature


# Evaluate a Beta-kernel based KDE at a single point x given by: 
# Beta kernel estimators for density functions — Song Xi Chen
# https://doi.org/10.1016/S0167-9473(99)00010-9

# x is assumed to be in [0,1] when used with the mapping logic in kernel_downweighting.
# h is a bandwidth parameter controlling the Beta shape: α = x/h + 1, β = (1-x)/h + 1.
# data should be points in the same domain as x (likely mapped to [0,1]).
function beta_kde_eval(x::Real, data::AbstractVector; h::Real)
    α = x / h + 1
    β = (1 - x) / h + 1

    return mean(pdf.(Beta(α, β), data))
end

# Evaluate the Beta-kernel KDE over a vector of xs.
# Returns a vector with the same length as xs.
function beta_kde(xs::AbstractVector, data::AbstractVector; h::Real)
    return [beta_kde_eval(x, data; h=h) for x in xs]
end

# Trapezoidal rule for numerical integration of ys over xs.
# NOTE: there is a likely bug here: `n = length(x)` references `x` which is undefined in this scope.
# Intended variable is probably `xs` and `n` is unused afterwards. The implementation uses diff(xs) so
# ensure xs has strictly increasing values. Also check that ys length matches xs.
function trapezoidal_rule_beta(xs::AbstractVector, ys::AbstractVector)
    n = length(xs)
    dx = diff(xs)
    integral = sum((ys[1:end - 1] + ys[2:end]) .* dx)/2
    return integral 
end

# combine everything into a function to return interpolant
# - x_samples: raw samples in original domain [a,b]
# - bandwidth: smoothing parameter (for Beta kernel it's used in beta_kde)
# - downweight_power: exponent applied to the density for downweighting
# - kernel_func: "Beta" or "Normal" controls which kernel implementation to use
# - sample_min, sample_max: optional overrides for the sample range [a,b]

function kernel_downweighting(x_samples::Vector{Float64}, bandwidth::Float64, downweight_power::Float64; kernel_func::String = "Beta", sample_min::Union{Nothing, Float64} = nothing, sample_max::Union{Nothing, Float64} = nothing) 
    # create kernel 
    # a, b = minimum(x_samples), maximum(x_samples)
    a, b = isnothing(sample_min) ? minimum(x_samples) : sample_min, isnothing(sample_max) ? maximum(x_samples) : sample_max
    # map to [0,1] on 2048 points to match with KernelDensity.jl default number of points for KDE
    xs = LinRange(0, 1, 2048)
    x_grid = a .+ (b - a) .* xs


    if kernel_func == "Beta"
        # create beta density map back to [a,b] and normalize 
        # could also try mapping entire interval [sample_min, sample_max] to [0,1]
        beta_kde_density = beta_kde(xs, data_mapped, h = bandwidth)
        beta_kde_density = beta_kde_density ./ (b - a)
        discrete_density = beta_kde_density ./ trapezoidal_rule_beta(x_grid, beta_kde_density)
        # down-weight density 

    elseif kernel_func == "Normal"
        if isnothing(sample_min) && isnothing(sample_max)
            normal_kde = KernelDensity.kde(x_samples, bandwidth = bandwidth)
        else
            normal_kde = KernelDensity.kde(x_samples, bandwidth = bandwidth, boundary = (sample_min, sample_max))
        end
        normal_density = normal_kde.density
        println("Area under density before: ", trapezoidal_rule_beta(x_grid, normal_density))
        discrete_density = normal_density ./ trapezoidal_rule_beta(x_grid, normal_density)
        println("Area under density after: ", trapezoidal_rule_beta(x_grid, discrete_density))
    end
    density_downweighted = discrete_density .^ downweight_power
    # re-normalize 

    density_downweighted = density_downweighted ./ trapezoidal_rule_beta(x_grid, density_downweighted)
    # interpolate this 
    itp = interpolate(density_downweighted,BSpline(Quadratic(Line(OnGrid()))))

    # scale the interpolant to the original x range [a,b]
    itp_scaled = Interpolations.scale(itp, x_grid)

    # set interpolant to zero outside of the original data range [a,b]
    f = extrapolate(itp_scaled, 0.0)
    return (f_pdf = f, x_grid = x_grid)
end

# Compute expected value (integral of x * f_pdf(x)) using Gauss-Legendre quadrature on each sub-interval.
# f_pdf is expected to be callable element-wise, e.g. an interpolant such that f_pdf.(xq) works.
# nquad is number of quadrature points per subinterval.
function expected_value_piecewise_gauss(f_pdf, x_grid; nquad=5)
    ξ, w = gausslegendre(nquad)

    integral = 0.0

    for j in 1:length(x_grid)-1
        a = x_grid[j]
        b = x_grid[j+1]

        # map [-1,1] -> [a,b]
        xq = @. (b - a)/2 * ξ + (a + b)/2
        wq = @. (b - a)/2 * w

        integral += sum(wq .* xq .* f_pdf.(xq))
    end

    return integral
end

# Compute integral of f_pdf over x_grid using Gauss-Legendre quadrature on each sub-interval.
function gauss_integral(f_pdf, x_grid; nquad=5)
    ξ, w = gausslegendre(nquad)

    integral = 0.0

    for j in 1:length(x_grid)-1
        a = x_grid[j]
        b = x_grid[j+1]

        # map [-1,1] -> [a,b]
        xq = @. (b - a)/2 * ξ + (a + b)/2
        wq = @. (b - a)/2 * w

        integral += sum(wq .* f_pdf.(xq))
    end

    return integral
end

