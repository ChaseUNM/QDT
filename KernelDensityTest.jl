using Distributions 
using KernelDensity
using LinearAlgebra 
using Plots
using Interpolations
# generate normal data to test the kernel density estimator 

# now test the beta distribution from Distributions 

# dist = Beta(2,2)
dist = Normal()

x = rand(dist, 1000)
# histogram(x, normalize = true)

function beta_kde_eval(x::Real, data::AbstractVector; h::Real)
    α = x / h + 1
    β = (1 - x) / h + 1

    return mean(pdf.(Beta(α, β), data))
end

# function beta_kde_eval2(X_i::Real, x::Real; h::Real)
#     α = x / h + 1
#     β = (1 - x) / h + 1

#     return pdf.(Beta(α, β), X_i)
# end

# function beta_kde_single(xs::AbstractVector, X_i::Real; h::Real)
#     return [beta_kde_eval2(X_i, x; h = h) for x in xs]
# end

function beta_kde(xs::AbstractVector, data::AbstractVector; h::Real)
    return [beta_kde_eval(x, data; h=h) for x in xs]
end


# a,b = minimum(x), maximum(x)
# data_mapped = (x .- a) ./ (b - a)

# xs = LinRange(0, 1, 1000)
# x_grid = a .+ (b - a) .* xs
# y_vec = beta_kde_single(xs, 0.1, h = 0.2)



# beta_kde_density = beta_kde(xs, data_mapped ,h = 0.01)
# beta_kde_density = beta_kde_density ./ (b - a)
# normalize this 
# beta_kde_density = beta_kde_density ./ trapezoidal_rule_beta(x_grid, beta_kde_density)
# x = rand(Normal(0, 1), 1000) 
# compute the kernel density estimate
# kde_ans = KernelDensity.kde(x, bandwidth = 0.1)

# plot the results

# plot(kde_ans.x, kde_ans.density, label="Kernel Density Estimate", xlabel="x", ylabel="Density", title="Kernel Density Estimation")
# histogram!(x, normalize=true, label="Data Histogram", alpha=0.5)

# down-weighting now to smooth the data out

# get kde_ans.x as a vector of n points 
# n = length(kde_ans.x)
# want to be able to apply quadrature to KDE using right-hand rule 
# function right_hand_rule(kde_ans)
#     n = length(kde_ans.x)
#     # compute the differences between consecutive x values
#     dx = diff(kde_ans.x)
#     # compute the right-hand rule approximation of the integral
#     integral = sum(kde_ans.density[2:end] .* dx)
#     return integral
# end

# compute the integral using the right-hand rule
# integral_kde = right_hand_rule(kde_ans)

# compute the integral using the trapezoidal rule for comparison
# function trapezoidal_rule(kde_ans::UnivariateKDE)
#     n = length(kde_ans.x)
#     # compute the differences between consecutive x values
#     dx = diff(kde_ans.x)
#     # compute the trapezoidal rule approximation of the integral
#     integral = sum((kde_ans.density[1:end-1] + kde_ans.density[2:end]) .* dx) / 2
#     return integral
# end

function trapezoidal_rule_beta(xs::AbstractVector, ys::AbstractVector)
    n = length(x)
    dx = diff(xs)
    integral = sum((ys[1:end - 1] + ys[2:end]) .* dx)/2
    return integral 
end


# integral_trapezoidal = trapezoidal_rule(kde_ans)

# println("Integral using right-hand rule: ", integral_kde)
# println("Integral using trapezoidal rule: ", integral_trapezoidal)
# println("Integral of beta KDE using trapezoidal rule: ", trapezoidal_rule_beta(x_grid, beta_kde_density))

# now down-weight the data, previous was done just to show that the integral of the KDE is approximately 1, as it should be for a probability density function.

# power = 0.2 # down-weighting factor
# down-weight the density values
# kde_ans_downweighted = deepcopy(kde_ans)
# beta_kde_downweighted = deepcopy(beta_kde_density)
# kde_ans_downweighted.density .= kde_ans.density .^ power
# beta_kde_downweighted = beta_kde_downweighted .^ power
# kde_ans_area = right_hand_rule(kde_ans_downweighted)
# beta_area = trapezoidal_rule_beta(x_grid, beta_kde_downweighted)
# println("Area under down-weighted KDE: ", kde_ans_area)
# println("Area under down-weighted beta KDE: ", beta_area)   

# normalize the down-weighted density to ensure it integrates to 1
# kde_ans_downweighted.density .= kde_ans_downweighted.density ./ kde_ans_area
# beta_kde_downweighted = beta_kde_downweighted ./ beta_area
# check the integral again after normalization
# integral_kde_downweighted_normalized = right_hand_rule(kde_ans_downweighted)
# integral_beta_kde_downweighted_normalized = trapezoidal_rule_beta(x_grid, beta_kde_downweighted)

# println("Integral of normalized down-weighted KDE: ", integral_kde_downweighted_normalized)
# println("Integral of normalized down-weighted beta KDE: ", integral_beta_kde_downweighted_normalized)
#plot different densities on the same plot
# kernel_plots = plot(kde_ans.x, kde_ans.density, label="Original KDE", xlabel="x", ylabel="Density", title="Kernel Density Estimation with Down-weighting")
# plot!(kde_ans_downweighted.x, kde_ans_downweighted.density, label="Down-weighted KDE (power = $power)", linestyle=:dash)
# plot!(x_grid, beta_kde_density, label = "Original Beta KDE")
# plot!(x_grid, beta_kde_downweighted, label = "Down-weighted beta KDE (power = $power)")
# histogram!(x, normalize=true, label="Data Histogram", alpha=0.5)

# # test the down-weighted integral to see if it is still approximately 1
# integral_kde_downweighted = right_hand_rule(kde_ans_downweighted)
# println("Integral of down-weighted KDE: ", integral_kde_downweighted)

# left and right boundaries of density kernel 
# left_boundary = kde_ans_downweighted.x[1] 
# right_boundary = kde_ans_downweighted.x[end]
# println("left boundary: $left_boundary")
# println("right boundary: $right_boundary")

# display(kernel_plots)

# construct both piece-wise constant and linear PDF from density vector 
# has compact support on [left_boundary, right_boundary]

function density_pw_constant(density_object::UnivariateKDE, x::Float64)
    x_vals = density_object.x
    y_vals = density_object.density 
    left_boundary = x_vals[1]
    right_boundary = x_vals[end]
    h = step(x_vals)
    N = length(x_vals)

    if !(left_boundary <= x <= right_boundary)
        throw("input x ($x) is not in interval of density: [$left_boundary, $right_boundary]")
    end

    for i in 1:N 
        # get the left and right ends of the interval, this only works for constant step size 
        interval_left = left_boundary + (i - 1)*h
        interval_right = left_boundary + i*h
        if interval_left <= x < interval_right 
            return y_vals[i]
            break 
        end
    end
    
    if x == right_boundary 
        return y_vals[end]
    end
end

function density_pw_linear(density_object::UnivariateKDE, x::Float64)
    x_vals = density_object.x 
    y_vals = density_object.density 
    left_boundary = x_vals[1]
    right_boundary = x_vals[end]
    h = step(x_vals)
    N = length(x_vals)
    if !(left_boundary <= x <= right_boundary)
        throw("input x ($x) is not in interval of density: [$left_boundary, $right_boundary]")
    end

    for i in 1:N 
        # get the left and right ends of the interval, this only works for constant step size 
        interval_left = left_boundary + (i - 1)*h
        interval_right = left_boundary + i*h
        if interval_left <= x <= interval_right 
            return y_vals[i]*(x - interval_right)/(interval_left - interval_right) + y_vals[i + 1]*(x - interval_left)/(interval_right - interval_left)
            break 
        end
    end
    
end


# combine everything into a function to return interpolant
function kernel_downweighting(x_samples::Vector{Float64}, bandwidth::Float64, downweight_power::Float64) 
    # create kernel 
    a, b = minimum(x_samples), maximum(x_samples)
    # map to [0,1] on 1000 points
    xs = LinRange(0, 1, 1000)
    x_grid = a .+ (b - a) .* xs

    # create beta density map back to [a,b] and normalize 
    beta_kde_density = beta_kde(xs, data_mapped, h = bandwidth)
    beta_kde_density = beta_kde_density ./ (b - a)
    beta_kde_density = beta_kde_density ./ trapezoidal_rule_beta(x_grid, beta_kde_density)
    # down-weight density 
    beta_kde_downweighted = beta_kde_density .^ downweight_power
    
    # re-normalize 
    beta_kde_downweighted = beta_kde_downweighted ./ trapezoidal_rule_beta(x_grid, beta_kde_downweighted)
    println("area of beta_kde_downweighted ", trapezoidal_rule_beta(x_grid, beta_kde_downweighted))
    # interpolate this 
    itp = interpolate(beta_kde_downweighted, BSpline(Quadratic(Line(OnGrid()))))
    itp_scaled = Interpolations.scale(itp, x_grid)
    return (f_pdf = itp_scaled, x_pts = x_grid)
end

output = kernel_downweighting(x, 0.1, 0.3)
f = output.f_pdf 
x_grid = output.x_pts
kernel_plot = plot(x_grid, f(x_grid), label = "Kernel")
histogram!(x, normalize = true, alpha = 0.5)

# combine into a function 
# function kernel_downweighting(x_samples::Vector{Float64}, bandwidth::Float64, downweight_power::Float64)

#     # create 
#     kde_ans = KernelDensity.kde(x_samples, bandwidth = bandwidth)
#     kde_ans_downweighted = deepcopy(kde_ans)
#     kde_ans_downweighted.density = kde_ans_density .^ downweight_power

# end
