using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.PlotMeasures, LaTeXStrings
include("../../src/QDT.jl")

#====================================================================================
    PARAMETERS
====================================================================================#

# Parameters of the physical qudits
ω1 = 4.5
ω2 = 4.6
ξ12 = 0.1
Ne = 2      # DO NOT CHANGE THESE. Ne > 2 and Ng > 0 are not supported
Ng = 0      # for DigitalQubitPair instances.


# Rotating frame frequency --- defines the frequency of the control carrier wave
ω_rot = 0.5*(ω1+ω2)


# Control parametrization: B-splines
degree    = 2
n_splines = 10 
dt        = 0.25
T         = 50.0
carrier_freqs = [0, ξ12, 2*ξ12]
max_control_amplitude = 0.1
seed = 45634


#====================================================================================
    CARRIER-CONTROL RISK LANDSCAPE
====================================================================================#

digital_q_pair = DigitalQubitPair(ω1, ω2, ξ12, ω_rot)
phys_device = PhysicalDevice(digital_q_pair; M_spam_order=0.0)

# Creating the carrier control
base_control = FortranBSplineControl(degree, n_splines, T)
controller = Vector{AbstractControl}([
                    CarrierControl(base_control, (ω1-ω_rot) .- carrier_freqs),
                    CarrierControl(base_control, (ω2-ω_rot) .- carrier_freqs) 
                ])
rng = Xoshiro(seed)
n_freq = length(carrier_freqs)
control_coeffs = max_control_amplitude * rand(rng, 2 * 2*n_splines*n_freq)

# Run the controls on the physical qubit pair, measuring noisy
# population data
n_readout_samples = 100000
control_obs = run_control(
                    phys_device, 
                    controller, 
                    control_coeffs, 
                    n_readout_samples, 
                    dt=dt
                )
                    
# Prior and posterior --- needed for W2 distance evaluation
prior     = UniformPrior(RectangularDomain([4.0 5.0; 4.0 5.0; 0.0 1.0]))
posterior = W2Posterior(digital_q_pair, control_obs, prior)

# Evaluate W2 distance over (ω₁,ω₂) range
ω_min = 4.3
ω_max = 4.7
n_ωᵢ = 100
ω_range = LinRange(ω_min,ω_max,n_ωᵢ)

W2_landscape = zeros(n_ωᵢ,n_ωᵢ)
θ = [ω1, ω2, ξ12]
λ = 1.0
for i = 1:n_ωᵢ
    @printf("i = %d of %d\n", i, n_ωᵢ)
    for j = 1:n_ωᵢ
        θ[1] = ω_range[i]
        θ[2] = ω_range[j]
        W2_landscape[i,j] = log(posterior, θ, λ)[2][1]
    end
end

#====================================================================================
    PLOTTING THE LOSS-LANDSCAPE
====================================================================================#

f = heatmap(
    ω_range, ω_range, log10.(W2_landscape)', 
    xlabel = L"$ω_1$", ylabel = L"$ω_2$", 
    xguidefontsize=18, yguidefontsize=18,
    tickfontsize=12, left_margin = 2mm, right_margin = 8mm,
    title = L"log W2 Loss vs. $(ω_1, ω_2)$ at the true $ξ_{12}$",
    titlefontsize=20,
    dpi=512, size=(800,600)
)
