# File: /Users/chase/QDT/characterization_control.jl
# Purpose: End-to-end characterization + control optimization loop for a single qudit.
# Notes:
# - This file runs an iterative loop: generate/optimize controls, run physical experiments (simulated),
#   infer parameter posteriors, update priors, and repeat until infidelity tolerance is met.
# - The script is written as a script (top-level); consider refactoring into functions for testability
#   and reusability (see improvement notes below).

using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, Printf
include("src/QDT.jl")
include("src/physical_qudit.jl")
include("src/prior.jl")
include("src/posterior.jl")
include("src/characterization.jl")

#====================================================================================
    PARAMETERS
====================================================================================#

# Parameter domain
ωmin = 4.0
ωmax = 5.0
ξmin = 0.1
ξmax = 0.3
param_domain = RectangularDomain([ωmin ωmax; ξmin ξmax])


# Parameters of the physical qudit
ω  = 4.62
ξ  = 0.18
Ne = 2
Ng = 0


# Rotating frame frequency --- defines the frequency of the control carrier wave
ω_rot = 4.5


# Measurement/SPAM settings
M_spam_order = 1e-4
n_readout_samples = 100000


# Control parametrization: B-splines
degree    = 2
n_splines = 10 
T         = 50
nsteps    = 250
dt        = T/nsteps
max_control_amplitude = 0.1


# MCMC Parameters
λ = 3.0;
n_samples = 10000
mcmc_burnin     = 2500
mcmc_thin       = 5
mcmc_iterations = mcmc_burnin + n_samples*mcmc_thin
# Initial parameter guesses
ω0 = 4.3
ξ0 = 0.26

# Prior downweighting power
downweight_power = 0.2

# Number of characterization+optimization iters to perform
max_iters = 2

# Target get infidelity
epsilon = 1e-4

#====================================================================================
    SETUP
====================================================================================#

# QuditControl used control the DigitalQudit and PhysicalQudit
control = FortranBSplineControl(degree, n_splines, T)

# DigitalQudit instance used for optimizing control signals
digital_q = DigitalQudit(Ne, Ng, ω, ξ, ω_rot, control)

# PhysicalQudit instance used for simulating real device outcomes
phys_q = PhysicalQudit(digital_q; M_spam_order=M_spam_order)



#====================================================================================
    CONSTANT-CONTROL CHARACTERIZATION
====================================================================================#

@printf("CONSTANT-CONTROL CHARACTERIZATION\n")

# Run the constant controls on the physical qudit, measuring noisy
# population data
control_coeffs = 0.5*max_control_amplitude*ones(control.N_coeff)
const_control_obs = run_control(phys_q, control_coeffs, n_readout_samples)

# Prior and posterior
prior     = UniformPrior(param_domain)
posterior = W2Posterior(digital_q, const_control_obs, prior; λ=λ)

# Run an initial W2-chain inference constant control data
α0 = [ω0; ξ0]
const_control_char_event = run_w2_chain(
                                posterior, α0,
                                iterations=mcmc_iterations,
                                burnin=mcmc_burnin,
                                thin=mcmc_thin
                            )

#====================================================================================
    POSTERIOR VISUALIZATION
====================================================================================#

n_omega = 2001
omegas = LinRange(ωmin+0.35, ωmax-0.35, n_omega)

prior_vs_omega = zeros(n_omega)
logpost_vs_omega = zeros(n_omega)
risk_vs_omega = zeros(n_omega)
for i = 1:n_omega
    prior_vs_omega[i] = exp(log(prior, [omegas[i], ξ]))
    (logpost, Φ) = log(posterior, [omegas[i], ξ])
    risk_vs_omega[i] = Φ[1]
    logpost_vs_omega[i] = logpost
end