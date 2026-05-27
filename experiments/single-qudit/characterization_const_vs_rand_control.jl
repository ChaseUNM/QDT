# File: /Users/chase/QDT/characterization_control.jl
# Purpose: End-to-end characterization + control optimization loop for a single qudit.
# Notes:
# - This file runs an iterative loop: generate/optimize controls, run physical experiments (simulated),
#   infer parameter posteriors, update priors, and repeat until infidelity tolerance is met.
# - The script is written as a script (top-level); consider refactoring into functions for testability
#   and reusability (see improvement notes below).

using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, Printf, LaTeXStrings
include("../src/QDT.jl")
include("../src/physical_qudit.jl")
include("../src/prior.jl")
include("../src/posterior.jl")
include("../src/characterization.jl")


#====================================================================================
    PARAMETERS
====================================================================================#

# Parameters of the physical qudit
ω  = 4.62
ξ  = 0.19
Ne = 2
Ng = 0


# Parameter domain
ωmin = 4.0
ωmax = 5.0
ξmin = ξ-0.1
ξmax = ξ+0.2
param_domain = RectangularDomain([ωmin ωmax; ξmin ξmax])


# Rotating frame frequency --- defines the frequency of the control carrier wave
ω_rot = 4.5


# Measurement/SPAM settings
M_spam_order = 1e-4
n_readout_samples = 100000


# Control parametrization: B-splines
degree    = 2
n_splines = 10 
T         = 50
nsteps    = 200
dt        = T/nsteps
max_control_amplitude = 0.1
rand_control_seed = 56781


# MCMC Parameters
λ = 1.0;
n_samples       = 1000
mcmc_burnin     = 2500
mcmc_thin       = 3
mcmc_iterations = mcmc_burnin + n_samples*mcmc_thin
mcmc_seed = 1234
# Initial parameter guesses
ω0 = ω_rot
ξ0 = 0.26


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
    CONSTANT CONTROL CHARACTERIZATION
====================================================================================#

@printf("CONSTANT CONTROL CHARACTERIZATION\n")

# Run the constant controls on the physical qudit, measuring noisy
# population data
N_coeff = control.N_coeff
control_coeffs = zeros(N_coeff)
control_coeffs[1:Int(N_coeff/2)] .= 0.5*max_control_amplitude
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
                                thin=mcmc_thin,
                                rng_seed=mcmc_seed
                            )


#====================================================================================
    RANDOM CONTROL CHARACTERIZATION
====================================================================================#

@printf("RANDOM CONTROL CHARACTERIZATION\n")

# Generate and run random controls on the physical qudit
rng = Xoshiro(rand_control_seed)
control_coeffs .= max_control_amplitude * (0.5 .- rand(rng, N_coeff))
rand_control_obs = run_control(phys_q, control_coeffs, n_readout_samples)

# Posterior based on the random control data
posterior_rand_control = W2Posterior(digital_q, rand_control_obs, prior; λ=λ)

# Run an initial W2-chain inference using the random control data
rand_control_char_event = run_w2_chain(
                                posterior_rand_control, 
                                α0,
                                iterations=mcmc_iterations,
                                burnin=mcmc_burnin,
                                thin=mcmc_thin,
                                rng_seed=mcmc_seed
                            )


#====================================================================================
    POSTERIOR VS OMEGA
====================================================================================#

n_omega = 1201
omegas = LinRange(ωmin+0.25, ωmax-0.25, n_omega)

const_logpost = zeros(n_omega)
const_risk    = zeros(n_omega)
rand_logpost = zeros(n_omega)
rand_risk    = zeros(n_omega)
for i = 1:n_omega
    # Constant control
    (logpost, Φ) = log(posterior, [omegas[i], ξ])
    const_risk[i] = Φ[1]
    const_logpost[i] = logpost
    # Random control
    (logpost, Φ) = log(posterior_rand_control, [omegas[i], ξ])
    rand_risk[i] = Φ[1]
    rand_logpost[i] = logpost
end

#====================================================================================
    PLOT RISK vs. OMEGA
====================================================================================#

f = plot(
    linewidth=2,
    xlabel=L"Digital Qubit Frequency $\omega$",
    xlabelfontsize=14,
    xguidefontsize=12,
    ylabel=L"Empirical Risk $\Phi$ (Wasserstein)",
    ylabelfontsize=14,
    yguidefontsize=12,
    yscale=:log10,
    dpi=256
)

plot!(omegas, rand_risk, linewidth=2, label="Random Control")
plot!(omegas, const_risk, linewidth=2, label="Const. Control")

savefig(f, "const_vs_rand_control_W2_risk.svg")


#====================================================================================
    PLOT POSTERIOR vs. OMEGA
====================================================================================#

f = plot(
    linewidth=2,
    xlabel=L"Digital Qubit Frequency $\omega$",
    xlabelfontsize=14,
    xguidefontsize=12,
    ylabel="Posterior (unnormalized)",
    ylabelfontsize=14,
    yguidefontsize=12,
    dpi=256
)

plot!(omegas, exp.(rand_logpost), label="Random Control")
plot!(omegas, exp.(const_logpost), label="Const. Control")

savefig(f, "const_vs_rand_control_posterior.svg")

