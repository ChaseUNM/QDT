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
Ng = 1


# Rotating frame frequency --- defines the frequency of the control carrier wave
ω_rot = 4.5


# Measurement/SPAM settings
M_spam_order = 1e-4
n_readout_samples = 100000


# Control parametrization: B-splines
degree    = 2
n_splines = 10 
T         = 40
nsteps    = 200
dt        = T/nsteps
max_control_amplitude = 0.1


# Gate set
gates = [PauliX, PauliY, Hadamard]
gate_set = [unitary(g) for g = gates]
N_gates = length(gates)


# MCMC Parameters
n_samples = 50
mcmc_burnin     = 2500
mcmc_thin       = 10
mcmc_iterations = mcmc_burnin + n_samples*mcmc_thin
# Initial parameter guesses
ω0 = 4.3
ξ0 = 0.26

# Prior downweighting power
downweight_power = 0.2

# Number of characterization+optimization iters to perform
max_iters = 5


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

# Initial prior and posterior
init_prior     = UniformPrior(param_domain)
init_posterior = W2Posterior(digital_q, const_control_obs, init_prior)

# Run an initial W2-chain inference constant control data
α0 = [ω0; ξ0]
const_control_char_event = run_w2_chain(
                                init_posterior, α0,
                                iterations=mcmc_iterations,
                                burnin=mcmc_burnin,
                                thin=mcmc_thin
                            )


#====================================================================================
    MAIN OPTIMIZATION + RE-CHARACTERIZATION LOOP

Alternating between risk-neutral optimization and characterization
====================================================================================#

opt_events  = Matrix{OptimizationEvent}(undef, max_iters, N_gates)
obs_events  = Matrix{ObservationEvent}(undef, max_iters, N_gates)
char_events = Vector{CharacterizationEvent}(undef, max_iters+1)
char_events[1] = const_control_char_event

# Initial, random control coefficients ("betas") for each gate
control_coeffs = (0.5 .- rand(control.N_coeff,N_gates)) * max_control_amplitude


for i in 1:max_iters

    @printf("OPTIMIZATION + RE-CHARACTERIZATION LOOP, ITER %d\n", i)

    # Set the digital qudit to use the parameter samples generated during
    # the previous characterization event
    set_parameters(digital_q, char_events[i].samples)


    # Optimize each of the gates
    for j = 1:N_gates
        @printf("  Optimizing gate %s ...\n", string(gates[j]))

        # Run the optimization loop
        opt_events[i,j] = optimize_control(digital_q, control_coeffs[:,j], gates[j],
                                           max_amplitude=max_control_amplitude)
        control_coeffs[:,j] = opt_events[i,j].control_coeffs

        # Evaluate the optimized controls on the physical qudit
        @printf("  ... Evaluating gate %d, ", string(gates[j]))
        obs_events[i,j] = run_control(phys_q, control_coeffs[:,j], 
                                      n_readout_samples; target_gate=gates[j])
        @printf("Measured Fidelity = %.2e\n", obs_event.measured_infidelity)
    end

    # Check termination condition: both measured infidelities below epsilon for all gates
    if all([obs_events[i,j].measured_infidelity < epsilon for j = 1:N_gates])
        println("TERMINATING, ALL MEASURED FIDELITIES BELOW ϵ = %.2e\n", epsilon)
        break 
    end

    @printf("  Begining re-characterization post optimization ... \n")

    # Build a new Truncated Gaussian prior from the most recently generated 
    # parameter samples
    prior = TructGaussianPrior(char_events[i].samples, downweight_power, param_domain)
    
    # Collect all previous observation events into an array
    event_obs_i = [const_control_obs; reshape(obs_events[1:i,:], :)]

    # New posterior based on the new prior and all previous observations
    posterior = W2Posterior(digital_q, event_obs_i, prior)

    @printf("  ... Running MCMC to sample new postering\n")
    char_events[i+1] = run_w2_chain(
                            posterior, prior.μ,
                            iterations=mcmc_iterations,
                            burnin=mcmc_burnin,
                            thin=mcmc_thin
                        )
    @printf("  ... Done.\n")
end
