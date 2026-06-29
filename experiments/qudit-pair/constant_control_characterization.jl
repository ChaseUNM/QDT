using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.PlotMeasures, LaTeXStrings
include("../../src/QDT.jl")

#====================================================================================
    PARAMETERS
====================================================================================#

# Parameters of the physical qudits
ω1 = 4.62
ω2 = 4.43
ξ12 = 0.14
Ne = 2      # DO NOT CHANGE THESE. Ne > 2 and Ng > 0 are not supported
Ng = 0      # for DigitalQubitPair instances.
ξ_self = 0.0


# Parameter domain
ωmin = 4.0
ωmax = 5.0
ξmin = 0.05
ξmax = 0.21
param_domain = RectangularDomain([ωmin ωmax; ωmin ωmax; ξmin ξmax])


# Rotating frame frequency --- defines the frequency of the control carrier wave
ω_rot = 4.5


# Measurement/SPAM settings
M_spam_order = 1e-4
n_readout_samples = 100000


# Control parametrization: B-splines
degree    = 2
n_splines = 8 
dt        = 0.25
T_const_ctrl = 25.0
T_1q_gate    = 50.0
T_2q_gate    = 200.0
nsteps_1q_gate = round(Int, T_1q_gate/dt)
max_control_amplitude = 0.1
ipopt_options = ["max_iter" => 50, 
                 "print_level" => 5, 
                 "limited_memory_max_history" => 250
                ]
# Number of samples to use for optimization
n_samples_opt_1q = 5

# Gate set
one_qubit_gates  = [PauliX, PauliZ]


# MCMC Parameters
λ0              = 1.0;
n_samples_const_ctrl = 256
mcmc_burnin     = 1000
mcmc_thin       = 5
mcmc_seed       = 314159
mcmc_rng        = Xoshiro(mcmc_seed)

# Initial parameter guesses
#      ω1   ω2  ξ12
α0 = [4.2; 4.5; 0.1]


# Prior downweighting power
downweight_power = 0.1


# Target get infidelity
epsilon = 1e-4


#====================================================================================
    SETUP THE DEVICES
====================================================================================#

# DigitalQudit instance used for optimizing single qubit control signals
digital_q  = DigitalQudit(Ne, Ng, ω1, ξ_self, ω_rot)

# DigitalQubitPair instance used for optimizing two qubit control signals
digital_q_pair = DigitalQubitPair(ω1, ω2, ξ12, ω_rot)

# PhysicalDevice instance used for simulating real device outcomes
phys_device = PhysicalDevice(digital_q_pair; M_spam_order=M_spam_order)


#====================================================================================
    CONSTANT-CONTROL CHARACTERIZATION (BOTH QUBITS TOGETHER)
====================================================================================#

@printf("CONSTANT CONTROL CHARACTERIZATION\n")

# Creating a controller with constant output
N_amp = 1
const_controller = Vector{AbstractControl}([
                            GRAPEControl(N_amp, T_const_ctrl), 
                            GRAPEControl(N_amp, T_const_ctrl)
                    ])
const_control_coeffs = max_control_amplitude * [1.0, 0.0, 1.0, 0.0];

# Run the constant controls on the physical qubit pair, measuring noisy
# population data
const_control_obs = run_control(
                        phys_device, 
                        const_controller, 
                        const_control_coeffs, 
                        n_readout_samples, 
                        dt=dt
                    )
                    
# Prior and posterior
init_prior     = UniformPrior(param_domain)
init_posterior = W2Posterior(digital_q_pair, const_control_obs, init_prior)

# Run an initial W2-chain inference constant control data
mcmc_iterations = mcmc_burnin + n_samples_const_ctrl*mcmc_thin
const_control_char_event = run_w2_chain(
                                init_posterior, α0; λ0=λ0,
                                iterations=mcmc_iterations,
                                burnin=mcmc_burnin,
                                thin=mcmc_thin,
                                rng=mcmc_rng
                            )


#====================================================================================
    PLOTTING THE SAMPLES
====================================================================================#

f = plot(
    title="Absolute Error by Sample",
    titlefontsize=20,
    xlabel=L"Sample Index $k$",
    xguidefontsize=18,
    ylabel=L"|\alpha_i^{(k)} - \hat{\alpha}_i| / |\hat{\alpha}_i|",
    yguidefontsize=18,
    tickfontsize=12,
    left_margin = 2mm,
    legendfontsize=15,
    dpi=512, size=(800,600)
)

αhat = [ω1, ω2, ξ12]
labels = [ L"$ω_1$ $(i = 1)$", L"$ω_2$ $(i = 2)$", L"$ξ_{12}$ $(i = 3)$"]

samples = const_control_char_event.samples
for i = 1:3
    errors = abs.(samples[i,:] .- αhat[i]) 
    scatter!(errors, label=labels[i], alpha=0.75, yscale=:log10)
end