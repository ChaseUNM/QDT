using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions, JLD2, Printf
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
nsteps_2q_gate = round(Int, T_2q_gate/dt)
max_control_amplitude = 0.1
ipopt_options = ["max_iter" => 50, 
                 "print_level" => 5, 
                 "limited_memory_max_history" => 250
                ]
# Number of samples to use for optimization
n_samples_opt_1q = 5

# Gate set
one_qubit_gates  = [PauliX, PauliZ]
two_qubit_gates  = [CNOT12, CNOT21, CZ]


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


# Number of characterization+optimization iters to perform
max_iters = 2


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
    OPTIMIZE SINGLE-QUBIT GATES
====================================================================================#

controller_1q_gate = FortranBSplineControl(degree, n_splines, T_1q_gate)
N_coeff_1q_gate = controller_1q_gate.N_coeff

n_qubits   = 2
N_1q_gates = length(one_qubit_gates)
opt_events_1q_gate  = Array{OptimizationEvent,3}(undef, max_iters, n_qubits, N_1q_gates)
obs_events_1q_gate  = Array{ObservationEvent,3}(undef,  max_iters, n_qubits, N_1q_gates)
char_events_1q_gate = Vector{CharacterizationEvent}(undef, max_iters+1)
char_events_1q_gate[1] = const_control_char_event

# Initial, random control coefficients ("betas") for each gate
control_coeffs = (0.5 .- rand(N_coeff_1q_gate,n_qubits,N_1q_gates))*max_control_amplitude


for i in 1:max_iters

    @printf("SINGLE-QUDIT GATES | OPTIMIZATION + RE-CHARACTERIZATION LOOP, ITER %d\n", i)

    for q = 1:n_qubits
        @printf("\n === QUBIT %d ===\n", q)

        # Set the digital qudit to use the parameter samples generated during
        # the previous characterization event
        ns = size(char_events_1q_gate[i].samples,2)
        selected_samples = rand(1:ns, n_samples_opt_1q)
        set_parameters(digital_q, 
            char_events_1q_gate[i].samples[q,selected_samples], 
            zeros(n_samples_opt_1q) # no self-kerr because we restruct to qu*b*its
        )

        # Optimize each of the gates
        for j = 1:N_1q_gates
            @printf("  Optimizing gate %s ...\n", string(one_qubit_gates[j]))

            # Run the optimization loop
            opt_events_1q_gate[i,q,j] = optimize_control(
                                            digital_q, controller_1q_gate, 
                                            control_coeffs[:,q,j], 
                                            one_qubit_gates[j],
                                            max_amplitude=max_control_amplitude,
                                            options=ipopt_options
                                        )
            control_coeffs[:,q,j] = opt_events_1q_gate[i,q,j].control_coeffs

            # Evaluate the optimized controls on the PHYSICAL QUBIT PAIR
            @printf("  ... Evaluating gate %s\n", string(one_qubit_gates[j]))
            obs_events_1q_gate[i,q,j] = run_control(
                                            phys_device, 
                                            controller_1q_gate, 
                                            control_coeffs[:,q,j], 
                                            n_readout_samples; 
                                            target_gate=one_qubit_gates[j],
                                            which_qubit=q
                                        )
            @printf("  ... Measured Infidelity = %.2e\n", 
                    obs_events_1q_gate[i,q,j].measured_infidelity)
        end

    end

    # Check termination condition: both measured infidelities below epsilon for all gates
    if all([obs_events_1q_gate[i,q,j].measured_infidelity < epsilon for q = 1:n_qubits 
                                                            for j = 1:N_1q_gates])
        @printf("TERMINATING, ALL MEASURED FIDELITIES BELOW ϵ = %.2e\n", epsilon)
        break 
    end

    @printf("  Begining re-characterization post optimization ... \n")

    # Build a new Truncated Gaussian prior from the most recently generated 
    # parameter samples
    prior = TructGaussianPrior(char_events_1q_gate[i].samples, param_domain, downweight_power)

    # New posterior based on the new prior and new observations
    posterior = W2Posterior(digital_q_pair, vec(obs_events[i,:,:]), prior)

    @printf("  ... Running MCMC to sample new posterior\n")
    char_events_1q_gate[i+1] = run_w2_chain(
                                    posterior, prior.μ; λ0=λ0,
                                    iterations=mcmc_iterations,
                                    burnin=mcmc_burnin,
                                    thin=mcmc_thin, rng=mcmc_rng
                                )
    @printf("  ... Done.\n")
end


#====================================================================================
    OPTIMIZE TWO-QUBIT GATES
====================================================================================#

controller_2q_gate = [
    FortranBSplineControl(degree, n_splines, T_2q_gate),
    FortranBSplineControl(degree, n_splines, T_2q_gate)
]