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
max_control_amplitude = 0.1
ipopt_options = ["max_iter" => 50, 
                 "print_level" => 5, 
                 "limited_memory_max_history" => 250
                ]
# Number of samples to use for optimization
n_samples_opt_1q = 10

# Gate set
one_qubit_gates  = [Identity, PauliX, PauliZ]


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
    OPTIMIZE SINGLE-QUBIT GATES
====================================================================================#

controller_1q_gate = FortranBSplineControl(degree, n_splines, T_1q_gate)
N_coeff_1q_gate = controller_1q_gate.N_coeff

n_qubits   = 2
N_1q_gates = length(one_qubit_gates)
opt_events_1q_gate  = Array{OptimizationEvent,2}(undef, n_qubits, N_1q_gates)
obs_events_1q_gate  = Array{ObservationEvent,2}(undef, n_qubits, N_1q_gates)

# Initial, random control coefficients ("betas") for each gate
control_coeffs = (0.5 .- rand(N_coeff_1q_gate,n_qubits,N_1q_gates))*max_control_amplitude


@printf("SINGLE-QUDIT GATE OPTIMIZATION\n")

for q = 1:n_qubits
    @printf("\n === QUBIT %d ===\n", q)

    # Set the digital qudit to use the parameter samples generated during
    # the previous characterization event
    ns = size(const_control_char_event.samples,2)
    selected_samples = rand(1:ns, n_samples_opt_1q)
    set_parameters(digital_q, 
        const_control_char_event.samples[q,selected_samples], 
        zeros(n_samples_opt_1q) # no self-kerr because we restruct to qu*b*its
    )

    # Optimize each of the gates
    for j = 1:N_1q_gates
        @printf("  Optimizing gate %s ...\n", string(one_qubit_gates[j]))

        # Run the optimization loop
        opt_events_1q_gate[q,j] = optimize_control(
                                        digital_q, controller_1q_gate, 
                                        control_coeffs[:,q,j], 
                                        one_qubit_gates[j],
                                        max_amplitude=max_control_amplitude,
                                        options=ipopt_options
                                    )
        control_coeffs[:,q,j] = opt_events_1q_gate[q,j].control_coeffs

        # Evaluate the optimized controls on the PHYSICAL QUBIT PAIR
        @printf("  ... Evaluating gate %s\n", string(one_qubit_gates[j]))
        obs_events_1q_gate[q,j] = run_control(
                                        phys_device, 
                                        controller_1q_gate, 
                                        control_coeffs[:,q,j], 
                                        n_readout_samples; 
                                        target_gate=one_qubit_gates[j],
                                        which_qubit=q
                                    )
        @printf("  ... Measured Infidelity = %.2e\n", 
                obs_events_1q_gate[q,j].measured_infidelity)
    end

end


#====================================================================================
    TEST CONTROLS WITH DIFFERENT ξ12

    Specifically for gates A ⊗ B where one of A or B is the identity
====================================================================================#

n_steps = 100
ξ12_range = LinRange(0, ξmax, n_steps)
single_gate_state_infidelity = zeros(n_qubits,N_1q_gates,n_steps)
single_gate_pop_infidelity   = zeros(n_qubits,N_1q_gates,n_steps)

# Loop over qubits
for q = 1:n_qubits
    @printf("\n === QUBIT %d ===\n", q)

    # Loop over gates
    for j = 1:N_1q_gates
        @printf("  Gate %s ...\n", string(one_qubit_gates[j]))

        # Loop over ξ12 settings
        for k = 1:n_steps

            # Set ξ12 for the physical qubit
            phys_device.device.xi = ξ12_range[k]

            # Run the qubit q's control for gate j for this setting of ξ12
            Ψ = run_control(
                        phys_device.device, 
                        controller_1q_gate, 
                        control_coeffs[:,q,j], 
                        which_qubit=q
                )[1,:,end,:]
            Ψ ./= norm.(eachcol(Ψ))
            U = unitary(phys_device.device, one_qubit_gates[j], which_qubit=q)
            single_gate_state_infidelity[q,j,k] = infidelity(U,Ψ,4)
            single_gate_pop_infidelity[q,j,k]   = infidelity_population(abs2.(Ψ).^2, abs2.(U))
        end
    end
end


#====================================================================================
    DEV
====================================================================================#

# Select a qubit and a gate
q = 1
g = 1

# Run a control on the physical device without ξ12 = 0.0 and no SPAM
phys_device.device.xi = 0.0
obs_2q = run_control(   phys_device.device, 
                        controller_1q_gate, 
                        control_coeffs[:,q,g],
                        which_qubit=q
                    )
Ψ_2q = obs_2q[1,:,end,:]

# Run the same control on a DigitalQudit
set_parameters(digital_q, q==1 ? ω1 : ω2, 0.0)
obs_1q = run_control(digital_q, 
                    controller_1q_gate, 
                    control_coeffs[:,q,g] )
Ψ_1q = obs_1q[1,:,end,:]

# Measuring infidelity
Id = I[1:2,1:2] 
inf_1q = infidelity(Ψ_1q, unitary(one_qubit_gates[q]), 2)
inf_2q = infidelity(Ψ_2q, unitary(phys_device.device, one_qubit_gates[q], which_qubit=q), 4)
state_dist = norm(Ψ_2q - kron(Ψ_1q,Id))
pop_dist = norm(abs.(Ψ_2q) - kron(abs.(Ψ_1q),Id))

# Printing the stats
@printf("|Ψ₁  - U₁ | = %.2e\n", inf_1q)
@printf("|Ψ₁₂ - U₁₂| = %.2e\n", inf_2q)
@printf("|Ψ₁  - Ψ₁₂| = %.2e\n", state_dist)
@printf("|abs.(Ψ₁₂) - abs.(Ψ₁₂)| = %.2e\n", pop_dist)



#====================================================================================
    TEST CONTROL WITH DIFFERENT ξ12
    
    This time, perform gates X₁ ⊗ X₂, X₁ ⊗ Z₂, etc 
====================================================================================#

n_steps = 100
ξ12_range = LinRange(0, ξmax, n_steps)
paired_gate_state_infidelity = zeros(N_1q_gates, N_1q_gates, n_steps)
paired_gate_pop_infidelity   = zeros(N_1q_gates, N_1q_gates, n_steps)

# Loop over pairs of gates
for j1 = 1:N_1q_gates
    for j2 = 1:N_1q_gates
        @printf("  Gate %s₁ ⊗ %s₂ ...\n", string(one_qubit_gates[j1]), 
                                            string(one_qubit_gates[j2]))

        # Loop over ξ12 settings
        for k = 1:n_steps

            # Set ξ12 for the physical qubit
            phys_device.device.xi = ξ12_range[k]

            # Run the qubit q's control for gate j for this setting of ξ12
            Ψ = run_control(phys_device.device, 
                        AbstractControl[controller_1q_gate, controller_1q_gate], 
                        [control_coeffs[:,1,j1]; control_coeffs[:,2,j2]], 
                    )[1,:,end,:]
            Ψ ./= norm.(eachcol(Ψ))'
            U₁ = unitary(one_qubit_gates[j1])
            U₂ = unitary(one_qubit_gates[j2])
            U₁₂ = kron(U₁,U₂)
            paired_gate_state_infidelity[j1,j2,k] = infidelity(Ψ, U₁₂, 4)
            paired_gate_pop_infidelity[j1,j2,k] = infidelity_population(abs2.(Ψ), abs2.(U₁₂))
        end
    end
end



#====================================================================================
    Plot some of the errors vs. ξ12 

====================================================================================#

state_infidelity_data = [
    single_gate_state_infidelity[1,1,:],
    single_gate_state_infidelity[2,1,:],
    single_gate_state_infidelity[2,2,:],
    paired_gate_state_infidelity[1,1,:],
    paired_gate_state_infidelity[1,2,:]
]

pop_infidelity_data = [
    single_gate_pop_infidelity[1,1,:],
    single_gate_pop_infidelity[2,1,:],
    single_gate_pop_infidelity[2,2,:],
    paired_gate_pop_infidelity[1,1,:],
    paired_gate_pop_infidelity[1,2,:]
]

plot_data = [state_infidelity_data, pop_infidelity_data]

labels = [
    L"$X_1$",
    L"$X_2$",
    L"$Z_2$",
    L"$X_1 X_1$",
    L"$X_1 Z_2$"
]

subplot_titles = [
    L"State Infidelity $\mathcal{F}(\Psi,\hat{U})$", 
    L"Population Infidelity $\mathcal{F}(\vert\Psi\vert^2,\vert\hat{U}\vert^2)$"
]

subplots = Vector{Any}(undef, 2)
for i = 1:2
    subplots[i] = plot(
        xlabel=L"Cross-Kerr $ξ_{12}$",
        xguidefontsize=18,
        title=subplot_titles[i],
        titlefontsize=20,
        tickfontsize=12,
        left_margin = 2mm,
        bottom_margin = 5mm,
        legendfontsize=15,
    )
    for j = eachindex(labels)
        plot!(ξ12_range, plot_data[i][j], linewidth=2, label=labels[j])
    end
end


f = plot(
    subplots[1], subplots[2], layout=(1,2),
    plot_title="Gate Infidelity vs. Cross-Kerr Strength",
    plot_titlefontsize=25, dpi=512, size=(1200,600), plot_titlevspan=0.1
)

savefig(f, "figures/gate_infidelity_vs_cross_kerr.svg")
