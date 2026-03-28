using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
include("src/digital_device.jl")
include("src/physical_device.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device 
Ne = 2
Ng = 0
omega = 4.5
omega_bias = 0.02
omega_stdev = 0.001
xi = 0.0
xi_bias = 0.0
xi_stdev = 0.0
n_samples = 20

# Gates
gates = [PauliX, PauliY, PauliZ]
N_gates = length(gates)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 6
n_iters_opt = 100
seed = 3141

# Physical Qudit
M_spam_order = 1e-3
n_readout_samples = 2000

############################################################################# 
# SETUP
#############################################################################

# Create parameter sampler
omega_sampler = Normal(omega + omega_bias, omega_stdev)
xi_sampler    = Normal(xi + xi_bias, xi_stdev)

# Create the "physical device"
phys_q = PhysicalQudit(
            Ne, Ng, 
            omega, xi,
            M_spam_order=M_spam_order
         )

# Create a Qudit
q = DigitalQudit(Ne, Ng)
N = Ne + Ng

# Initial parameter samples
omega_samples = sort(rand(omega_sampler, n_samples))
xi_samples = sort(rand(xi_sampler, n_samples))
add_param_samples(q, omega_samples, xi_samples)

# Create a control for each gate
for i = 1:N_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Float64)
end

############################################################################# 
# CONTROL OPTIMIZATION
#############################################################################

# Optimize the controls
for i = 1:N_gates
    optimize_control(q, gates[i])
end

# Fidelity for each parameter sample
dt = 0.01
predicted_infidelity = zeros(N_gates, n_samples)
for i = 1:N_gates
    Psi = run_control(q, q.controls[gates[i]], dt=dt)
    U = unitary(gates[i], Ne)
    for j = 1:n_samples
        Psi_j = Psi[j,:,:]
        Psi_j = Psi_j ./ norm.(eachcol(Psi_j))
        predicted_infidelity[i,j] = infidelity(Psi_j, U, Ne)
    end
end

############################################################################# 
# MEASURED FIDELITY
#############################################################################

dt = 0.01
measured_state_infidelity = zeros(N_gates)
measured_population_infidelity = zeros(N_gates)
for i = 1:N_gates
    s_inf, p_inf = measure_infidelity(
                        phys_q,
                        gates[i],
                        q.controls[gates[i]],
                        n_readout_samples,
                        dt=dt
                   ) 
    measured_state_infidelity[i] = s_inf
    measured_population_infidelity[i] = p_inf         
end