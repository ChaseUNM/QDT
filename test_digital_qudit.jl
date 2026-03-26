using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
include("src/digital_device.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device 
Ne = 2
Ng = 0
ω = 4.5
ω_stdev = 0.001
ξ = 0.0
ξ_stdev = 0.0
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

############################################################################# 
# SETUP
#############################################################################

iteration = 0

# Create a Qudit
q = DigitalQudit(Ne, Ng)

# Initial parameter samples
ω_samples = ω .+ ω_stdev * randn(n_samples)
ξ_samples = ξ .+ ξ_stdev * randn(n_samples)
add_param_samples(q, iteration, ω_samples, ξ_samples)

# Create a control for each gate
for i = 1:N_gates
    # Control for this gate
    control_obj = FortranBSplineControl(degree, n_splines, T_gate)
    qcontrol = QuditControl(control_obj)
    q.controls[gates[i]] = qcontrol
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Float64)
end

# Optimize the controls
for i = 1:N_gates
    optimize_control(q, gates[i])
end