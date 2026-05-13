using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
include("src/qdt.jl")
include("src/measurement.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device
N = 1 
Ne = [2]
Ng = [0]
ω_true = 4.5
ω_rot = ω_true
Mspam_order = 5E-3

# Gates
x_gate = [0 1.0; 1.0 0]
y_gate = [0 im; -im 0]
z_gate = [1.0 0; 0 -1.0]
gate_set = [x_gate, y_gate, z_gate]
N_gates = length(gate_set)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 3
n_iters_opt = 100

# Device "characterization"
n_ω_samples = 10
ω_stdev = 0.01

# Make the QDT
Twin = QDT( N, Ne, Ng, ω_rot, 
            gate_set, Mspam_order, 
            n_param_samples=n_ω_samples,
            T_gate=T_gate, 
            control_degree=degree, control_nsplines=n_splines)

# Generate parameter samples
ω_samples = Normal(ω_true, ω_stdev)
Twin.param_samples = rand(ω_samples, Twin.n_param_samples) 

# Optimize the control for this sample set
ipopt_options = ["max_iter" => n_iters_opt, "print_level" => 3]
optimize_controls(Twin, ipopt_options=ipopt_options; ϵ₁ = 0.2)