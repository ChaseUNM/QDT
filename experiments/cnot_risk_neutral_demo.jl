using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
include("../src/digital_qudit.jl")
include("../src/digital_device.jl")

################################################################# 
# PARAMETERS
#################################################################

# Physical Device 
Ne = 2
Ng = 0
omega1 = 4.5
omega2 = 4.8
xi1    = 0.21
xi2    = 0.23
xi12   = 0.1  # Artificially large to allow fast coupling. Actual value: 1e-6 
J12    = 0.0  # REMOVED DUE TO OMISSION IN IMPLEMENTATION!

# Parameter Uncertainies
omega1_stdev = 0.0001 * omega1
omega2_stdev = 0.0001 * omega2
xi1_stdev  = 0.000 * xi1
xi2_stdev  = 0.000 * xi2
xi12_stdev = 0.000 * xi12
J12_stdev  = 0.000 * J12
n_samples  = 20

# Controls
gate = CNOT
T_gate = 550
degree = 3
n_splines = 10
n_iters_warmup = 32
n_iters_opt = 100


#################################################################
# SETUP
#################################################################

# Create parameter samplers
omega1_sampler = Normal(omega1, omega1_stdev)
omega2_sampler = Normal(omega2, omega1_stdev)
xi1_sampler   = Normal(xi1,  xi1_stdev)
xi2_sampler   = Normal(xi2,  xi2_stdev)
xi12_sampler  = Normal(xi12, xi12_stdev)
J12_sampler   = Normal(J12,  J12_stdev)

# Parameter samples coming from "characterization"
omega1_samples = rand(omega1_sampler, n_samples)
omega2_samples = rand(omega2_sampler, n_samples)
xi1_samples   = rand(xi1_sampler, n_samples)
xi2_samples   = rand(xi2_sampler, n_samples)
xi12_samples  = rand(xi12_sampler, n_samples)
J12_samples   = rand(J12_sampler, n_samples)

# Create two digital qudits
q1 = DigitalQudit(Ne, Ng)
q2 = DigitalQudit(Ne, Ng)

# Put the qubits into a pair 
pair = DigitalQuditPair(q1, q2)

# Create the controls for the CNOT gate
base_control = FortranBSplineControl(degree, n_splines, T_gate)
carrier_freqs = [0, -mean(xi12_samples), -2*mean(xi12_samples)]
q1_control = CarrierControl(base_control, carrier_freqs)
q2_control = CarrierControl(base_control, carrier_freqs)
add_control(pair, CNOT, q1_control, q2_control)
# Infidelity for this gate
pair.infidelity[CNOT] = History(Float64)


#################################################################
# WARM-UP
#
# Optimize the controls at the parameter samples closest to the sample mean
#################################################################

@printf("======== CONTROL WARM-UP ========\n")

n_samples_warmup = 20

# Set the digital qudits parameters to sample closest to the 
# sample mean in terms of frequencies. These parameters seem
# to be the most important in terms of control optimization.
#
# Selecting an index
omega_samples = [omega1_samples omega2_samples];
omega_mean = [mean(omega1_samples) mean(omega2_samples)]
dist_to_mean = norm.(eachrow(omega_samples .- omega_mean))
order = sortperm(dist_to_mean)
i1 = order[1:n_samples_warmup]
#
# Setting the parameters
add_param_samples(q1, omega1_samples[i1], xi1_samples[i1])
add_param_samples(q2, omega2_samples[i1], xi2_samples[i1])
add_param_samples(pair, xi12_samples[i1], J12_samples[i1])
#
# Set the rotating frequencies of the qubits to the sample means
q1.omega_rot = omega_mean[1]
q2.omega_rot = omega_mean[2]

# Optimize the control for this point estimate of the parameters
optimize_control(pair, CNOT, options=[
                                "max_iter" => n_iters_warmup, 
                                "print_level" => 5, "limited_memory_max_history" => 250
                              ])

# Set the digital device parameters to store the samples
update_param_samples(q1, omega1_samples, xi1_samples)
update_param_samples(q2, omega2_samples, xi2_samples)
update_param_samples(pair, xi12_samples, J12_samples) 

# Fidelity for each parameter sample
dt = 0.05
Psi = run_control(pair, 
                  pair.controls[CNOT][1], 
                  pair.controls[CNOT][2], 
                  dt=dt)
infidelity_post_warmup = zeros(n_samples)
U = unitary(CNOT, Ne*Ne)
for j = 1:n_samples
    Psi_j = Psi[j,:,:]
    foreach(normalize!, eachcol(Psi_j))
    infidelity_post_warmup[j] = infidelity(Psi_j, U, Ne*Ne)
end



# #################################################################
# # RISK NEUTRAL OPTIMIZATION
# #################################################################

# @printf("\n\n======== RISK NEUTRAL CONTROL ========\n")

# # Set the digital device parameters to store the samples
# update_param_samples(q1, omega1_samples, xi1_samples)
# update_param_samples(q2, omega2_samples, xi2_samples)
# update_param_samples(pair, xi12_samples, J12_samples) 

# # Optimize the controls in the risk neutral setting.
# # This will start from the control signals determined 
# # during the warm-up period
# optimize_control(pair, CNOT, 
#                  options=[
#                     "max_iter" => n_iters_opt, 
#                     "print_level" => 5, "limited_memory_max_history" => 250
#                  ]
#                 )

# # Fidelity for each parameter sample
# dt = 0.005
# Psi = run_control(pair, 
#                   pair.controls[CNOT][1], 
#                   pair.controls[CNOT][2], 
#                   dt=dt)
# predicted_infidelity = zeros(n_samples)
# U = unitary(CNOT, Ne+Ng)
# for j = 1:n_samples
#     Psi_j = Psi[j,:,:]
#     foreach(normalize!, eachcol(Psi_j))
#     predicted_infidelity[j] = infidelity(Psi_j, U, Ne*Ne)
# end