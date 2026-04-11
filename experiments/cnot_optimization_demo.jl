using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
include("../src/digital_qudit.jl")
include("../src/digital_device.jl")

################################################################# 
# PARAMETERS
#################################################################

# Device 
Ne = 2
Ng = 2
omega1 = 4.5
omega2 = 4.8
xi1    = 0.21
xi2    = 0.23
xi12   = 0.1 # Artificially large to allow fast coupling. Actual value: 1e-6 
J12    = 0.0 # 2*pi * 2.3E-3

# Controls
gate = CNOT
T_gate = 550
degree = 2
n_splines = 6
# carrier_freqs = [0, xi12, 2*xi12]
carrier_freqs = [0, xi12, 2*xi12]

# Optimization params
n_iters_opt = 100
dt_opt = 0.2


#################################################################
# SETUP
#################################################################

# Create two qudits
q1 = DigitalQudit(Ne, Ng)
q2 = DigitalQudit(Ne, Ng)
# Set their parameter vlaues
add_param_samples(q1, omega1, xi1)
add_param_samples(q2, omega2, xi2)
# Set them to a shared rotating frame
# omega_rot = 0.5*(q1.omega_rot+q2.omega_rot)
# q1.omega_rot = omega_rot
# q2.omega_rot = omega_rot

# Put the qubits into a pair 
pair = DigitalQuditPair(q1, q2)
# Set their coupling values 
add_param_samples(pair, xi12, J12) 

# Create the controls for the CNOT gate
base_control = FortranBSplineControl(degree, n_splines, T_gate)
q1_control = CarrierControl(
                base_control, 
                (omega1-q1.omega_rot) .- carrier_freqs
             )
q2_control = CarrierControl(
                base_control, 
                (omega2-q2.omega_rot) .- carrier_freqs
             )
add_control(pair, CNOT, q1_control, q2_control)
# Infidelity for this gate
pair.infidelity[CNOT] = History(Float64)


#################################################################
# CONTROL OPTIMIZATION
#################################################################

# Optimize the CNOT gate
optimize_control(pair, CNOT, dt=dt_opt, 
                options=["max_iter" => n_iters_opt, "print_level" => 5, "limited_memory_max_history" => 250])

# Testing the control
dt = 0.005
Psi = run_control(pair, 
                  pair.controls[CNOT][1], 
                  pair.controls[CNOT][2], 
                  dt=dt)
Psi = Psi[1,:,:]
foreach(normalize!, eachcol(Psi))
U = unitary(CNOT, [1,2], [q1.Ne+q1.Ng, q2.Ne+q2.Ng], [q1.Ne, q2.Ne])
predicted_infidelity = infidelity(Psi, U, q1.Ne*q2.Ne)
@printf("Gate infidelity: %.3e\n", predicted_infidelity)
