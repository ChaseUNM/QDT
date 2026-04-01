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
omega2 = 4.55
xi1 = 0.2
xi2 = 0.22
xi12 = 1e-5
J12  = 2.3E-3

# Controls
gate = CNOT
T_gate = 400
degree = 2
n_splines = 10
n_iters_opt = 100

#################################################################
# SETUP
#################################################################

# Create two qudits
q1 = DigitalQudit(Ne, Ng)
q2 = DigitalQudit(Ne, Ng)
# Set their parameter samples
add_param_samples(q1, omega1, xi1)
add_param_samples(q2, omega2, xi2)

# Put the qubits into a pair 
pair = DigitalQuditPair(q1, q2)
# Set their coupling values 
add_param_samples(pair, xi12, J12) 

# Create a control for the CNOT gate
# Control for this gate
q1_control = FortranBSplineControl(degree, n_splines, T_gate)
q2_control = FortranBSplineControl(degree, n_splines, T_gate)
add_control(pair, CNOT, q1_control, q2_control)
# Infidelity for this gate
pair.infidelity[CNOT] = History(Float64)


#################################################################
# CONTROL OPTIMIZATION
#################################################################

# Optimize the CNOT gate
optimize_control(pair, CNOT)

# Testing the control
dt = 0.005
Psi = run_control(pair, 
                  pair.controls[CNOT][1], 
                  pair.controls[CNOT][2], 
                  dt=dt)
Psi = Psi[1,:,:]
foreach(normalize!, eachcol(Psi))
U = unitary(CNOT, [1,2], [q1.Ne, q2.Ne], [q1.Ng, q2.Ng])
predicted_infidelity = infidelity(Psi, U, q1.Ne*q2.Ne)
@printf("Gate infidelity: %.3e\n", predicted_infidelity)
