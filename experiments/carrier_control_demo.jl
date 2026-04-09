using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
include("../src/digital_qudit.jl")

################################################################# 
# PARAMETERS
#################################################################

# Device 
Ne = 2
Ng = 0
omega = 4.5
xi = 0.2

# Controls
gate = PauliX
T_gate = 50
degree = 2
n_splines = 10
carrier_freqs = [0, xi, 2*xi]

#################################################################
# SETUP
#################################################################

# Create a qudit
q = DigitalQudit(Ne, Ng)
add_param_samples(q, omega, xi)

# Create a control for the gate
base_control = FortranBSplineControl(degree, n_splines, T_gate)
control = CarrierControl(base_control, carrier_freqs)
add_control(q, gate, control)
# Infidelity for this gate
q.infidelity[gate] = History(Float64)


#################################################################
# CONTROL OPTIMIZATION
#################################################################

# Optimize the gate
optimize_control(q, gate)

# Testing the control
dt = 0.005
Psi = run_control(q, 
                  q.controls[gate], 
                  dt=dt)
Psi = Psi[1,:,:]
foreach(normalize!, eachcol(Psi))
U = unitary(gate, Ne)
predicted_infidelity = infidelity(Psi, U, Ne)
@printf("Gate infidelity: %.3e\n", predicted_infidelity)
