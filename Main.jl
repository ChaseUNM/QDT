using LinearAlgebra, Plots, QuantumGateDesign, Random, Distributions
include("src/qdt.jl")
include("src/measurement.jl")

true_parameter = 4.5
omega_rot = true_parameter

U0 = [1 0; 0 1]
N = 1 
Ne = [2]
Ng = [0]
x_gate = [0 1.0; 1.0 0]
y_gate = [0 im; -im 0]
z_gate = [1.0 0; 0 -1.0]
gate_set = [x_gate, y_gate, z_gate]
N_gates = length(gate_set)
Mspam_order = 5E-3
Twin = QDT(N, Ne, Ng, omega_rot, gate_set, Mspam_order)

maximum_size = 1000

center = 4.5
std_dev = 0.1
param_samples = Normal(center, std_dev)
omega_rot = center
Twin.param_samples = rand(param_samples, Twin.n_param_samples) 



std_dev = 0.1
param_samples = Normal(true_parameter, std_dev)
Twin.param_samples = rand(param_samples, Twin.n_param_samples)
optimize_controls(Twin)
# optimize_controls(Twin)
# measured_infidelity(Twin, true_parameter)
# save_state(Twin, history, true_parameter)

#Case where characterize and then optimize
history = qdt_history(maximum_size, N_gates)
x_range = zeros(maximum_size)
for i in 1:maximum_size 
    omega = true_parameter + 0.3*sin(2*pi*i/maximum_size)
    x_range[i] = omega
    # global true_parameter += (-1)^i*0.01*i
    std_dev = 0.0
    measured_infidelity(Twin, omega)
    save_state(Twin, history, omega)
end

cols = palette(:default)
measured_population_infidelity_plot = plot(x_range, history.measured_population_infidelity', yscale =:log10, labels = ["Meas: Gate 1" "Meas: Gate 2" "Meas: Gate 3"], ylabel = "Population Infidelity", color=[cols[1] cols[2] cols[3]], yticks = [1e-3, 1e-2, 1e-1, 1e0])
plot!(x_range, history.predicted_infidelity', yscale =:log10, labels = ["Pred: Gate 1" "Pred: Gate 2" "Pred: Gate 3"], legend_columns = 2, legend=:outertop, linestyle =:dash, color=[cols[1] cols[2] cols[3]])
vline!(Twin.param_samples, alpha = 0.5, color =:grey)
savefig("/Results/RN_plot_std_dev_$(std_dev).png")
# measured_state_infidelity_plot = plot(history.measured_state_infidelity', yscale=:log10, labels = ["Gate 1" "Gate 2" "Gate 3"], ylabel = "State Infidelity")



# mutable struct PhysicalDevice 
#     transition_freq::
#     self_kerr::
#     zz_coupling::
#     dipole_dipole::

    



# mutable struct qdt
#     #inputs:
#     N::Int64 #Number of subsystems 
#     Ne::Vector{Int64} #Number of essential energy levels in each subsystem
#     Ng::Vector{Int64} #Number of guard energy levels in each subsystem 
#     transition_samples::Vector{Float64} #Samples of Transition frequencies
#     degree::Int64 = 2
#     nsplines::Int64 = 2
#     T::Float64 #Duration of gate (same T for each gate for now)
#     steps::Int64 #steps for numerical integration 

#     #create schrodinger prob: 
#     probs = [SchrodingerProb([0 0; 0 i], real_control_ops, imag_control_ops, U0, tf, nsteps) for i in samples]
#     #now do Risk neutral optimization 
#     control = FortranBSplineControl(degree, nsplines, steps)
#     max_control_parameter = 0.1
#     pcof_l = -max_control_parameter
#     pcof_u = max_control_parameter
#     pcof0 = (0.5 .- rand(control.N_coeff)) .* max_control_parameter 
#     opt_ret = optimize_prob(probs, control, pcof0, hadamard_gate, pcof_lbound=pcof_l, pcof_ubound=pcof_u, cost_type=:Infidelity, ipopt_options=["max_iter" => 100])
#     pcof_optimal = opt_ret.x 
#     #get overall predicted and infidelity individual predicted infidelity




#     #nsplines will be coming from DM 
#     #specify initial control pcof (begin with random but think about)
#     #call optimize with transition samples, with specified control parameters
#     #Pass in samples of transition frequencies



#     #in some way need to feed in the model hamiltonian (whether that's just parameters or a matrix)
#     #need samples, can either pass in a vector of samples or a pre-specified distribution 
#     #probably want to determine structure of control pulse (degree, # of splines)
#     #for functions want to add functions like a simulate and optimize function (both of these will just call QuantumGateDesign.jl)
#     #set of gates to optimize(at most 2 qubit gates)
#     #pass in number of subsystems, energy levels for each qudit (essential and computational)
#     #

#     #returns: 
#     #want history of population of states
#     #want history of how parameters got updated
#     #

# mutable struct PhysicalTwin 
#     #in some way feed in the model hamiltonian (whether that's just parameters or a matrix)
#     #don't need samples (this is deterministic)
#     #