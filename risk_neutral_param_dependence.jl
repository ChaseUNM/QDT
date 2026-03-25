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
n_ω_samples = 1
ω_stdev = 0.1

# Range of frequencies to show in RN plots
n_ω_plot = 501
Δω = 0.4
ω_range = [ω_true - Δω, ω_true + Δω]


#############################################################################
# RUN EXPERIMENT
#############################################################################

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
optimize_controls(Twin, ipopt_options=ipopt_options)

# Measure fidelity at different omega values
ωs = LinRange(ω_range[1], ω_range[2], n_ω_plot)
measured_population_infidelity = zeros(n_ω_plot,N_gates)
measured_state_infidelity = zeros(n_ω_plot,N_gates)
for i in 1:n_ω_plot
    measured_infidelity(Twin, ωs[i], add_SPAM=false)
    measured_population_infidelity[i,:] = Twin.measured_population_infidelity;
    measured_state_infidelity[i,:] = Twin.measured_state_infidelity;
end

# Measure fidelity at the samples
measured_state_infidelity_at_samples = zeros(Twin.n_param_samples,N_gates)
for i in 1:Twin.n_param_samples
    measured_infidelity(Twin, Twin.param_samples[i], add_SPAM=false)
    measured_state_infidelity_at_samples[i,:] = Twin.measured_state_infidelity;
end



#############################################################################
# PLOT RESULTS
#############################################################################

cols = palette(:default)
measured_population_infidelity_plot = plot(
    ωs, 
    measured_state_infidelity, 
    yscale =:log10, 
    labels = ["Meas: Gate 1" "Meas: Gate 2" "Meas: Gate 3"], 
    ylabel = "State Infidelity", 
    color=[cols[1] cols[2] cols[3]], 
    dpi=256
    )
hline!(
    reshape(Twin.predicted_infidelity,1,N_gates) /n_ω_samples, 
    labels = ["Pred: Gate 1" "Pred: Gate 2" "Pred: Gate 3"], 
    legend_columns = 2, 
    legend=:outertop, linestyle =:dash, 
    color=[cols[1] cols[2] cols[3]]
)
scatter!(
    Twin.param_samples, 
    measured_state_infidelity_at_samples, 
    yscale =:log10, color=[cols[1] cols[2] cols[3]],
    markersize=1, labels=[]
)
vline!(Twin.param_samples, alpha = 0.5, color =:grey)

# savefig("/Results/RN_plot_std_dev_$(std_dev).png")


