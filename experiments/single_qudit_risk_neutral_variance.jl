using LinearAlgebra, QuantumGateDesign, Random, Distributions, Printf
using Plots, Plots.Measures, LaTeXStrings, Plots.PlotMeasures
using JLD2
include("../src/digital_qudit.jl")
include("../src/util.jl")

############################################################################# 
# PARAMETERS
#############################################################################

# Device 
Ne = 2
Ng = 2
omega = 4.5
omega_bias = 0.00
omega_stdev = 0.001 * omega
xi = 0.2 # no xi variance
n_samples = 100

# Gates
gates = [PauliX, PauliY, PauliZ]
n_gates = length(gates)
T_gate = 15 * pi

# Controls
degree = 2
n_splines = 8
n_iters_opt = 100

# Range of omega values over which to eval fidelity
n_omega_eval = 100
d_omega = 1.5 * omega_stdev  
omega_min = omega - d_omega
omega_max = omega + d_omega
dt_eval_fidelity = 0.05

# Number of times at which to evaluate the control functions
# for plotting
n_t_eval = 256


n_trials = 2 # number of trials to perform

seed = 1234 # rng seed

############################################################################# 
# SETUP
#############################################################################

# Set the seed for the RNG
Random.seed!(seed)

# Frequency parameter sampler
omega_sampler = Normal(omega + omega_bias, omega_stdev)

# Range of omegas over which to eval infidelities
omegas = collect(LinRange(omega_min,omega_max,n_omega_eval))
xi_eval = xi * ones(n_omega_eval) # No variance in self-kerr

# Create a Qudit
q = DigitalQudit(Ne, Ng)
N = Ne + Ng
add_param_samples(q, omega, xi) # Initializes the param hist

# Create a control for each gate
for i = 1:n_gates
    # Control for this gate
    qcontrol = FortranBSplineControl(degree, n_splines, T_gate)
    add_control(q, gates[i], qcontrol)
    # Infidelities for this gate
    q.infidelity[gates[i]] = History(Vector{Float64})
end


############################################################################# 
# RUN EXPERIMENT
#############################################################################

controls = zeros(n_trials, n_gates, 2, n_t_eval)
infidelities = zeros(n_trials, n_gates, n_omega_eval)
omega_samples = zeros(n_samples)

for t = 1:n_trials

    t_start = time()
    @printf("\n==== TRIAL %d of %d ==== \n", t, n_trials)

    # Parameter samples
    omega_samples .= sort(rand(omega_sampler, n_samples))
    xi_samples = xi * ones(n_samples)
    update_param_samples(q, omega_samples, xi_samples)

    for i = 1:n_gates
        @printf("  Optimizing Gate %d of %d\n", i, n_gates)
        # Randomize the controls
        randomize_coeffs(q.controls[gates[i]])
        # Optimize the gate
        optimize_control(q, gates[i], 
                            options=["max_iter" => n_iters_opt, "print_level" => 0, "limited_memory_max_history" => 250])
        # Plot the control signal in the rotating frame
        control_obj = q.controls[gates[i]].objs.values[end]
        control_coeffs = q.controls[gates[i]].coeffs.values[end]
        controls[t,i,1,:] = [
                eval_p_derivative(control_obj, time, control_coeffs, 0)
                for time in LinRange(0, T_gate, n_t_eval)
            ]
        controls[t,i,2,:] = [
                eval_q_derivative(control_obj, time, control_coeffs, 0)
                for time in LinRange(0, T_gate, n_t_eval)
            ]
    end

    # Set the parameter samples to be evenly spaced between
    # omega_min and omega_max

    update_param_samples(q, omegas, xi_eval)
    q.omega_rot = mean(omega_samples) # Use the same rotatining frame frequency as the initial set of samples, as this is the frame in which the controls where optimized in

    # Measure fidelity over grid of omega values
    for i = 1:n_gates
        @printf("  Evaluating Gate %d of %d\n", i, n_gates)
        U = unitary(gates[i], Ne+Ng) 
        Psi = run_control(
                q, 
                q.controls[gates[i]], 
                dt=dt_eval_fidelity
              )
        for k = 1:n_omega_eval
            Psi_k = Psi[k,:,:]
            foreach(normalize!, eachcol(Psi_k))
            infidelities[t,i,k] = infidelity(Psi_k, U, Ne)
        end
    end
    t_end = time()
    @printf("Done. Runtime = %.0f s\n", t_end-t_start)
end

############################################################################# 
# SAVE DATA TO FILE
#############################################################################

output_dir = "./data/single_qudit_risk_neutral_variance"
mkpath(output_dir)

filename = @sprintf(
    "N%d+%d_samples%d_trials%d_seed%d.jld2",
    Ne, Ng, n_samples, n_trials, seed
)
full_path = joinpath(output_dir, filename)

jldsave(full_path; omega, xi, omegas, controls, infidelities)


############################################################################# 
# PLOT INFIDELITIES
#############################################################################

c_min = 1e-5
c_max = 1.0

# Plot Infidelities vs parameter value
subplots = Any[]
gate_names = ["X", "Y", "Z"]
for i = 1:n_gates
    title_str = @sprintf(
                    "Fidelity by Trial, \$%s\$ Gate", 
                    gate_names[i]
                )
    p_i = plot(
            title= LaTeXString(title_str), 
            titlefontsize=16, 
            xlabel=L"Frequency Error $(\omega-\omega_*)/\omega_*$", xlabelfontsize=14,
            ylabel="Infidelity (Predicted)", ylabelfontsize=14, 
            size=(600,500), dpi=512, 
            ylim=(c_min,c_max), yscale=:log10,
            right_margin = 10mm
        )
    for t = 1:n_trials
        plot!(
            (omegas .- omega) ./ omegas,
            infidelities[t,i,:],
            color=:blue, alpha=0.5, legend=false
        )
    end

    push!(subplots, p_i)

end

f1 = plot(
    subplots[1], subplots[2], subplots[3],
    layout=(1,3), size=(1500,500), dpi=512, 
    bottom_margin=10mm, top_margin=10mm
)


# Create figure output directory
fig_dir = "./figures/single_qudit_risk_neutral_variance"
mkpath(fig_dir)

# Saving to file
file_prefix = @sprintf(
    "N%d+%d_samples%d_trials%d_seed%d",
    Ne, Ng, n_samples, n_trials, seed
)
fig_path = joinpath(fig_dir, file_prefix * "_infidelities.svg")
savefig(f1, fig_path)



############################################################################# 
# PLOT CONTROL SIGNALS
#############################################################################

# Plot Infidelities vs parameter value
control_plots = Any[]
gate_names = ["X", "Y", "Z"]
for i = 1:n_gates
    title_str = @sprintf(
                    "Controls by Trial, \$%s\$ Gate", 
                    gate_names[i]
                )
    p_i = plot(
            title= LaTeXString(title_str), 
            titlefontsize=16, 
            xlabel=L"Time $t$", xlabelfontsize=14,
            ylabel="Control Amplitude, p(t) or q(t)", ylabelfontsize=14, 
            size=(600,500), dpi=512, 
            ylim=(-0.1,0.1),
            right_margin = 10Plots.mm
        )
    for t = 1:n_trials
        plot!(
            LinRange(0, T_gate, n_t_eval),
            controls[t,i,1,:],
            color=:blue, alpha=0.5, legend=false
        )
        plot!(
            LinRange(0, T_gate, n_t_eval),
            controls[t,i,2,:],
            color=:red, alpha=0.5, legend=false
        )
    end

    push!(control_plots, p_i)

end


f2 = plot(
    control_plots[1], control_plots[2], control_plots[3],
    layout=(1,3), size=(1500,500), dpi=512,
    bottom_margin=10mm, top_margin=10mm
)

fig_path = joinpath(fig_dir, file_prefix * "_controls.svg")
savefig(f2, fig_path)
